"""Multi-LLM consensus and confidence scoring (v0.9 Step 5).

Given N LLM completion outputs for the same input, this module computes:
- **agreement_score**: how much the outputs agree on adjustment values
  (1 - normalized_range where range is max-min divided by the 0.2 possible span)
- **completeness**: fraction of outputs that produced valid adjustments
- **confidence**: agreement × completeness (0 → 1)
- **aggregated_adjustments**: mean of valid adjustment values (if confidence is high)

The consensus is purely deterministic — no randomness, no external calls.
Low confidence means "the LLMs disagree" and the result is an empty dict
(zero influence). High confidence means "the LLMs agree" and the mean
adjustment is accepted (subject to further validation).
"""

from __future__ import annotations

from statistics import mean
from typing import Any

from swarm.config import CONFIDENCE_THRESHOLD

#: The only adjustment fields recognized during consensus.
_CONSENSUS_FIELDS = frozenset({"price_weight_delta", "delivery_weight_delta"})

#: Minimum confidence for adjustments to be aggregated (vs. discarded).


def compute_llm_consensus(completions: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute consensus and confidence from N LLM completion outputs.

    Args:
        completions: List of LLM output dicts (each typically has
            ``suggested_adjustments``).

    Returns:
        A dict with:
        - ``confidence``: float in [0, 1] — overall trust in the LLM outputs
        - ``agreement_score``: float in [0, 1] — inter-model agreement level
        - ``completeness``: float in [0, 1] — fraction producing valid adjustments
        - ``aggregated_adjustments``: dict of accepted mean adjustments (or {})
        - ``num_completions``: int — total number of completion outputs
    """
    if not completions:
        return {
            "confidence": 0.0,
            "agreement_score": 0.0,
            "completeness": 0.0,
            "aggregated_adjustments": {},
            "num_completions": 0,
        }

    num = len(completions)

    # --- Step A: extract valid adjustments from each completion ---
    valid_adjustments_per_completion: list[dict[str, float]] = []
    for output in completions:
        if not isinstance(output, dict):
            continue
        proposed = output.get("suggested_adjustments")
        if not isinstance(proposed, dict):
            continue
        valid: dict[str, float] = {}
        for key, value in proposed.items():
            if key not in _CONSENSUS_FIELDS:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            # Clamp to the accepted delta range for consensus aggregation
            if -0.1 <= numeric <= 0.1:
                valid[key] = numeric
        if valid:
            valid_adjustments_per_completion.append(valid)

    # --- Step B: completeness factor ---
    completeness = len(valid_adjustments_per_completion) / num

    # --- Step C: agreement score per field ---
    all_fields = _CONSENSUS_FIELDS
    field_values: dict[str, list[float]] = {field: [] for field in all_fields}
    for adjustments in valid_adjustments_per_completion:
        for field in all_fields:
            if field in adjustments:
                field_values[field].append(adjustments[field])

    field_agreements: list[float] = []
    for _, values in field_values.items():
        if len(values) == 0:
            continue
        if len(values) == 1:
            field_agreements.append(1.0)
            continue
        # Agreement = 1 - normalized_range
        # Range is max - min; normalize by max possible range for ±0.1 bounds (=0.2)
        value_range = max(values) - min(values)
        agreement = max(0.0, 1.0 - (value_range / 0.2))
        field_agreements.append(agreement)

    agreement_score = 0.0 if not field_agreements else mean(field_agreements)

    # --- Step D: confidence = agreement × completeness ---
    confidence = agreement_score * completeness

    # --- Step E: aggregate (mean) if confidence is high enough ---
    aggregated: dict[str, float] = {}
    if confidence >= CONFIDENCE_THRESHOLD:
        for field in all_fields:
            values = field_values.get(field, [])
            if values:
                aggregated[field] = round(mean(values), 4)

    return {
        "confidence": round(confidence, 4),
        "agreement_score": round(agreement_score, 4),
        "completeness": round(completeness, 4),
        "aggregated_adjustments": aggregated,
        "num_completions": num,
    }
