"""LLM drift detection from consensus history (v0.9 Step 11).

Detects when the LLM consensus signal is drifting — e.g., confidence declining
rapidly, stability dropping below threshold, or sudden changes in agreement.
"""

from __future__ import annotations

from typing import Any

from swarm.config import CONFIDENCE_DROP_THRESHOLD
from swarm.utils.llm_stability import STABILITY_THRESHOLD, TRUST_THRESHOLD


def detect_drift(
    history: list[dict[str, Any]],
    stability_threshold: float | None = None,
    trust_threshold: float | None = None,
) -> tuple[bool, list[str]]:
    """Detect drift in LLM consensus signals across history records.

    Returns ``(drift_detected, reasons)`` where ``reasons`` is a list of
    human-readable strings describing each detected drift condition.

    Drift conditions checked:
    - Confidence dropped by more than ``CONFIDENCE_DROP_THRESHOLD`` between consecutive rounds.
    - Stability below ``STABILITY_THRESHOLD`` (or override) in the latest round.
    - Trust score below ``TRUST_THRESHOLD`` (or override) in the latest round.
    - Fewer than 2 completions available (low signal-to-noise).

    Args:
        history: List of consensus history records.
        stability_threshold: Optional adaptive threshold override for stability.
        trust_threshold: Optional adaptive threshold override for trust.
    """
    if not history:
        return False, []

    reasons: list[str] = []

    latest = history[-1]

    if len(history) >= 2:
        prev = history[-2]
        conf_drop = prev.get("confidence", 0.0) - latest.get("confidence", 0.0)
        if conf_drop > CONFIDENCE_DROP_THRESHOLD:
            reasons.append(f"confidence_drop:{round(conf_drop, 4)}")

    stab_thr = stability_threshold if stability_threshold is not None else STABILITY_THRESHOLD
    latest_stability = latest.get("stability", 0.0)
    if latest_stability < stab_thr:
        reasons.append("stability_below_threshold")

    trust_thr = trust_threshold if trust_threshold is not None else TRUST_THRESHOLD
    latest_trust = latest.get("trust", 0.0)
    if latest_trust < trust_thr:
        reasons.append("trust_below_threshold")

    latest_completions = latest.get("num_completions", 0)
    if latest_completions < 2:
        reasons.append("insufficient_completions")

    return len(reasons) > 0, reasons
