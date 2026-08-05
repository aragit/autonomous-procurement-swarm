"""Temporal stability metric for LLM consensus history (v0.9 Step 6).

Measures how much the LLM's adjustment suggestions drift across time.
Low drift → high stability → more trust.
High drift → low stability → less trust.

The metric uses a range-based normalised drift per field, then averages across
all fields and all consecutive pairs of rounds.  A tolerance band of ±0.02 is
treated as "same suggestion" (zero drift) to avoid penalising insignificant
numerical noise.

Formula::

    For each field, for each consecutive pair (t_i, t_{i+1}):
        drift_i = max(0, |val_{i+1} - val_i| - TOLERANCE) / (0.2 - TOLERANCE)
    stability = 1 - mean(all drifts)

where 0.2 is the full ±0.1 acceptance range.
"""

from __future__ import annotations

from typing import Any

#: Adjustment fields recognised during stability computation.
_STABILITY_FIELDS = frozenset({"price_weight_delta", "delivery_weight_delta"})

#: Minimum history depth required for a meaningful stability score.
_MIN_HISTORY = 2

#: Smallest meaningful delta; anything within this band is "no drift".
TOLERANCE = 0.02

#: Full span of the accepted delta range (±0.1 → 0.2).
_DELTA_RANGE = 0.2

#: Minimum stability needed for the temporal dimension to pass.
STABILITY_THRESHOLD = 0.5

#: Minimum trust score for adjustments to be applied.
#: trust = consensus_confidence * temporal_stability
TRUST_THRESHOLD = 0.7

#: Minimum stability needed for the temporal dimension to pass.
STABILITY_THRESHOLD = 0.5


def compute_temporal_stability(history: list[dict[str, Any]]) -> float:
    """Compute temporal stability from a list of consensus history records.

    Args:
        history: List of dicts, each with ``aggregated_adjustments`` (a
            dict of ``field → float``) and ``round`` (int).  Ordered
            oldest-first.

    Returns:
        A float in [0, 1].  ``1.0`` means perfectly stable (no drift),
        ``0.0`` means maximum drift.

    Rules:
        - Empty history or single record → ``0.0`` (can't measure stability).
        - Records without ``aggregated_adjustments`` are skipped for that
          field in the pair but do not cause errors.
        - Drift per field is normalised by ``_DELTA_RANGE`` minus tolerance.
    """
    if len(history) < _MIN_HISTORY:
        return 0.0

    sorted_history = sorted(history, key=lambda h: h.get("round", 0))

    drifts: list[float] = []

    for i in range(len(sorted_history) - 1):
        curr = sorted_history[i].get("aggregated_adjustments", {})
        next_ = sorted_history[i + 1].get("aggregated_adjustments", {})

        for field in _STABILITY_FIELDS:
            if field in curr and field in next_:
                try:
                    curr_val = float(curr[field])
                    next_val = float(next_[field])
                except (TypeError, ValueError):
                    continue

                abs_diff = abs(next_val - curr_val)
                drift = max(0.0, abs_diff - TOLERANCE) / (_DELTA_RANGE - TOLERANCE)
                drifts.append(drift)

    if not drifts:
        return 0.0

    mean_drift = sum(drifts) / len(drifts)
    stability = max(0.0, 1.0 - mean_drift)
    return round(stability, 4)
