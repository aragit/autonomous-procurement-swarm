"""LLM drift detection from consensus history (v0.9 Step 11).

Detects when the LLM consensus signal is drifting — e.g., confidence declining
rapidly, stability dropping below threshold, or sudden changes in agreement.
"""

from __future__ import annotations

from typing import Any

from swarm.utils.llm_stability import STABILITY_THRESHOLD, TRUST_THRESHOLD


def detect_drift(
    history: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    """Detect drift in LLM consensus signals across history records.

    Returns ``(drift_detected, reasons)`` where ``reasons`` is a list of
    human-readable strings describing each detected drift condition.

    Drift conditions checked:
    - Confidence dropped by more than 0.15 between consecutive rounds.
    - Stability below ``STABILITY_THRESHOLD`` in the latest round.
    - Trust score below ``TRUST_THRESHOLD`` in the latest round.
    - Fewer than 2 completions available (low signal-to-noise).
    """
    if not history:
        return False, []

    reasons: list[str] = []

    latest = history[-1]

    if len(history) >= 2:
        prev = history[-2]
        conf_drop = prev.get("confidence", 0.0) - latest.get("confidence", 0.0)
        if conf_drop > 0.15:
            reasons.append(
                f"confidence_drop:{round(conf_drop, 4)}"
            )

    latest_stability = latest.get("stability", 0.0)
    if latest_stability < STABILITY_THRESHOLD:
        reasons.append("stability_below_threshold")

    latest_trust = latest.get("trust", 0.0)
    if latest_trust < TRUST_THRESHOLD:
        reasons.append("trust_below_threshold")

    latest_completions = latest.get("num_completions", 0)
    if latest_completions < 2:
        reasons.append("insufficient_completions")

    return len(reasons) > 0, reasons
