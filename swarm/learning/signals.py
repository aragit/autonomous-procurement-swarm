"""Learning signal computation from persisted trace data (v1.0 Step 19).

Derives interpretable signals that link past LLM decisions to outcomes,
enabling future feedback-driven improvement.

All functions are pure and deterministic: the same ``trace`` dict always
produces the same signals.
"""

from __future__ import annotations

from typing import Any

from swarm.config import ENABLE_LEARNING
from swarm.utils.llm_drift import detect_drift


def compute_learning_signals(trace: dict[str, Any]) -> dict[str, float]:
    """Compute learning signals from a complete trace.

    Analyzes the relationship between LLM consensus metrics and
    procurement outcomes to surface potential issues:

    - **confidence_vs_outcome_gap**: high confidence but low outcome
      indicates overconfidence.
    - **stability_vs_success**: low stability combined with failure
      indicates volatile LLM guidance.
    - **trust_vs_outcome**: low trust score with poor outcome suggests
      the system correctly distrusted the LLM.
    - **drift_impact**: drift detected with a bad outcome indicates
      that LLM drift hurt the result.

    Args:
        trace: A full trace dict from :func:`load_full_trace` with
            ``llm_history`` and ``feedback`` keys.

    Returns:
        A dict with the four signal values (all floats in [0, 1]).
        Returns all-zero signals when learning is disabled, history
        is empty, or feedback is missing.
    """
    if not ENABLE_LEARNING:
        return _zero_signals()

    history: list[dict[str, Any]] = trace.get("llm_history", [])
    feedback: dict[str, Any] | None = trace.get("feedback")

    if not history or feedback is None:
        return _zero_signals()

    success = bool(feedback.get("success", False))
    outcome = 1.0 if success else 0.0

    avg_confidence = _safe_avg(
        [h.get("confidence", 0.0) for h in history],
    )
    avg_stability = _safe_avg(
        [h.get("stability", 0.0) for h in history],
    )
    avg_trust = _safe_avg(
        [h.get("trust", 0.0) for h in history],
    )

    drift_detected, _ = detect_drift(history)

    # High confidence + low outcome = overconfidence gap
    confidence_vs_outcome_gap = max(0.0, avg_confidence - outcome)

    # Low stability + failure = instability issue
    stability_vs_success = 1.0 - avg_stability if not success and avg_stability < 0.5 else 0.0

    # Low trust + bad outcome = the system correctly didn't trust the LLM
    trust_vs_outcome = max(0.0, (1.0 - avg_trust) * (1.0 - outcome))

    # Drift detected + bad outcome = drift impacted the result
    drift_impact = max(0.0, 1.0 - outcome) if drift_detected and not success else 0.0

    return {
        "confidence_vs_outcome_gap": round(confidence_vs_outcome_gap, 4),
        "stability_vs_success": round(stability_vs_success, 4),
        "trust_vs_outcome": round(trust_vs_outcome, 4),
        "drift_impact": round(drift_impact, 4),
    }


def _safe_avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _zero_signals() -> dict[str, float]:
    return {
        "confidence_vs_outcome_gap": 0.0,
        "stability_vs_success": 0.0,
        "trust_vs_outcome": 0.0,
        "drift_impact": 0.0,
    }
