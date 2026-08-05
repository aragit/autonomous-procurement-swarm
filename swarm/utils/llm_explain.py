"""Aggregated explainability across LLM consensus history (v0.9 Step 12).

Produces a consolidated explanation that summarizes the full trajectory of
LLM influence decisions across rounds — including drift analysis — with a
deterministic summary template.
"""

from __future__ import annotations

from typing import Any

from swarm.utils.llm_drift import detect_drift


def aggregate_explanations(
    history: list[dict[str, Any]],
    current_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an aggregated explanation across all consensus rounds.

    Args:
        history: Full consensus history (from ``get_llm_consensus_history``).
        current_decision: The explanation dict from ``build_llm_explanation``
            for the current round, or ``None`` if no LLM context was used.

    Returns:
        A dict with:
        - ``rounds``: per-round summary of confidence, stability, trust, reason.
        - ``aggregate``: overall metrics (average confidence, stability, trust).
        - ``drift``: drift analysis result (detected flag + reasons).
        - ``current``: the current round explanation dict (if provided).
        - ``summary``: a deterministic one-line summary of the overall state.
    """
    if not history:
        return {
            "rounds": [],
            "aggregate": {
                "avg_confidence": 0.0,
                "avg_stability": 0.0,
                "avg_trust": 0.0,
                "total_rounds": 0,
            },
            "drift": {"detected": False, "reasons": []},
            "current": current_decision or {},
            "summary": "No LLM history available.",
        }

    rounds: list[dict[str, Any]] = []
    confidences: list[float] = []
    stabilities: list[float] = []
    trusts: list[float] = []

    for record in history:
        conf = record.get("confidence", 0.0)
        stab = record.get("stability", 0.0)
        trust = record.get("trust", 0.0)
        reason = record.get("decision_reason", "unknown")
        rounds.append(
            {
                "round": record.get("round", 0),
                "confidence": round(conf, 4),
                "stability": round(stab, 4),
                "trust": round(trust, 4),
                "reason": reason,
            }
        )
        confidences.append(conf)
        stabilities.append(stab)
        trusts.append(trust)

    total = len(history)
    drift_detected, drift_reasons = detect_drift(history)

    summary_parts = [
        f"{total} rounds",
        f"avg_conf={round(sum(confidences) / total, 2)}",
        f"drift={'yes' if drift_detected else 'no'}",
    ]
    summary = " | ".join(summary_parts)

    return {
        "rounds": rounds,
        "aggregate": {
            "avg_confidence": round(sum(confidences) / total, 4),
            "avg_stability": round(sum(stabilities) / total, 4),
            "avg_trust": round(sum(trusts) / total, 4),
            "total_rounds": total,
        },
        "drift": {"detected": drift_detected, "reasons": drift_reasons},
        "current": current_decision or {},
        "summary": summary,
    }
