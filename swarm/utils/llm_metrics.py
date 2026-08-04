"""Aggregated LLM metrics from consensus history (v0.9 Step 10).

Reads consensus history records (produced by ``llm_memory.record_llm_consensus``)
and computes aggregate statistics for observability and drift monitoring.
"""

from __future__ import annotations

from typing import Any

MAX_HISTORY = 5


def compute_llm_metrics(
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute aggregate metrics across consensus history records.

    Returns a dict with:
    - ``acceptance_rate``: fraction of rounds where LLM was accepted (0.0–1.0).
    - ``avg_confidence``: mean confidence across rounds.
    - ``avg_stability``: mean temporal stability across rounds.
    - ``avg_trust``: mean trust score across rounds.
    - ``history_depth``: number of records in the history.
    - ``max_history_depth``: maximum expected history depth.
    """
    if not history:
        return {
            "acceptance_rate": 0.0,
            "avg_confidence": 0.0,
            "avg_stability": 0.0,
            "avg_trust": 0.0,
            "history_depth": 0,
            "max_history_depth": MAX_HISTORY,
        }

    total = len(history)
    accepted = sum(
        1
        for r in history
        if r.get("decision_reason", "accepted") == "accepted"
    )
    confidences = [r.get("confidence", 0.0) for r in history]
    stabilities = [r.get("stability", 0.0) for r in history]
    trusts = [r.get("trust", 0.0) for r in history]

    return {
        "acceptance_rate": round(accepted / total, 4),
        "avg_confidence": round(sum(confidences) / total, 4),
        "avg_stability": round(sum(stabilities) / total, 4),
        "avg_trust": round(sum(trusts) / total, 4),
        "history_depth": total,
        "max_history_depth": MAX_HISTORY,
    }
