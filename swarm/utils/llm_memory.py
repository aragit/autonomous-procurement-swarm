"""Persistent consensus history for temporal stability (v0.9 Step 6).

Each time the :class:`StrategyAgent` computes an LLM consensus, the result is
recorded as an artifact on :class:`SwarmState`.  This module provides the
record/retrieve API so that the temporal evaluator can measure drift across
multiple consensus rounds for the same ``correlation_id``.

Replay safety: history is reconstructed purely from artifacts in state — no
hidden state, no side-effecting stores.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from swarm.core.artifact import Artifact
from swarm.core.state import SwarmState

MAX_HISTORY = 5


def record_llm_consensus(
    state: SwarmState,
    *,
    correlation_id: str,
    consensus: dict[str, Any],
    round_number: int,
    stability: float | None = None,
    trust: float | None = None,
    decision_reason: str | None = None,
    parent_ids: list[str] | None = None,
    by: str = "strategy_agent",
) -> Artifact:
    """Persist a consensus result as a replay-safe artifact in ``state``.

    The artifact is tagged ``{"llm_consensus": True}`` and stamped with
    ``round_number`` so the temporal evaluator can reconstruct the chronological
    order of consensus rounds. Optional ``stability``, ``trust`` and
    ``decision_reason`` fields enrich the record for metrics and drift
    analysis (Steps 10-11).
    """
    data: dict[str, Any] = {
        "correlation_id": correlation_id,
        "round": round_number,
        "timestamp": datetime.now(UTC).isoformat(),
        "confidence": consensus.get("confidence", 0.0),
        "agreement_score": consensus.get("agreement_score", 0.0),
        "completeness": consensus.get("completeness", 0.0),
        "num_completions": consensus.get("num_completions", 0),
        "aggregated_adjustments": consensus.get("aggregated_adjustments", {}),
    }
    if stability is not None:
        data["stability"] = stability
    if trust is not None:
        data["trust"] = trust
    if decision_reason is not None:
        data["decision_reason"] = decision_reason
    artifact = Artifact(
        kind="llm_consensus",
        name=f"llm_consensus_{correlation_id}_{round_number}",
        data=data,
        tags={"llm_consensus": "true"},
        parent_ids=parent_ids or [],
        created_by=by,
        correlation_id=correlation_id,
    )
    state.put_artifact(artifact)
    return artifact


def get_llm_consensus_history(
    state: SwarmState,
    *,
    correlation_id: str,
    limit: int = MAX_HISTORY,
) -> list[dict[str, Any]]:
    """Return the most recent ``limit`` consensus records for ``correlation_id``.

    Results are ordered oldest-first (round ascending).  If no history exists,
    an empty list is returned — the caller handles the single-data-point case.
    """
    matches = state.find_artifacts(
        kind="llm_consensus",
        correlation_id=correlation_id,
    )

    records: list[dict[str, Any]] = []
    for artifact in matches:
        if not isinstance(artifact.data, dict):
            continue
        records.append(dict(artifact.data))

    records.sort(key=lambda r: r.get("round", 0))
    return records[-limit:] if limit else records
