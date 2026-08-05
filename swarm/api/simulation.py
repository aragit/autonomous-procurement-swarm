"""Simulation and replay API (v1.0 Step 21).

Exposes deterministic, read-only replay endpoints so operators can validate
whether the adaptive policy would improve past decisions:

- ``POST /simulation/replay/{trace_id}`` — re-run one trace, return the
  original decision, the replayed decision and a comparison.
- ``GET /simulation/run`` — batch-replay recent traces and aggregate outcomes.

These endpoints never mutate the production database: the replay engine
isolates all writes to a throwaway store.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from swarm.config import SIMULATION_LIMIT
from swarm.simulation.replay_engine import (
    TraceNotFoundError,
    replay_trace,
    simulate_all_traces,
)

router = APIRouter()


@router.post("/simulation/replay/{trace_id}")
def replay_endpoint(trace_id: str) -> dict[str, Any]:
    """Replay a single persisted trace and compare the outcome.

    Response shape::

        {
            "original": {...},       # persisted artifacts (decision)
            "replayed": {...},       # re-run result
            "comparison": {          # same_supplier, score_delta, decision_changed
                "same_supplier": bool,
                "score_delta": float,
                "decision_changed": bool,
            },
            "trace_id": str,
        }
    """
    try:
        result = replay_trace(trace_id, use_adaptive=True)
    except TraceNotFoundError:
        return {"error": "trace_not_found", "trace_id": trace_id}
    return result


@router.get("/simulation/run")
def run_simulation(limit: int = Query(default=SIMULATION_LIMIT, ge=1)) -> dict[str, Any]:
    """Batch-replay recent traces and return an aggregated summary.

    Query params:
        limit: maximum number of recent traces to replay (default 100).

    Response shape::

        {
            "summary": {
                "total": int,
                "changed_decisions": int,
                "avg_score_delta": float,
                "improvement_rate": float,
            }
        }
    """
    summary = simulate_all_traces(limit=limit)
    return {"summary": summary}
