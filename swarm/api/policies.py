"""Closed-loop policy management API (v1.1 Step 22).

Exposes three endpoints that close the learning loop without ever running the
live (stateful) swarm:

* ``POST /policies/learn``  — run the deterministic grid search over the trace
  corpus and return the best candidate. Idempotent: the same traces always
  yield the same candidate ``version``.
* ``POST /policies/promote`` — atomically promote a learned candidate to the
  active policy, applying the promotion + safety rules. Rejects if the
  candidate is already promoted.
* ``GET /policies/active``  — the currently-active policy (or the config
  baseline when nothing has been promoted).

These endpoints never mutate production traces: candidate evaluation runs
through the isolated replay engine with pinned thresholds/weights.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from swarm.learning.learner import get_active_policy, learn_candidates, promote_policy
from swarm.storage.event_store import load_policy

router = APIRouter()


class PromoteRequest(BaseModel):
    """Body for ``POST /policies/promote``."""

    version: str
    force: bool = False


@router.get("/policies/active")
def get_active_policy_endpoint() -> dict[str, Any]:
    """Return the currently-active policy, or the config baseline.

    When a candidate has been promoted, that policy's thresholds/weights are
    returned. Before any promotion the static config baseline is returned so
    callers always see a valid policy shape.
    """
    policy = get_active_policy()
    record = policy.to_record()
    return {
        "version": record["version"],
        "thresholds": record["thresholds"],
        "weights": record["weights"],
        "strategy": record["strategy"],
        "metric": record["metric"],
        "active": True,
    }


@router.post("/policies/learn")
def learn_policy_endpoint() -> dict[str, Any]:
    """Run the closed-loop policy grid search and return the best candidate.

    Idempotent: identical persisted traces (and DB contents) produce the same
    best candidate ``version`` on every call. Candidate evaluation never
    touches production traces — replays run in an isolated, throwaway store.

    Response ``status``:

    * ``"ok"``              — a best candidate was produced (see ``best``).
    * ``"insufficient_data"`` — fewer than ``MIN_TRACES_FOR_LEARNING`` traces
      carry feedback, so no candidate was learned.
    * ``"no_candidates"``   — corpus large enough but every candidate was
      rejected by the safety guardrail.
    """
    result = learn_candidates()
    if result["status"] == "insufficient_data":
        return result
    return result


@router.post("/policies/promote")
def promote_policy_endpoint(request: PromoteRequest) -> dict[str, Any]:
    """Atomically promote a learned candidate to the active policy.

    Applies the Step 22 promotion rule
    (``metric > active.metric + IMPROVEMENT_MARGIN`` AND
    ``decision_stability >= active.decision_stability``) and the safety guardrail.
    Rejects if the candidate is already promoted — the operation is atomic and
    never silently no-ops.

    Set ``force=true`` to bypass the improvement rule for rollback to a
    previously promoted or baseline policy. Safety and already-promoted checks
    still apply during forced promotion.
    """
    result = promote_policy(request.version, force=request.force)
    if result["status"] == "rejected":
        raise HTTPException(
            status_code=409,
            detail={
                "version": result["version"],
                "reason": result["reason"],
            },
        )
    return result


@router.get("/policies/{version}")
def get_policy_endpoint(version: str) -> dict[str, Any]:
    """Look up a stored policy by version."""
    try:
        record = load_policy(version)
    except Exception:
        record = None
    if record is None:
        raise HTTPException(status_code=404, detail=f"No policy with version {version}")
    return record
