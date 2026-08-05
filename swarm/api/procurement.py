"""Single-entry procurement execution endpoint with full observability (v0.9 Step 15).

Exposes ``POST /procurement/run`` — a deterministic end-to-end endpoint that
runs the full procurement swarm and returns the result plus the complete LLM
observability context (metrics, drift, explanation).

After execution, all events, artifacts, and LLM history are persisted to
the event store (``swarm.storage.event_store``) so the dashboard can be
served from the database rather than in-memory state.
"""

from __future__ import annotations

import hashlib
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from swarm import Swarm
from swarm.core.state import SwarmState
from swarm.domain.events import CREATE_REQUIREMENT_INTENT
from swarm.domain.wiring import build_procurement_swarm
from swarm.learning.adaptive_policy import (
    get_adaptive_thresholds,
    get_config_thresholds,
    override_thresholds,
)
from swarm.storage.event_store import store_artifact, store_event, store_feedback
from swarm.utils.llm_drift import detect_drift
from swarm.utils.llm_explain import aggregate_explanations
from swarm.utils.llm_memory import get_llm_consensus_history
from swarm.utils.llm_metrics import compute_llm_metrics

router = APIRouter()
_swarm_states: dict[str, SwarmState] = {}


class ExecutionError(Exception):
    """Raised when the procurement swarm cannot complete a run."""


class RequirementPayload(BaseModel):
    material: str = Field(..., min_length=1)
    quantity: int = Field(default=1000, gt=0)
    budget: float = Field(default=2_000_000.0, gt=0)
    target_lead_time_days: int = Field(default=30, gt=0)
    max_carbon_per_unit: float | None = Field(default=None, gt=0)
    goal: str | None = Field(default=None)
    supplier_count: int = Field(default=5, gt=0)


def generate_trace_id(payload: RequirementPayload) -> str:
    """Deterministic trace ID derived from the requirement payload."""
    raw = (
        f"{payload.material}:{payload.quantity}:{payload.budget}:"
        f"{payload.target_lead_time_days}:{payload.max_carbon_per_unit}"
    )
    hash_hex = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"TRACE-{hash_hex}"


def create_initial_state(trace_id: str, goal: str) -> SwarmState:
    return SwarmState(request_id=trace_id, goal=goal)


class FeedbackPayload(BaseModel):
    trace_id: str = Field(..., min_length=1)
    outcome_score: float = Field(..., ge=0.0, le=1.0)
    success: bool
    latency_ms: float = Field(..., ge=0.0)
    user_feedback: str | None = Field(default=None)


async def run_swarm(trace_id: str, requirement: RequirementPayload) -> Swarm:
    """Build and run the procurement swarm for a requirement."""
    swarm = build_procurement_swarm(
        request_id=trace_id,
        goal=requirement.goal or (f"Source {requirement.quantity} units of {requirement.material}"),
    )
    await swarm.start()
    correlation_id = f"{trace_id}-CONV"
    await swarm.send_message(
        CREATE_REQUIREMENT_INTENT,
        {
            "text": f"Source {requirement.quantity} units of {requirement.material}",
            "material": requirement.material,
            "quantity": requirement.quantity,
            "budget": requirement.budget,
            "target_lead_time_days": requirement.target_lead_time_days,
            "max_carbon_per_unit": requirement.max_carbon_per_unit,
        },
        sender="procurement_api",
        correlation_id=correlation_id,
    )
    await swarm.shutdown()
    return swarm


def get_selected_supplier(state: SwarmState) -> dict[str, Any]:
    decision = state.get_artifact("decision")
    if decision is None:
        return {"selected_supplier": None, "score": 0.0}
    data = decision.data
    ranked = data.get("reasoning", {}).get("ranked", [])
    score = ranked[0].get("score", 0.0) if ranked else 0.0
    return {
        "selected_supplier": data.get("selected_supplier"),
        "score": round(score, 4),
    }


def get_strategy(state: SwarmState) -> dict[str, Any]:
    strategy = state.get_artifact("strategy")
    if strategy is None:
        return {}
    return strategy.data


def store_state(trace_id: str, state: SwarmState) -> None:
    _swarm_states[trace_id] = state


def get_stored_state(trace_id: str) -> tuple[SwarmState, str]:
    if trace_id not in _swarm_states:
        raise HTTPException(
            status_code=404,
            detail=f"No state stored for trace_id {trace_id}",
        )
    return _swarm_states[trace_id], f"{trace_id}-CONV"


async def execute_procurement(
    requirement: RequirementPayload,
    *,
    trace_id: str | None = None,
    adaptive: bool = True,
    thresholds: dict[str, float] | None = None,
) -> tuple[dict[str, Any], SwarmState]:
    """Run the procurement swarm and assemble the observability response.

    This is the shared execution primitive: the ``/procurement/run`` endpoint uses
    it (and persists the result afterward) while the replay/simulation engine uses
    it with ``thresholds`` pre-supplied and an isolated database so that replaying
    never mutates production state.

    Args:
        requirement: The validated requirement payload.
        trace_id: Optional pre-computed trace id; generated from the requirement
            when omitted.
        adaptive: When False, config-only thresholds are used (no database read).
        thresholds: Optional explicit thresholds to pin during the run. When
            supplied, the ``adaptive`` flag is ignored for threshold selection.

    Returns:
        A ``(response, state)`` tuple where ``response`` mirrors the endpoint
        payload and ``state`` is the final :class:`SwarmState` (for persistence).
    """
    if trace_id is None:
        trace_id = generate_trace_id(requirement)

    if thresholds is None:
        thresholds = get_adaptive_thresholds() if adaptive else get_config_thresholds()

    with override_thresholds(thresholds):
        try:
            swarm = await run_swarm(trace_id, requirement)
        except Exception as exc:
            raise ExecutionError(str(exc)) from exc

    state = swarm.state
    strategy = get_strategy(state)
    result = get_selected_supplier(state)

    correlation_id = f"{trace_id}-CONV"
    history = get_llm_consensus_history(state, correlation_id=correlation_id)
    metrics = compute_llm_metrics(history)
    drift_detected, drift_reasons = detect_drift(
        history,
        stability_threshold=thresholds["stability_threshold"],
        trust_threshold=thresholds["trust_threshold"],
    )

    llm_fallback = strategy.get("llm_fallback", {}) if strategy else {}
    explanation = aggregate_explanations(
        history,
        current_decision=llm_fallback,
        stability_threshold=thresholds["stability_threshold"],
        trust_threshold=thresholds["trust_threshold"],
    )

    response: dict[str, Any] = {
        "result": result,
        "strategy": strategy,
        "llm": {
            "used": llm_fallback.get("used", False),
            "reason": llm_fallback.get("reason", "no_llm_data"),
            "metrics": metrics,
            "drift": {"detected": drift_detected, "reasons": drift_reasons},
            "explain": explanation,
            "thresholds_used": thresholds,
            "thresholds_source": "adaptive" if adaptive else "static",
        },
        "trace_id": trace_id,
    }
    return response, state


@router.post("/procurement/run")
async def run_procurement(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Run the full procurement swarm and return result + LLM observability.

    After execution, all events, artifacts, and LLM history are persisted to
    the event store (``swarm.storage.event_store``) so the dashboard can be
    served from the database rather than in-memory state.
    """
    requirement_dict = payload.get("requirement") if isinstance(payload, dict) else None
    if requirement_dict is None:
        return {"error": "invalid_requirement", "detail": "Missing 'requirement' field in payload."}

    try:
        requirement = RequirementPayload(**requirement_dict)
    except Exception as exc:
        return {"error": "invalid_requirement", "detail": str(exc)}

    trace_id = generate_trace_id(requirement)

    try:
        response, state = await execute_procurement(requirement, trace_id=trace_id)
    except ExecutionError as exc:
        return {"error": "execution_failure", "detail": str(exc)}

    store_state(trace_id, state)

    # Persist to event store for durable, replayable state

    # Record the original procurement request so replays can reconstruct inputs
    store_event(trace_id, "procurement_request", requirement.model_dump())

    for event in state.events:
        store_event(trace_id, event.type, event.payload or {})

    strategy_artifact = state.get_artifact("strategy")
    if strategy_artifact is not None:
        store_artifact(trace_id, "strategy", strategy_artifact.data)

    decision_artifact = state.get_artifact("decision")
    if decision_artifact is not None:
        store_artifact(trace_id, "result", decision_artifact.data)

    return response


@router.post("/procurement/feedback")
async def submit_feedback(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Submit outcome feedback for a trace.

    Stores feedback (outcome score, success flag, latency, and optional
    user feedback) linked to a trace ID.  This data is used by the
    learning signal computation.
    """
    if not isinstance(payload, dict):
        return {"error": "invalid_feedback", "detail": "payload must be a JSON object"}

    data = payload.get("feedback", payload)
    if not isinstance(data, dict):
        return {"error": "invalid_feedback", "detail": "feedback must be an object"}

    try:
        fb = FeedbackPayload(**data)
    except Exception as exc:
        return {"error": "invalid_feedback", "detail": str(exc)}

    store_feedback(
        trace_id=fb.trace_id,
        outcome_score=fb.outcome_score,
        success=fb.success,
        latency_ms=fb.latency_ms,
        user_feedback=fb.user_feedback,
    )

    return {"status": "stored", "trace_id": fb.trace_id}
