"""Single-entry procurement execution endpoint with full observability (v0.9 Step 15).

Exposes ``POST /procurement/run`` — a deterministic end-to-end endpoint that
runs the full procurement swarm and returns the result plus the complete LLM
observability context (metrics, drift, explanation).
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
from swarm.utils.llm_drift import detect_drift
from swarm.utils.llm_explain import aggregate_explanations
from swarm.utils.llm_memory import get_llm_consensus_history
from swarm.utils.llm_metrics import compute_llm_metrics

router = APIRouter()
_swarm_states: dict[str, SwarmState] = {}


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


async def run_swarm(trace_id: str, requirement: RequirementPayload) -> Swarm:
    """Build and run the procurement swarm for a requirement."""
    swarm = build_procurement_swarm(
        request_id=trace_id,
        goal=requirement.goal or (
            f"Source {requirement.quantity} units of {requirement.material}"
        ),
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


@router.post("/procurement/run")
async def run_procurement(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Run the full procurement swarm and return result + LLM observability."""
    requirement_dict = payload.get("requirement") if isinstance(payload, dict) else None
    if requirement_dict is None:
        return {"error": "invalid_requirement", "detail": "Missing 'requirement' field in payload."}

    try:
        requirement = RequirementPayload(**requirement_dict)
    except Exception as exc:
        return {"error": "invalid_requirement", "detail": str(exc)}

    trace_id = generate_trace_id(requirement)

    try:
        swarm = await run_swarm(trace_id, requirement)
    except Exception as exc:
        return {"error": "execution_failure", "detail": str(exc)}

    state = swarm.state
    store_state(trace_id, state)

    strategy = get_strategy(state)
    result = get_selected_supplier(state)

    # Extract LLM observability
    correlation_id = f"{trace_id}-CONV"
    history = get_llm_consensus_history(state, correlation_id=correlation_id)
    metrics = compute_llm_metrics(history)

    drift_detected, drift_reasons = detect_drift(history)

    llm_fallback = strategy.get("llm_fallback", {}) if strategy else {}
    explanation = aggregate_explanations(history, current_decision=llm_fallback)

    response = {
        "result": result,
        "strategy": strategy,
        "llm": {
            "used": llm_fallback.get("used", False),
            "reason": llm_fallback.get("reason", "no_llm_data"),
            "metrics": metrics,
            "drift": {"detected": drift_detected, "reasons": drift_reasons},
            "explain": explanation,
        },
        "trace_id": trace_id,
    }
    return response
