"""FastAPI application exposing v0.9 LLM observability endpoints (Step 13).

Can be used standalone (via ``set_state``) or integrated into the main API
(via ``set_state_provider``) where the host application controls state lookup.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, HTTPException

from swarm.core.state import SwarmState
from swarm.utils.llm_drift import detect_drift
from swarm.utils.llm_explain import aggregate_explanations
from swarm.utils.llm_memory import get_llm_consensus_history
from swarm.utils.llm_metrics import compute_llm_metrics

app = FastAPI(title="Autonomous Procurement Swarm — LLM Observability")

_state: SwarmState | None = None
_state_provider: Callable[[str], SwarmState | None] | None = None


def set_state(state: SwarmState) -> None:
    """Bind a :class:`SwarmState` instance to the FastAPI app."""
    global _state
    _state = state


def set_state_provider(provider: Callable[[str], SwarmState | None]) -> None:
    """Set a callable that resolves a :class:`SwarmState` by correlation ID.

    Used when integrating into a host application that manages multiple states.
    """
    global _state_provider
    _state_provider = provider


def _get_state(correlation_id: str) -> SwarmState:
    if _state_provider is not None:
        state = _state_provider(correlation_id)
        if state is not None:
            return state
    if _state is not None:
        return _state
    raise HTTPException(status_code=503, detail="Swarm state not initialized.")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/llm/metrics/{correlation_id}")
async def get_llm_metrics(correlation_id: str) -> dict[str, Any]:
    """Return aggregated LLM metrics for a given correlation ID."""
    state = _get_state(correlation_id)
    history = get_llm_consensus_history(state, correlation_id=correlation_id)
    return compute_llm_metrics(history)


@app.get("/llm/drift/{correlation_id}")
async def get_llm_drift(correlation_id: str) -> dict[str, Any]:
    """Return drift detection results for a given correlation ID."""
    state = _get_state(correlation_id)
    history = get_llm_consensus_history(state, correlation_id=correlation_id)
    drift_detected, reasons = detect_drift(history)
    return {"drift_detected": drift_detected, "reasons": reasons}


@app.get("/llm/explanation/{correlation_id}")
async def get_llm_explanation(correlation_id: str) -> dict[str, Any]:
    """Return aggregated explainability summary for a given correlation ID."""
    state = _get_state(correlation_id)
    history = get_llm_consensus_history(state, correlation_id=correlation_id)
    return aggregate_explanations(history)


@app.get("/llm/history/{correlation_id}")
async def get_llm_history(correlation_id: str) -> dict[str, Any]:
    """Return the raw consensus history for a given correlation ID."""
    state = _get_state(correlation_id)
    history = get_llm_consensus_history(state, correlation_id=correlation_id)
    return {"correlation_id": correlation_id, "records": history}
