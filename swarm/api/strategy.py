"""FastAPI application exposing v0.9 LLM observability endpoints (Step 13).

Can be used standalone (via ``set_state``) or integrated into the main API
(via ``set_state_provider``) where the host application controls state lookup.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, HTTPException

from swarm.config import ENABLE_LEARNING
from swarm.core.state import SwarmState
from swarm.learning.adaptive_policy import get_adaptive_thresholds
from swarm.learning.signals import compute_learning_signals
from swarm.storage.event_store import load_full_trace
from swarm.utils.llm_drift import detect_drift
from swarm.utils.llm_explain import aggregate_explanations
from swarm.utils.llm_memory import get_llm_consensus_history
from swarm.utils.llm_metrics import compute_llm_metrics

app = FastAPI(title="Autonomous Procurement Swarm — LLM Observability")

_state: SwarmState | None = None
_state_provider: Callable[[str], SwarmState | None] | None = None


def _zero_learning_signals() -> dict[str, float]:
    return {
        "confidence_vs_outcome_gap": 0.0,
        "stability_vs_success": 0.0,
        "trust_vs_outcome": 0.0,
        "drift_impact": 0.0,
    }


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


def get_state(trace_id: str) -> SwarmState | None:
    """Resolve a :class:`SwarmState` by trace ID.

    Returns ``None`` if no state is available for the given trace ID,
    rather than raising. Used by the dashboard endpoint to allow
    a graceful ``trace_not_found`` response.
    """
    if _state_provider is not None:
        return _state_provider(trace_id)
    return _state


def build_dashboard(trace_id: str) -> dict[str, Any]:
    """Aggregate the complete LLM decision context for a given trace ID.

    Consolidates metrics, drift detection, explanation, history, feedback,
    and learning signals into a single deterministic response by reusing
    existing utilities.

    Tries in-memory state first (via ``get_state``), then falls back to
    the persistent event store (``load_full_trace``) for durable,
    replayable data.

    Args:
        trace_id: The trace/correlation ID to look up.

    Returns:
        A dict with ``trace_id``, ``summary``, ``decision``, ``metrics``,
        ``drift``, ``history``, ``feedback``, and ``learning_signals``
        fields. If no state is found, returns ``{"error": "trace_not_found"}``.
    """
    # Try in-memory state first, then fall back to DB-backed state
    state = get_state(trace_id)
    db_state: dict[str, Any] | None = None
    full_trace: dict[str, Any] | None = None
    if state is not None:
        history = get_llm_consensus_history(state, correlation_id=trace_id)
    else:
        full_trace = load_full_trace(trace_id)
        db_state = {
            "events": full_trace["events"],
            "artifacts": full_trace["artifacts"],
            "llm_history": full_trace["llm_history"],
        }
        history = full_trace["llm_history"]
        # If DB has no data either, the trace doesn't exist
        if not db_state.get("events") and not db_state.get("artifacts") and not history:
            return {"error": "trace_not_found"}

    metrics = compute_llm_metrics(history)
    thresholds = get_adaptive_thresholds()
    drift_detected, drift_reasons = detect_drift(
        history,
        stability_threshold=thresholds["stability_threshold"],
        trust_threshold=thresholds["trust_threshold"],
    )

    last = history[-1] if history else {}
    decision: dict[str, Any] = {
        "used_llm": last.get("decision_reason") == "accepted",
        "reason": last.get("decision_reason", "no_data"),
    }

    explain = aggregate_explanations(
        history,
        current_decision=decision,
        stability_threshold=thresholds["stability_threshold"],
        trust_threshold=thresholds["trust_threshold"],
    )

    formatted_history: list[dict[str, Any]] = [
        {
            "round": i + 1,
            "confidence": h.get("confidence", 0.0),
            "stability": h.get("stability", 0.0),
            "trust": h.get("trust", 0.0),
            "decision_reason": h.get("decision_reason", "no_data"),
        }
        for i, h in enumerate(history)
    ]

    return {
        "trace_id": trace_id,
        "summary": explain.get("summary", ""),
        "decision": decision,
        "metrics": metrics,
        "drift": {
            "drifting": drift_detected,
            "reasons": drift_reasons,
        },
        "history": formatted_history,
        "feedback": full_trace.get("feedback") if full_trace else None,
        "learning_signals": compute_learning_signals(full_trace)
        if full_trace
        else _zero_learning_signals(),
        "adaptive": {
            "enabled": ENABLE_LEARNING,
            "thresholds": {
                "confidence": thresholds["confidence_threshold"],
                "stability": thresholds["stability_threshold"],
                "trust": thresholds["trust_threshold"],
            },
        },
        "replay": {
            "available": full_trace is not None and bool(full_trace.get("events")),
        },
    }


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


@app.get("/llm/dashboard/{trace_id}")
async def llm_dashboard(trace_id: str) -> dict[str, Any]:
    """Return the complete LLM decision context for a given trace ID."""
    return build_dashboard(trace_id)
