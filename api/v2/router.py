"""FastAPI router for the v2 (Mesh Runtime) procurement API.

Routes:
    POST /v2/procurement/run            — run a procurement through the mesh
    GET  /v2/procurement/{trace_id}/status — blackboard snapshot / stats

The router depends only on the :class:`~api.v2.runtime.MeshRuntime` protocol,
so it is fully testable with an injected fake runtime and does not require Ray.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from api.v2.models import (
    ProcurementErrorResponse,
    ProcurementRunRequest,
    ProcurementRunResponse,
    ProcurementStatusResponse,
)
from api.v2.runtime import get_runtime, make_trace_id

router = APIRouter(
    prefix="/v2",
    tags=["v2-mesh"],
)


def _requirement_dict(request: ProcurementRunRequest, correlation_id: str) -> dict[str, Any]:
    """Translate the v2 request into the requirement dict the mesh expects."""
    return {
        "correlation_id": correlation_id,
        "constraints": {
            "material": request.material,
            "quantity": request.quantity,
            "budget": request.budget,
            "target_lead_time_days": request.target_lead_time_days,
            "max_carbon_per_unit": request.max_carbon_per_unit,
        },
        "goal": request.goal or f"Source {request.quantity} units of {request.material}",
        "governance_policy": request.governance_policy,
        "enable_neuro": request.enable_neuro,
        "neuro_llm_base_url": request.neuro_llm_base_url,
        "neuro_llm_model": request.neuro_llm_model,
    }


@router.post("/procurement/run", response_model=ProcurementRunResponse)
async def run_procurement(
    request: ProcurementRunRequest,
    background_tasks: BackgroundTasks,
) -> ProcurementRunResponse:
    """Initialise the ProcurementCluster, submit a requirement, await the DECISION.

    The full perceive → reason → validate → act pipeline runs to completion
    (deterministic phases are synchronous; the optional neuro path honours its
    retry budget) and the final buyer DECISION is returned inline.  The mesh
    cluster is kept alive in the background so the status endpoint can serve a
    live blackboard snapshot; it is torn down by the runtime's lifecycle hook.
    """
    trace_id = make_trace_id()
    correlation_id = f"{trace_id}-CORR"
    requirement = _requirement_dict(request, correlation_id)

    try:
        runtime = get_runtime()
        result = await runtime.run_procurement(trace_id, requirement)
    except Exception as exc:
        logger = __import__("structlog").get_logger(__name__)
        logger.error("v2_procurement_failed", trace_id=trace_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Procurement run failed: {exc}",
        ) from exc

    return result


@router.get(
    "/procurement/{trace_id}/status",
    response_model=ProcurementStatusResponse,
    responses={404: {"model": ProcurementErrorResponse}},
)
async def get_procurement_status(trace_id: str) -> ProcurementStatusResponse:
    """Pull the current state for ``trace_id`` from the blackboard snapshot/stats."""
    runtime = get_runtime()
    status_response = await runtime.get_status(trace_id)
    if status_response is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No procurement run found for trace_id '{trace_id}'",
        )
    return status_response


@router.get(
    "/procurement/health",
    response_model=dict[str, str],
)
async def mesh_health() -> dict[str, str]:
    """Lightweight liveness probe that confirms a runtime is wired."""
    runtime = get_runtime()
    try:
        # The protocol is always satisfiable by an injected runtime; the call is
        # only to confirm the runtime object is callable/healthy.
        _ = runtime
        return {"status": "ready"}
    except Exception as exc:  # pragma: no cover
        return {"status": "unavailable", "detail": str(exc)}
