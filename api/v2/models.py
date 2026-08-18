"""Pydantic models for the v2 (Mesh Runtime) API endpoints.

These mirror the legacy :class:`api.SwarmRequirementRequest` but are kept
self-contained so the v2 package has no dependency on the legacy asyncio API
module.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

_VALID_MATERIALS = {"steel", "aluminum", "copper", "plastic", "lumber", "rubber"}


class ProcurementRunRequest(BaseModel):
    """Payload for ``POST /v2/procurement/run``.

    Submits a requirement into the distributed Ray mesh and awaits the final
    DECISION written to the blackboard by the BuyerActor.
    """

    material: str = Field(default="steel", min_length=1)
    quantity: int = Field(default=1000, gt=0)
    budget: float = Field(default=2_000_000.0, gt=0)
    target_lead_time_days: int = Field(default=30, gt=0)
    max_carbon_per_unit: float | None = Field(default=None, gt=0)
    goal: str | None = Field(default=None)
    governance_policy: str = Field(default="standard")
    enable_neuro: bool = Field(
        default=False,
        description=(
            "When true, ScoutActor and NegotiatorActor use schema-constrained "
            "LLM generation (OpenAI-compatible backend) validated by the "
            "SafetyKernelActor with an auto-correction retry loop."
        ),
    )
    neuro_llm_base_url: str | None = Field(
        default=None,
        description="OpenAI-compatible endpoint for the neuro backend (e.g. vLLM/Ollama).",
    )
    neuro_llm_model: str | None = Field(
        default=None,
        description="Model name served by the neuro backend.",
    )

    @field_validator("material")
    @classmethod
    def validate_material(cls, v: str) -> str:
        if v.lower() not in _VALID_MATERIALS:
            raise ValueError(f"material must be one of {sorted(_VALID_MATERIALS)} (got '{v}')")
        return v.lower()


class ProcurementRunResponse(BaseModel):
    """Response for ``POST /v2/procurement/run``."""

    trace_id: str
    correlation_id: str
    status: str = "completed"
    decision: dict[str, Any] | None = None
    buyer_result: dict[str, Any] | None = None


class ProcurementStatusResponse(BaseModel):
    """Response for ``GET /v2/procurement/{trace_id}/status``.

    Pulls the current state from :meth:`DistributedBlackboard.snapshot` /
    :meth:`DistributedBlackboard.stats`.
    """

    trace_id: str
    correlation_id: str
    status: str
    decision: dict[str, Any] | None = None
    channels: dict[str, int] = Field(default_factory=dict)
    blackboard_stats: dict[str, Any] = Field(default_factory=dict)


class ProcurementErrorResponse(BaseModel):
    """Error envelope returned on failure."""

    trace_id: str | None = None
    status: str = "error"
    error: str
    detail: str | None = None
