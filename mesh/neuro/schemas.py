"""Pydantic schemas for schema-constrained ("neuro") generation.

Each archetype that emits LLM-generated content owns a structured response
model here.  The models are passed to the LLM backend in ``structured`` mode so
the backend can coerce (or fail-fast) raw model output into a validated Pydantic
object before it ever reaches the :class:`~mesh.actors.base.SafetyKernelActor`.

These schemas are intentionally ray-free so they can be imported and unit-tested
in isolation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

_VALID_MATERIALS = {"steel", "aluminum", "copper", "plastic", "lumber", "rubber"}
_VALID_PAYMENT_TERMS = {"net_30", "net_60", "cod", "letter_of_credit"}


class SupplierDiscoveryItem(BaseModel):
    """A single supplier discovered by the ScoutActor's LLM path."""

    supplier_id: str = Field(..., min_length=1)
    material: str = Field(..., min_length=1)
    base_cost_per_unit: float = Field(..., gt=0)
    logistics_premium_per_unit: float = Field(..., ge=0)
    capacity_units: int = Field(..., gt=0)
    current_utilization: float = Field(..., ge=0.0, le=1.0)
    min_margin_pct: float = Field(..., ge=0.0, le=1.0)
    reliability_score: float = Field(..., ge=0.0, le=1.0)
    esg_carbon_per_unit: float = Field(..., ge=0.0)

    @field_validator("material")
    @classmethod
    def validate_material(cls, v: str) -> str:
        if v not in _VALID_MATERIALS:
            raise ValueError(f"material must be one of {sorted(_VALID_MATERIALS)} (got '{v}')")
        return v


class ScoutProposal(BaseModel):
    """Structured output for the ScoutActor.

    Replaces the deterministic :func:`scout._build_pool` when a neuro backend is
    configured.  The full supplier pool is produced as one validated payload.
    """

    correlation_id: str = Field(..., min_length=1)
    requirement_trace_id: str = Field(default="")
    material: str = Field(..., min_length=1)
    quantity: int = Field(..., gt=0)
    target_lead_time_days: int = Field(..., gt=0)
    spot_price: float = Field(..., gt=0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    suppliers: list[SupplierDiscoveryItem] = Field(..., min_length=1)

    @field_validator("material")
    @classmethod
    def validate_material(cls, v: str) -> str:
        if v not in _VALID_MATERIALS:
            raise ValueError(f"material must be one of {sorted(_VALID_MATERIALS)} (got '{v}')")
        return v

    def to_pool_dict(self) -> dict[str, Any]:
        """Render the structured proposal as the pool dict the pipeline expects."""
        return {
            "material": self.material,
            "quantity": self.quantity,
            "target_lead_time_days": self.target_lead_time_days,
            "spot_price": self.spot_price,
            "suppliers": [s.model_dump() for s in self.suppliers],
        }

    def to_kernel_payload(self) -> dict[str, Any]:
        """Render as a flat payload the SafetyKernelActor can validate.

        The kernel validates aggregate economic bounds (price, lead time,
        payment terms, material, budget, confidence).  For a scout pool we
        expose the spot-derived ask price, the target lead time, the material and
        the LLM confidence so the kernel can gate obviously unsafe discoveries.
        """
        return {
            "material": self.material,
            "price": self.spot_price,
            "lead_time_days": self.target_lead_time_days,
            "quantity": self.quantity,
            "payment_terms": "net_30",
            "confidence": self.confidence,
        }


class QuoteMetadata(BaseModel):
    """Metadata accompanying a NegotiatorActor quote."""

    quantity: int = Field(..., gt=0)
    lead_time_days: int = Field(..., gt=0)
    carbon_footprint_kg: float = Field(..., ge=0.0)
    reliability_score: float = Field(..., ge=0.0, le=1.0)


class NegotiatorQuote(BaseModel):
    """Structured quote produced by the NegotiatorActor's LLM path."""

    supplier_id: str = Field(..., min_length=1)
    price: float = Field(..., gt=0)
    terms: str = Field(default="net_30")
    metadata: QuoteMetadata

    @field_validator("terms")
    @classmethod
    def validate_terms(cls, v: str) -> str:
        if v not in _VALID_PAYMENT_TERMS:
            raise ValueError(f"terms must be one of {sorted(_VALID_PAYMENT_TERMS)} (got '{v}')")
        return v

    def to_kernel_payload(
        self, material: str, quantity: int, budget: float | None
    ) -> dict[str, Any]:
        """Render as a flat payload the SafetyKernelActor can validate."""
        return {
            "supplier_id": self.supplier_id,
            "material": material,
            "price": self.price,
            "payment_terms": self.terms,
            "quantity": quantity,
            "lead_time_days": self.metadata.lead_time_days,
            "budget": budget if budget is not None else 0.0,
            "total_price": round(self.price * quantity, 2),
            "confidence": 1.0,
        }


class NegotiatorProposal(BaseModel):
    """Structured output for the NegotiatorActor per-supplier quote."""

    correlation_id: str = Field(..., min_length=1)
    supplier_id: str = Field(..., min_length=1)
    eval_trace_id: str = Field(default="")
    pool_trace_id: str = Field(default="")
    quote: NegotiatorQuote
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    def to_kernel_payload(
        self, material: str, quantity: int, budget: float | None
    ) -> dict[str, Any]:
        """Flat payload the SafetyKernelActor validates for a quote."""
        return {
            "supplier_id": self.supplier_id,
            "material": material,
            "price": self.quote.price,
            "payment_terms": self.quote.terms,
            "quantity": quantity,
            "lead_time_days": self.quote.metadata.lead_time_days,
            "budget": budget if budget is not None else 0.0,
            "total_price": round(self.quote.price * quantity, 2),
            "confidence": self.confidence,
        }
