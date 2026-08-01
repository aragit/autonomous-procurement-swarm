"""Pydantic schemas for Contract Net Protocol messages."""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class MessageType(StrEnum):
    # CNP Auction Phase
    RFQ = "rfq"
    BID = "bid"
    BID_BOND = "bid_bond"
    AWARD = "award"
    REJECT_BID = "reject_bid"

    # Bilateral Bartering Phase
    OFFER = "offer"
    COUNTER = "counter"
    ACCEPT = "accept"
    REJECT = "reject"

    # System
    ALERT = "alert"
    VALIDATE = "validate"


class RFQPayload(BaseModel):
    """Request for Quote broadcast by buyer."""

    session_id: str
    material: str
    quantity: int = Field(gt=0)
    max_unit_price: float = Field(gt=0)
    target_lead_time_days: int = Field(gt=0)
    delivery_window_start: str  # YYYY-MM-DD
    delivery_window_end: str  # YYYY-MM-DD
    payment_terms: str = Field(pattern=r"^(net_30|net_60|cod|letter_of_credit)$")
    required_bid_bond_pct: float = Field(default=0.05, ge=0.0, le=1.0)

    @field_validator("delivery_window_end")
    @classmethod
    def end_after_start(cls, v: str, info: ValidationInfo) -> str:
        start = info.data.get("delivery_window_start")
        if start and v < start:
            raise ValueError("delivery_window_end must be after delivery_window_start")
        return v


class BidPayload(BaseModel):
    """Sealed bid response from supplier."""

    session_id: str
    supplier_id: str
    unit_price: float = Field(gt=0)
    lead_time_days: int = Field(gt=0)
    carbon_footprint_kg: float = Field(ge=0)
    reliability_score: float = Field(ge=0.0, le=1.0)
    bid_bond_amount: float = Field(ge=0)
    delivery_date: str  # YYYY-MM-DD
    justification: str | None = None


class AwardPayload(BaseModel):
    """Contract award notification to winning supplier."""

    session_id: str
    supplier_id: str
    unit_price: float
    quantity: int
    delivery_date: str
    payment_terms: str


class RejectBidPayload(BaseModel):
    """Formal rejection to non-winning supplier."""

    session_id: str
    supplier_id: str
    reason: str


class CNPMessage(BaseModel):
    """Wrapper for all CNP messages."""

    type: MessageType
    payload: dict[str, Any]
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    @classmethod
    def from_payload(
        cls, msg_type: MessageType, payload: BaseModel | dict[str, Any]
    ) -> "CNPMessage":
        if isinstance(payload, BaseModel):
            payload = payload.model_dump()
        return cls(type=msg_type, payload=payload)
