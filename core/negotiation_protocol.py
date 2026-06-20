"""
Structured negotiation protocol.
Defines message schema, validation, and state transitions.
"""

import json
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class MessageType(Enum):
    OFFER = "offer"
    COUNTER = "counter"
    ACCEPT = "accept"
    REJECT = "reject"
    REPORT = "report"
    ALERT = "alert"
    VALIDATE = "validate"
    PERMIT = "permit"
    DECLARE = "declare"
    SANCTION = "sanction"


@dataclass
class NegotiationMessage:
    type: MessageType
    material: str
    quantity: Optional[int] = None
    unit_price: Optional[float] = None
    counter_price: Optional[float] = None
    final_price: Optional[float] = None
    delivery_date: Optional[str] = None
    payment_terms: Optional[str] = None
    justification: Optional[str] = None
    reason: Optional[str] = None
    deadline: Optional[str] = None
    agent: Optional[str] = None
    message_valid: Optional[bool] = None
    errors: List[str] = field(default_factory=list)
    outcome: Optional[str] = None
    terms: Optional[Dict] = None
    severity: Optional[str] = None
    event: Optional[str] = None
    expected_impact: Optional[str] = None
    recommendation: Optional[str] = None
    violation: Optional[str] = None
    penalty: Optional[str] = None
    trend: Optional[str] = None
    volatility: Optional[float] = None
    geo_risk: Optional[float] = None
    supply_disruption: Optional[bool] = None
    raw_json: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_llm_output(cls, text: str) -> "NegotiationMessage":
        """Parse LLM JSON output into structured message."""
        # Extract JSON from potential markdown wrappers
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from text with regex
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON in LLM output: {text[:200]}")
            else:
                raise ValueError(f"No JSON found in LLM output: {text[:200]}")

        msg_type = MessageType(data.get("type", "reject"))

        return cls(
            type=msg_type,
            material=data.get("material", "unknown"),
            quantity=data.get("quantity"),
            unit_price=data.get("unit_price"),
            counter_price=data.get("counter_price"),
            final_price=data.get("final_price"),
            delivery_date=data.get("delivery_date"),
            payment_terms=data.get("payment_terms"),
            justification=data.get("justification"),
            reason=data.get("reason"),
            deadline=data.get("deadline"),
            agent=data.get("agent"),
            message_valid=data.get("message_valid"),
            errors=data.get("errors", []),
            outcome=data.get("outcome"),
            terms=data.get("terms"),
            severity=data.get("severity"),
            event=data.get("event"),
            expected_impact=data.get("expected_impact"),
            recommendation=data.get("recommendation"),
            violation=data.get("violation"),
            penalty=data.get("penalty"),
            trend=data.get("trend"),
            volatility=data.get("volatility"),
            geo_risk=data.get("geo_risk"),
            supply_disruption=data.get("supply_disruption"),
            raw_json=data,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for logging/ledger."""
        d = {
            "type": self.type.value,
            "material": self.material,
        }
        for field_name in [
            "quantity", "unit_price", "counter_price", "final_price",
            "delivery_date", "payment_terms", "justification", "reason",
            "deadline", "agent", "message_valid", "errors", "outcome",
            "terms", "severity", "event", "expected_impact", "recommendation",
            "violation", "penalty", "trend", "volatility", "geo_risk",
            "supply_disruption",
        ]:
            val = getattr(self, field_name)
            if val is not None:
                d[field_name] = val
        return d


class ProtocolValidator:
    """Validates negotiation messages against business rules."""

    VALID_MATERIALS = {"steel", "aluminum", "copper", "plastic", "lumber", "rubber"}
    VALID_PAYMENT_TERMS = {"net_30", "net_60", "cod", "letter_of_credit"}
    MAX_TURNS = 10
    MAX_PRICE = 1000000.0
    MIN_PRICE = 0.01

    @staticmethod
    def validate(msg: NegotiationMessage) -> List[str]:
        """Return list of validation errors. Empty list = valid."""
        errors = []

        # Material check
        if msg.material not in ProtocolValidator.VALID_MATERIALS:
            errors.append(f"Invalid material: {msg.material}")

        # Price checks
        price = msg.unit_price or msg.counter_price or msg.final_price
        if price is not None:
            if price <= 0:
                errors.append(f"Price must be positive: {price}")
            if price > ProtocolValidator.MAX_PRICE:
                errors.append(f"Price exceeds maximum: {price}")

        # Quantity check
        if msg.quantity is not None and msg.quantity <= 0:
            errors.append(f"Quantity must be positive: {msg.quantity}")

        # Payment terms check
        if msg.payment_terms and msg.payment_terms not in ProtocolValidator.VALID_PAYMENT_TERMS:
            errors.append(f"Invalid payment terms: {msg.payment_terms}")

        # Date format check (basic)
        if msg.delivery_date:
            import datetime
            try:
                datetime.datetime.strptime(msg.delivery_date, "%Y-%m-%d")
            except ValueError:
                errors.append(f"Invalid date format: {msg.delivery_date}")

        return errors

    @staticmethod
    def is_terminal(msg: NegotiationMessage) -> bool:
        """Check if message ends negotiation."""
        return msg.type in {MessageType.ACCEPT, MessageType.REJECT, MessageType.DECLARE}
