"""RequirementAgent — parses procurement requests into a structured requirement.

Wraps the existing RFQ normalization logic (``core.protocol.schema.RFQPayload``
with the delivery-window / payment-terms defaults used by
``BuyerOrchestrator.create_rfq``) so the free-text request becomes a validated,
serializable requirement artifact.
"""

from datetime import datetime, timedelta
from typing import Any

import structlog

from core.market_simulator import MarketSimulator
from core.protocol.schema import RFQPayload
from swarm.core.agent import BaseAgent
from swarm.core.capability import Capability
from swarm.core.event import Event, SwarmEventType
from swarm.core.state import SwarmState
from swarm.domain.artifacts import RequirementArtifact
from swarm.domain.events import CREATE_REQUIREMENT_INTENT, ProcurementEventType

logger = structlog.get_logger(__name__)

VALID_MATERIALS = ("steel", "aluminum", "copper", "plastic", "lumber", "rubber")


class RequirementAgent(BaseAgent):
    """Creates a :class:`RequirementArtifact` from a ``CreateRequirement`` message."""

    name = "requirement_agent"
    description = "Parses procurement requests into a structured requirement artifact"
    capabilities = [
        Capability(
            name="requirement.create",
            description="Creates a validated requirement artifact from a request",
        )
    ]

    def __init__(self) -> None:
        super().__init__()
        self._payload: dict[str, Any] = {}
        self._correlation_id: str | None = None
        self._requirement_data: dict[str, Any] | None = None
        self._pending = False

    async def perceive(self, event: Event) -> None:
        if event.replayed:
            return
        if (
            event.type == SwarmEventType.MESSAGE
            and event.message is not None
            and event.message.intent == CREATE_REQUIREMENT_INTENT
        ):
            self._payload = dict(event.message.payload)
            self._correlation_id = event.correlation_id or event.message.correlation_id
            self._pending = True

    async def reason(self, state: SwarmState) -> None:
        if not self._pending:
            return
        self._requirement_data = self._build_requirement(self._payload, self._correlation_id)
        logger.info(
            "agent_executing",
            agent=self.name,
            phase="requirement_parsed",
            correlation_id=self._correlation_id,
        )

    async def act(self, state: SwarmState) -> None:
        if not self._pending or self._requirement_data is None:
            return
        artifact = RequirementArtifact(
            data=self._requirement_data,
            created_by=self.name,
            correlation_id=self._correlation_id,
        )
        state.put_artifact(artifact)
        logger.info(
            "artifact_created",
            agent=self.name,
            kind=artifact.kind,
            name=artifact.name,
            correlation_id=self._correlation_id,
        )
        await self.publish_event(
            Event(
                type=ProcurementEventType.REQUIREMENT_CREATED,
                source=self.name,
                payload={"artifact": artifact.name, "requirement": self._requirement_data},
                correlation_id=self._correlation_id,
            )
        )
        self._pending = False
        self._requirement_data = None

    @classmethod
    def _build_requirement(
        cls, payload: dict[str, Any], correlation_id: str | None
    ) -> dict[str, Any]:
        """Normalize a request payload into the RequirementArtifact data contract."""
        text = str(payload.get("text") or payload.get("description") or "")
        material = str(payload.get("material") or payload.get("item") or "steel").lower()
        if material not in VALID_MATERIALS:
            material = "steel"
        quantity = int(payload.get("quantity") or 1000)
        budget = float(payload.get("budget") or 500_000.0)
        max_unit_price = payload.get("max_unit_price")
        if max_unit_price is None:
            spot_price = MarketSimulator(seed=42).get_current_state(material).spot_price
            max_unit_price = round(spot_price * 1.2, 2)
        max_unit_price = max(float(max_unit_price or 1.0), 1.0)
        target_lead_time_days = int(payload.get("target_lead_time_days") or 30)

        constraints: dict[str, Any] = {
            "material": material,
            "quantity": quantity,
            "max_unit_price": max_unit_price,
            "target_lead_time_days": target_lead_time_days,
            "budget": budget,
        }
        metadata: dict[str, Any] = {"raw": payload}
        if max_unit_price > 0:
            metadata["rfq"] = cls._build_rfq(
                correlation_id, material, quantity, max_unit_price, target_lead_time_days
            )
        return {"text": text, "constraints": constraints, "metadata": metadata}

    @staticmethod
    def _build_rfq(
        session_id: str | None,
        material: str,
        quantity: int,
        max_unit_price: float,
        target_lead_time_days: int,
    ) -> dict[str, Any]:
        """Validate the structured requirement as a core RFQ (existing logic)."""
        today = datetime.now()
        rfq = RFQPayload(
            session_id=(session_id or "procurement")[:32],
            material=material,
            quantity=quantity,
            max_unit_price=max_unit_price,
            target_lead_time_days=target_lead_time_days,
            delivery_window_start=(today + timedelta(days=14)).strftime("%Y-%m-%d"),
            delivery_window_end=(today + timedelta(days=90)).strftime("%Y-%m-%d"),
            payment_terms="net_30",
        )
        return rfq.model_dump()
