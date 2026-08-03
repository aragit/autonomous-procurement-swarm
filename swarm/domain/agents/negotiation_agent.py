"""NegotiationAgent — generates deterministic quotes for each evaluated supplier.

Quotes are derived deterministically from each supplier's CostModel-shaped
profile (the same floor-price rule the CNP auction uses), so a given requirement
always produces the same quotes. No randomness, no LLM.

Phase 3: the agent reacts to each ``SupplierEvaluated`` event individually and
publishes one ``QuoteGenerated`` event per supplier, so quoting runs in
parallel with — and immediately after — each evaluation instead of waiting for
the whole evaluation phase.
"""

from typing import Any, cast

import structlog

from swarm.core.agent import BaseAgent
from swarm.core.capability import Capability
from swarm.core.event import Event
from swarm.core.state import SwarmState
from swarm.domain.artifacts import (
    REQUIREMENT_ARTIFACT_NAME,
    SUPPLIER_LIST_ARTIFACT_NAME,
    QuoteArtifact,
    evaluation_artifact_name,
)
from swarm.domain.events import ProcurementEventType
from swarm.domain.pricing import (
    DEFAULT_PAYMENT_TERMS,
    carbon_footprint,
    floor_price,
    lead_time_days,
)

logger = structlog.get_logger(__name__)


class NegotiationAgent(BaseAgent):
    """Generates a deterministic quote for one evaluated supplier."""

    name = "negotiation_agent"
    description = "Generates deterministic quotes for each evaluated supplier"
    capabilities = [
        Capability(
            name="supplier.negotiate",
            description="Produces a deterministic quote for a supplier",
        )
    ]

    def __init__(self) -> None:
        super().__init__()
        self._correlation_id: str | None = None
        self._list_artifact: str = SUPPLIER_LIST_ARTIFACT_NAME
        self._supplier_id: str = ""
        self._quote: dict[str, Any] | None = None
        self._pending = False

    async def perceive(self, event: Event) -> None:
        if event.replayed:
            return
        if event.type == ProcurementEventType.SUPPLIER_EVALUATED:
            self._pending = True
            self._correlation_id = event.correlation_id
            self._supplier_id = str(event.payload.get("supplier_id") or "")
            self._quote = None

    async def reason(self, state: SwarmState) -> None:
        if not self._pending:
            return
        pool = state.get_artifact(self._list_artifact)
        requirement = state.get_artifact(REQUIREMENT_ARTIFACT_NAME)
        if pool is None or requirement is None:
            self._pending = False
            return
        supplier = self._find_supplier(pool, self._supplier_id)
        if supplier is None:
            self._pending = False
            return
        constraints = requirement.data.get("constraints", {})
        quantity = int(pool.data.get("quantity") or constraints.get("quantity") or 1000)
        target_lead = int(
            pool.data.get("target_lead_time_days") or constraints.get("target_lead_time_days") or 30
        )
        index = self._supplier_index(pool, self._supplier_id)

        self._quote = {
            "supplier_id": self._supplier_id,
            "price": floor_price(supplier),
            "terms": DEFAULT_PAYMENT_TERMS,
            "metadata": {
                "quantity": quantity,
                "lead_time_days": lead_time_days(supplier, target_lead, index),
                "carbon_footprint_kg": carbon_footprint(supplier, quantity),
                "reliability_score": supplier["reliability_score"],
            },
        }
        logger.info(
            "agent_executing",
            agent=self.name,
            phase="quote_generated",
            supplier_id=self._supplier_id,
            price=self._quote["price"],
            correlation_id=self._correlation_id,
        )

    async def act(self, state: SwarmState) -> None:
        if not self._pending or self._quote is None:
            return
        artifact = QuoteArtifact(
            name=f"quote_{self._supplier_id}",
            data=self._quote,
            parent_ids=[evaluation_artifact_name(self._supplier_id)],
            tags={"supplier": self._supplier_id},
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
                type=ProcurementEventType.QUOTE_GENERATED,
                source=self.name,
                payload={
                    "supplier_id": self._supplier_id,
                    "artifact": artifact.name,
                    "price": self._quote["price"],
                },
                correlation_id=self._correlation_id,
            )
        )
        self._pending = False
        self._quote = None

    @staticmethod
    def _find_supplier(pool: Any, supplier_id: str) -> dict[str, Any] | None:
        """The pool entry matching ``supplier_id``, or None."""
        for supplier in pool.data["suppliers"]:
            if str(supplier["supplier_id"]) == supplier_id:
                return cast("dict[str, Any]", supplier)
        return None

    @staticmethod
    def _supplier_index(pool: Any, supplier_id: str) -> int:
        """Position of ``supplier_id`` in the pool (stable ordering)."""
        for index, supplier in enumerate(pool.data["suppliers"]):
            if str(supplier["supplier_id"]) == supplier_id:
                return index
        return 0
