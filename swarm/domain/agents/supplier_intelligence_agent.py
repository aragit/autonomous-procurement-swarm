"""SupplierIntelligenceAgent — updates supplier performance from outcomes.

Reacts to the ``OutcomeRecorded`` event, looks up the originating
:class:`DecisionArtifact` from shared state to recover the supplier's decided
quote price, folds the outcome into a :class:`SupplierMemoryStore`, and writes a
:class:`SupplierPerformanceArtifact` (lineaged to the outcome by id). It then
publishes ``SupplierPerformanceUpdated`` so downstream evaluations can pick up the
new history.

Intelligence is deterministic and append-only: the same sequence of outcomes
always yields the same performance records. There is no autonomous learning —
metrics are running averages maintained by a clean, single-writer update rule.
"""


from typing import Any

import structlog

from swarm.core.agent import BaseAgent
from swarm.core.capability import Capability
from swarm.core.event import Event
from swarm.core.state import SwarmState
from swarm.domain.artifacts import (
    DECISION_ARTIFACT_NAME,
    SupplierPerformanceArtifact,
    supplier_performance_artifact_name,
)
from swarm.domain.events import ProcurementEventType
from swarm.memory import SupplierMemoryStore

logger = structlog.get_logger(__name__)


class SupplierIntelligenceAgent(BaseAgent):
    """Maintains cumulative supplier performance from recorded outcomes."""

    name = "supplier_intelligence_agent"
    description = "Updates supplier performance records from procurement outcomes"
    capabilities = [
        Capability(
            name="supplier.intelligence.update",
            description="Updates supplier performance from an outcome artifact",
        )
    ]

    def __init__(self, *, memory: SupplierMemoryStore | None = None) -> None:
        super().__init__(memory=memory)
        self._correlation_id: str | None = None
        self._outcome_artifact: str = ""
        self._outcome_id: str = ""
        self._pending = False
        self._supplier_id: str = ""
        self._reference_price: float | None = None
        self._performance: Any = None

    async def perceive(self, event: Event) -> None:
        if event.replayed:
            return
        if event.type == ProcurementEventType.OUTCOME_RECORDED:
            self._pending = True
            self._correlation_id = event.correlation_id
            self._outcome_artifact = str(event.payload.get("artifact") or "")
            self._outcome_id = ""
            self._supplier_id = str(event.payload.get("supplier_id") or "")
            self._reference_price = None

    async def reason(self, state: SwarmState) -> None:
        if not self._pending:
            return
        memory = self._memory()
        if memory is None:
            self._pending = False
            return
        outcome = state.get_artifact(self._outcome_artifact)
        if outcome is None:
            self._pending = False
            return
        self._outcome_id = outcome.id
        reference_price = self._reference_quote_price(state, self._supplier_id)

        updated = memory.update_from_outcome(
            {
                "supplier_id": outcome.data.get("supplier_id", self._supplier_id),
                "delivered_on_time": outcome.data.get("delivered_on_time"),
                "quality_score": outcome.data.get("quality_score"),
                "actual_price": outcome.data.get("actual_price"),
                "carbon_score": outcome.data.get("carbon_score"),
            },
            reference_price=reference_price,
        )
        self._reference_price = reference_price
        self._updated = updated
        logger.info(
            "agent_executing",
            agent=self.name,
            phase="supplier_performance_updated",
            supplier_id=updated.supplier_id,
            total_orders=updated.total_orders,
            delivery_reliability=round(updated.delivery_reliability, 4),
            correlation_id=self._correlation_id,
        )

    async def act(self, state: SwarmState) -> None:
        if not self._pending:
            return
        updated = getattr(self, "_updated", None)
        if updated is None:
            return
        metrics = {
            "total_orders": updated.total_orders,
            "successful_orders": updated.successful_orders,
            "average_delivery_score": updated.average_delivery_score,
            "average_quality_score": updated.average_quality_score,
            "average_price_competitiveness": updated.average_price_competitiveness,
            "average_carbon_score": updated.average_carbon_score,
        }
        artifact = SupplierPerformanceArtifact(
            name=supplier_performance_artifact_name(updated.supplier_id),
            data={
                "supplier_id": updated.supplier_id,
                "performance_metrics": metrics,
                "order_count": updated.total_orders,
                "updated_at": updated.last_updated.isoformat(),
            },
            parent_ids=[self._outcome_id] if self._outcome_id else [self._outcome_artifact],
            tags={"supplier": updated.supplier_id},
            created_by=self.name,
            correlation_id=self._correlation_id,
        )
        state.put_artifact(artifact)
        logger.info(
            "artifact_created",
            agent=self.name,
            kind=artifact.kind,
            name=artifact.name,
            supplier_id=updated.supplier_id,
            correlation_id=self._correlation_id,
        )
        await self.publish_event(
            Event(
                type=ProcurementEventType.SUPPLIER_PERFORMANCE_UPDATED,
                source=self.name,
                payload={
                    "supplier_id": updated.supplier_id,
                    "artifact": artifact.name,
                    "total_orders": updated.total_orders,
                    "delivery_reliability": round(updated.delivery_reliability, 4),
                },
                correlation_id=self._correlation_id,
            )
        )
        self._pending = False
        self._updated = None  # type: ignore[assignment]

    def _memory(self) -> SupplierMemoryStore | None:
        """The supplier memory store assigned to this agent."""
        memory = getattr(self, "memory", None)
        if isinstance(memory, SupplierMemoryStore):
            return memory
        return None

    @staticmethod
    def _reference_quote_price(state: SwarmState, supplier_id: str) -> float | None:
        """The decided quote unit price for ``supplier_id`` from the decision, if any."""
        decision = state.get_artifact(DECISION_ARTIFACT_NAME)
        if decision is None:
            return None
        ranked = decision.data.get("reasoning", {}).get("ranked", []) or []
        for entry in ranked:
            if str(entry.get("supplier_id")) == supplier_id:
                return float(entry.get("price") or 0.0) or None
        return None
