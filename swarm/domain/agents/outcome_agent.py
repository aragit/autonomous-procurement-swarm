"""OutcomeAgent — records a post-decision procurement outcome.

Reacts to a ``RecordProcurementOutcome`` message (the external feedback that a
selected supplier actually delivered), validates it, persists an
:class:`OutcomeArtifact` whose ``parent_ids`` reference the originating
:class:`DecisionArtifact` (by id), and publishes an ``OutcomeRecorded`` event
that the :class:`SupplierIntelligenceAgent` consumes.

Outcome recording is deterministic and append-only: it never modifies prior
artifacts or recomputes past decisions, it only adds an auditable record of
what happened after an award.
"""

from typing import Any

import structlog

from swarm.core.agent import BaseAgent
from swarm.core.capability import Capability
from swarm.core.event import Event, SwarmEventType
from swarm.core.state import SwarmState
from swarm.domain.artifacts import (
    OutcomeArtifact,
    outcome_artifact_name,
)
from swarm.domain.events import RECORD_OUTCOME_INTENT, ProcurementEventType

logger = structlog.get_logger(__name__)

_OUTCOME_FIELDS = (
    "supplier_id",
    "decision_id",
    "delivered_on_time",
    "quality_score",
    "actual_price",
    "carbon_score",
)


class OutcomeAgent(BaseAgent):
    """Records procurement outcomes as auditable artifacts."""

    name = "outcome_agent"
    description = "Records post-decision procurement outcomes for supplier intelligence"
    capabilities = [
        Capability(
            name="procurement.outcome.record",
            description="Validates and persists a procurement outcome artifact",
        )
    ]

    def __init__(self) -> None:
        super().__init__()
        self._payload: dict[str, Any] = {}
        self._correlation_id: str | None = None
        self._pending = False

    async def perceive(self, event: Event) -> None:
        if event.replayed:
            return
        if (
            event.type == SwarmEventType.MESSAGE
            and event.message is not None
            and event.message.intent == RECORD_OUTCOME_INTENT
        ):
            self._payload = dict(event.message.payload)
            self._correlation_id = event.correlation_id or event.message.correlation_id
            self._pending = True

    async def reason(self, state: SwarmState) -> None:
        if not self._pending:
            return
        missing = [field for field in _OUTCOME_FIELDS if field not in self._payload]
        if missing:
            logger.warning(
                "outcome_rejected",
                agent=self.name,
                reason="missing_fields",
                missing=missing,
                correlation_id=self._correlation_id,
            )
            self._pending = False

    async def act(self, state: SwarmState) -> None:
        if not self._pending:
            return
        decision_id = str(self._payload["decision_id"])
        supplier_id = str(self._payload["supplier_id"])
        artifact = OutcomeArtifact(
            name=outcome_artifact_name(decision_id),
            data={
                "supplier_id": supplier_id,
                "decision_id": decision_id,
                "delivered_on_time": bool(self._payload["delivered_on_time"]),
                "quality_score": float(self._payload["quality_score"]),
                "actual_price": float(self._payload["actual_price"]),
                "carbon_score": float(self._payload["carbon_score"]),
            },
            parent_ids=[decision_id],
            tags={"supplier": supplier_id},
            created_by=self.name,
            correlation_id=self._correlation_id,
        )
        state.put_artifact(artifact)
        logger.info(
            "artifact_created",
            agent=self.name,
            kind=artifact.kind,
            name=artifact.name,
            supplier_id=supplier_id,
            correlation_id=self._correlation_id,
        )
        await self.publish_event(
            Event(
                type=ProcurementEventType.OUTCOME_RECORDED,
                source=self.name,
                payload={
                    "supplier_id": supplier_id,
                    "decision_id": decision_id,
                    "artifact": artifact.name,
                },
                correlation_id=self._correlation_id,
            )
        )
        self._pending = False
        self._payload = {}
