"""ExecutionTrackingAgent — tracks the lifecycle of a purchase order.

Reacts to ``PurchaseOrderCreated`` and, using a :class:`SupplierConnector`,
records the order's realized execution status in an :class:`ExecutionStatusArtifact`
(lineaged to the :class:`PurchaseOrderArtifact`). It publishes
``ExecutionStatusUpdated`` so downstream stages (delivery outcome → supplier
intelligence) can fold execution results into supplier performance.

Tracking is a pure read of the connector, so the same order always maps to the
same status record — the execution trace is deterministic and replay-safe.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

import structlog

from swarm.core.agent import BaseAgent
from swarm.core.capability import Capability
from swarm.core.event import Event
from swarm.core.state import SwarmState
from swarm.domain.artifacts import (
    EXECUTION_STATUS_ARTIFACT_NAME,
    PURCHASE_ORDER_ARTIFACT_NAME,
    ExecutionStatusArtifact,
)
from swarm.domain.events import ProcurementEventType
from swarm.domain.order import (
    PurchaseOrder,
    PurchaseStatus,
    SupplierConnector,
    default_connector,
)
from swarm.integrations.base import (
    BaseConnector,
    ExternalStatus,
    record_external_call,
)
from swarm.utils.idempotency import IdempotencyGuard

logger = structlog.get_logger(__name__)


def _connector_system(connector: object) -> str:
    system = getattr(connector, "system", None)
    if isinstance(system, str):
        return system
    return type(connector).__name__.lower()


class ExecutionTrackingAgent(BaseAgent):
    """Tracks the execution lifecycle of a purchase order."""

    name = "execution_tracking_agent"
    description = "Tracks a purchase order's execution lifecycle with the supplier"
    capabilities = [
        Capability(
            name="procurement.execution.track",
            description="Tracks purchase order execution status to completion",
        )
    ]

    def __init__(
        self,
        *,
        connector: SupplierConnector | None = None,
        base_connector: BaseConnector | None = None,
    ) -> None:
        super().__init__()
        self._connector = connector if connector is not None else default_connector
        self._base_connector = base_connector
        self._guard = IdempotencyGuard()
        self._correlation_id: str | None = None
        self._pending = False
        self._order_artifact: str = "purchase_order"

    async def perceive(self, event: Event) -> None:
        if event.replayed:
            return
        if event.type == ProcurementEventType.PURCHASE_ORDER_CREATED:
            self._pending = True
            self._correlation_id = event.correlation_id
            self._order_artifact = str(
                event.payload.get("artifact") or PURCHASE_ORDER_ARTIFACT_NAME
            )

    async def reason(self, state: SwarmState) -> None:
        return

    async def act(self, state: SwarmState) -> None:
        if not self._pending:
            return
        artifact = self.track(state)
        if artifact is None:
            self._pending = False
            self._correlation_id = None
            return
        logger.info(
            "artifact_created",
            agent=self.name,
            kind=artifact.kind,
            name=artifact.name,
            order_id=artifact.data.get("order_id"),
            status=artifact.data.get("status"),
            correlation_id=self._correlation_id,
        )
        await self.publish_event(
            Event(
                type=ProcurementEventType.EXECUTION_STATUS_UPDATED,
                source=self.name,
                payload={
                    "artifact": artifact.name,
                    "order_id": artifact.data["order_id"],
                    "status": artifact.data["status"],
                },
                correlation_id=self._correlation_id,
            )
        )
        self._pending = False
        self._correlation_id = None

    def track(self, state: SwarmState) -> ExecutionStatusArtifact | None:
        """Record the execution status of the remembered purchase order.

        Used by the live :meth:`act` flow (on ``PurchaseOrderCreated``) and by
        the ``POST /swarm/{request_id}/execute`` endpoint (which creates the
        order on an ephemeral swarm and then tracks it). Returns the
        :class:`ExecutionStatusArtifact`, or ``None`` when no purchase order
        exists to track (idempotent — a present status record is returned as-is).
        """
        order_artifact = state.get_artifact(self._order_artifact)
        if order_artifact is None:
            return None
        if state.find_artifacts(kind="execution_status"):
            existing = state.get_artifact(EXECUTION_STATUS_ARTIFACT_NAME)
            return cast(ExecutionStatusArtifact, existing) if existing is not None else None

        order = PurchaseOrder(
            order_id=str(order_artifact.data.get("order_id", "")),
            decision_id=str(order_artifact.data.get("decision_id", "")),
            authorization_id=str(order_artifact.data.get("authorization_id", "")),
            supplier_id=str(order_artifact.data.get("supplier_id", "")),
            currency=str(order_artifact.data.get("currency", "USD")),
            total_amount=float(order_artifact.data.get("total_amount", 0.0)),
            status=PurchaseStatus(order_artifact.data.get("status", PurchaseStatus.CREATED.value)),
            created_at=str(order_artifact.data.get("created_at", "")),
            submitted_at=order_artifact.data.get("submitted_at"),
        )
        decision_id = str(order_artifact.data.get("decision_id", ""))
        status_value, lifecycle = self._track_order(state, order, decision_id)
        artifact = ExecutionStatusArtifact(
            data={
                "order_id": order.order_id,
                "purchase_order_id": order_artifact.id,
                "status": status_value,
                "lifecycle": lifecycle,
                "tracked_at": datetime.now(UTC).isoformat(),
            },
            parent_ids=[order_artifact.id],
            tags={
                "decision": order_artifact.tags.get("decision", order.decision_id),
                "supplier": str(order_artifact.data.get("supplier_id", "")),
            },
            created_by=self.name,
            correlation_id=self._correlation_id or order_artifact.correlation_id,
        )
        state.put_artifact(artifact)
        return artifact

    def _track_order(
        self, state: SwarmState, order: PurchaseOrder, decision_id: str
    ) -> tuple[str, list[str]]:
        """Resolve the order's realized status via the configured connector.

        Uses the :class:`BaseConnector` path when configured (deduplicated by the
        :class:`IdempotencyGuard` and emitting an :class:`ExternalCallArtifact`);
        otherwise falls back to the legacy :class:`SupplierConnector` path.
        """
        if self._base_connector is not None:
            system = _connector_system(self._base_connector)
            if self._guard.check(decision_id, "get_order_status"):
                existing = state.get_artifact(EXECUTION_STATUS_ARTIFACT_NAME)
                if existing is not None:
                    return (
                        str(existing.data.get("status", order.status.value)),
                        list(existing.data.get("lifecycle", [])),
                    )
            external = self._base_connector.get_order_status(order.order_id)
            self._guard.mark(decision_id, "get_order_status")
            record_external_call(
                state,
                system=system,
                action="get_order_status",
                request_payload={"order_id": order.order_id},
                response_payload=(
                    external.__dict__ if isinstance(external, ExternalStatus) else {}
                ),
                status="success" if external.status else "error",
                order_id=order.order_id,
                decision_id=decision_id,
                idempotency_key=f"{decision_id}:get_order_status",
                correlation_id=self._correlation_id,
                created_by=self.name,
            )
            return external.status, list(external.lifecycle)
        status = self._connector.track_order(order)
        return status.value, self._connector.order_lifecycle(order)
