"""PurchaseOrderAgent — creates a purchase order from an authorized decision.

Reacts to ``ApprovalGranted`` and the ``Execute`` intent message. Consumes the
:class:`ExecutionAuthorizationArtifact` (which must be ``authorized`` — a
rejected or pending decision produces no order) and writes a
:class:`PurchaseOrderArtifact` lineaged to the authorization. It then publishes
``PurchaseOrderCreated`` so the :class:`ExecutionTrackingAgent` can track the
order's lifecycle.

Order creation is deterministic: the same decision + authorization +
requirement + quote always yields the same order (stable id derived from the
decision). A supplier connector (``MockSupplierConnector`` by default) submits
the order and reports a deterministic initial status — no LLM, no autonomous
action.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, cast

import structlog

from swarm.core.agent import BaseAgent
from swarm.core.capability import Capability
from swarm.core.event import Event
from swarm.core.state import SwarmState
from swarm.domain.artifacts import (
    EXECUTION_AUTHORIZATION_ARTIFACT_NAME,
    PURCHASE_ORDER_ARTIFACT_NAME,
    REQUIREMENT_ARTIFACT_NAME,
    PurchaseOrderArtifact,
)
from swarm.domain.events import EXECUTE_INTENT, ProcurementEventType
from swarm.domain.order import (
    PurchaseOrder,
    PurchaseStatus,
    SupplierConnector,
    default_connector,
    order_id_for,
    order_to_dict,
)
from swarm.integrations.base import (
    BaseConnector,
    ExternalResponse,
    record_external_call,
)
from swarm.utils.idempotency import IdempotencyGuard

logger = structlog.get_logger(__name__)


def _connector_system(connector: object) -> str:
    system = getattr(connector, "system", None)
    if isinstance(system, str):
        return system
    return type(connector).__name__.lower()


def _external_to_status(response: ExternalResponse) -> PurchaseStatus:
    """Translate an external submission status onto the canonical PurchaseStatus."""
    status_str = str(response.status)
    for member in PurchaseStatus:
        if member.value == status_str:
            return member
    return PurchaseStatus.SUBMITTED


class PurchaseOrderAgent(BaseAgent):
    """Turns an authorized decision into a purchase order."""

    name = "purchase_order_agent"
    description = "Creates a purchase order once a decision is execution-authorized"
    capabilities = [
        Capability(
            name="procurement.order.create",
            description="Creates a purchase order from an authorized decision",
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
        self._authorization_artifact: str = EXECUTION_AUTHORIZATION_ARTIFACT_NAME

    async def perceive(self, event: Event) -> None:
        if event.replayed:
            return
        if event.type == ProcurementEventType.APPROVAL_GRANTED:
            self._pending = True
            self._correlation_id = event.correlation_id
            self._authorization_artifact = str(
                event.payload.get("artifact") or EXECUTION_AUTHORIZATION_ARTIFACT_NAME
            )
        elif event.type == "message":
            message = event.message
            if message is not None and message.intent == EXECUTE_INTENT:
                self._pending = True
                self._correlation_id = event.correlation_id

    async def reason(self, state: SwarmState) -> None:
        return

    async def act(self, state: SwarmState) -> None:
        if not self._pending:
            return
        artifact = self.create_order(state)
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
            correlation_id=self._correlation_id,
        )
        await self.publish_event(
            Event(
                type=ProcurementEventType.PURCHASE_ORDER_CREATED,
                source=self.name,
                payload={
                    "artifact": artifact.name,
                    "order_id": artifact.data["order_id"],
                    "decision_id": artifact.data["decision_id"],
                },
                correlation_id=self._correlation_id,
            )
        )
        self._pending = False
        self._correlation_id = None

    def create_order(self, state: SwarmState) -> PurchaseOrderArtifact | None:
        """Create a purchase order from the remembered authorization (Phase 7).

        Used by the live :meth:`act` flow (on ``ApprovalGranted``) and by the
        ``POST /swarm/{request_id}/execute`` endpoint (which resolves a pending
        authorization on an ephemeral swarm). Returns the
        :class:`PurchaseOrderArtifact`, or ``None`` when the decision is not
        authorized or an order already exists (idempotent).
        """
        authorization = state.get_artifact(self._authorization_artifact)
        if authorization is None:
            return None
        if authorization.data.get("authorization_status") != "authorized":
            logger.info(
                "agent_executing",
                agent=self.name,
                phase="order_blocked",
                reason="authorization_not_granted",
                authorization_status=authorization.data.get("authorization_status"),
            )
            return None

        if state.find_artifacts(kind="purchase_order"):
            existing = state.get_artifact(PURCHASE_ORDER_ARTIFACT_NAME)
            return cast(PurchaseOrderArtifact, existing) if existing is not None else None

        decision = state.get_artifact("decision")
        requirement = state.get_artifact(REQUIREMENT_ARTIFACT_NAME)
        if decision is None or requirement is None:
            return None
        supplier_id = str(decision.data.get("selected_supplier", ""))
        quote = state.get_artifact(f"quote_{supplier_id}") or state.get_artifact(
            f"quote_{supplier_id.lower()}"
        )

        order = PurchaseOrder.from_artifacts(
            order_id=order_id_for(str(decision.id)),
            decision_artifact=decision,
            authorization_artifact=authorization,
            requirement_artifact=requirement,
            quote_artifact=quote,
        )
        status, reference_id, _ = self._submit_order(state, order, str(decision.id))
        order = replace(order, status=status, submitted_at=datetime.now(UTC).isoformat())
        data = order_to_dict(order)
        data["purchase_order_id"] = data["order_id"]
        if reference_id:
            data["reference_id"] = reference_id
        artifact = PurchaseOrderArtifact(
            data=data,
            parent_ids=[authorization.id],
            tags={
                "decision": str(decision.id),
                "supplier": supplier_id,
                "authorization": str(authorization.id),
            },
            created_by=self.name,
            correlation_id=self._correlation_id or authorization.correlation_id,
        )
        state.put_artifact(artifact)
        return artifact

    def _submit_order(
        self, state: SwarmState, order: PurchaseOrder, decision_id: str
    ) -> tuple[PurchaseStatus, str, dict[str, Any]]:
        """Submit the order via the configured connector, emitting an audit record.

        Uses the :class:`BaseConnector` path when a ``base_connector`` is
        configured: the call is deduplicated by the :class:`IdempotencyGuard`,
        emits an :class:`ExternalCallArtifact`, and returns a canonical
        :class:`PurchaseStatus`. Otherwise it falls back to the legacy
        :class:`SupplierConnector` path for backward compatibility.
        """
        if self._base_connector is not None:
            system = _connector_system(self._base_connector)
            if self._guard.check(decision_id, "submit_order"):
                existing = state.get_artifact(PURCHASE_ORDER_ARTIFACT_NAME)
                if existing is not None:
                    current = PurchaseStatus(str(existing.data.get("status", order.status.value)))
                    return (
                        current,
                        str(existing.data.get("reference_id", "")),
                        {},
                    )
            response = self._base_connector.submit_order(order)
            self._guard.mark(decision_id, "submit_order")
            record_external_call(
                state,
                system=system,
                action="submit_order",
                request_payload={
                    "order_id": order.order_id,
                    "supplier_id": order.supplier_id,
                },
                response_payload=(
                    response.__dict__ if isinstance(response, ExternalResponse) else {}
                ),
                status="success" if response.success else "error",
                order_id=order.order_id,
                decision_id=decision_id,
                idempotency_key=f"{decision_id}:submit_order",
                correlation_id=self._correlation_id,
                created_by=self.name,
            )
            submitted = _external_to_status(response)
            return submitted, response.reference_id, {}
        status = self._connector.submit_order(order)
        return status, "", {}
