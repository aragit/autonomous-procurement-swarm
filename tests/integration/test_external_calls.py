"""Integration tests for ExternalCallArtifact emission and idempotency (Phase 8).

Verifies that every outbound connector call is audited, that the idempotency
guard prevents duplicate external side effects, and that replayed events never
re-invoke a connector.
"""

import pytest

from swarm import Event, SwarmState
from swarm.core.artifact import Artifact
from swarm.domain.agents import ExecutionTrackingAgent, PurchaseOrderAgent
from swarm.domain.artifacts import (
    EXECUTION_AUTHORIZATION_ARTIFACT_NAME,
)
from swarm.domain.events import ProcurementEventType
from swarm.integrations.base import ExternalResponse, ExternalStatus
from tests.unit.procurement_helpers import drive


class SpyConnector:
    """Counting BaseConnector that records every call for assertion."""

    system = "spy"

    def __init__(self) -> None:
        self.submitted: list = []
        self.status_calls: list = []

    def submit_order(self, order) -> ExternalResponse:
        self.submitted.append(order)
        return ExternalResponse(
            success=True,
            order_id=order.order_id,
            status="SUBMITTED",
            reference_id=f"SPY-{order.order_id}",
            payload={},
        )

    def get_order_status(self, order_id: str) -> ExternalStatus:
        self.status_calls.append(order_id)
        return ExternalStatus(
            order_id=order_id,
            status="DELIVERED",
            lifecycle=["SUBMITTED", "CONFIRMED", "SHIPPED", "DELIVERED"],
            reference_id=f"SPY-{order_id}",
        )

    def validate_supplier(self, supplier_id: str) -> bool:
        return True


def _seed(state: SwarmState, status: str = "authorized") -> None:
    state.put_artifact(
        Artifact(
            id="auth-1",
            kind=EXECUTION_AUTHORIZATION_ARTIFACT_NAME,
            name=EXECUTION_AUTHORIZATION_ARTIFACT_NAME,
            data={
                "decision_id": "dec-1",
                "authorization_status": status,
                "approved_by": "governance_sim",
            },
            correlation_id="REQ-EXT-01",
        )
    )
    state.put_artifact(
        Artifact(
            id="dec-1",
            kind="decision",
            name="decision",
            data={"selected_supplier": "MinerCorp_A", "reasoning": {"ranked": []}},
            correlation_id="REQ-EXT-01",
        )
    )
    state.put_artifact(
        Artifact(
            kind="requirement",
            name="requirement",
            data={
                "constraints": {
                    "material": "aluminum",
                    "quantity": 100,
                    "budget": 500_000.0,
                    "target_lead_time_days": 30,
                }
            },
            correlation_id="REQ-EXT-01",
        )
    )
    state.put_artifact(
        Artifact(
            kind="quote",
            name="quote_MinerCorp_A",
            data={"supplier_id": "MinerCorp_A", "price": 984.0, "currency": "USD"},
            correlation_id="REQ-EXT-01",
        )
    )


def _po_created_event(order_id: str = "PO-dec-1") -> Event:
    return Event(
        type=ProcurementEventType.PURCHASE_ORDER_CREATED,
        source="purchase_order_agent",
        payload={"artifact": "purchase_order", "order_id": order_id, "decision_id": "dec-1"},
        correlation_id="REQ-EXT-01",
    )


def _approval_granted_event() -> Event:
    return Event(
        type=ProcurementEventType.APPROVAL_GRANTED,
        source="approval_agent",
        payload={
            "artifact": EXECUTION_AUTHORIZATION_ARTIFACT_NAME,
            "decision_id": "dec-1",
            "authorization_status": "authorized",
            "approved_by": "governance_sim",
        },
        correlation_id="REQ-EXT-01",
    )


@pytest.mark.asyncio
async def test_purchase_order_creates_submission_external_call() -> None:
    state = SwarmState()
    _seed(state, status="authorized")
    agent = PurchaseOrderAgent(base_connector=SpyConnector())
    await drive(agent, state, _approval_granted_event())
    ext = state.find_artifacts(kind="external_call")
    assert len(ext) == 1
    assert ext[0].data["system"] == "spy"
    assert ext[0].data["action"] == "submit_order"
    assert ext[0].data["decision_id"] == "dec-1"


@pytest.mark.asyncio
async def test_submission_is_idempotent_no_duplicate_external_call() -> None:
    state = SwarmState()
    _seed(state, status="authorized")
    agent = PurchaseOrderAgent(base_connector=SpyConnector())
    first = agent.create_order(state)
    second = agent.create_order(state)
    assert first is not None and second is not None
    assert first.id == second.id
    ext = state.find_artifacts(kind="external_call")
    assert len(ext) == 1  # guard deduped the second call


@pytest.mark.asyncio
async def test_tracking_emits_external_call_and_is_idempotent() -> None:
    state = SwarmState()
    _seed(state, status="authorized")
    PurchaseOrderAgent(base_connector=SpyConnector()).create_order(state)
    agent = ExecutionTrackingAgent(base_connector=SpyConnector())
    first = agent.track(state)
    second = agent.track(state)
    assert first is not None and second is not None
    assert first.id == second.id
    ext = state.find_artifacts(kind="external_call")
    assert len(ext) == 2  # one submit (from PO), one get_order_status
    assert ext[1].data["action"] == "get_order_status"


@pytest.mark.asyncio
async def test_replayed_purchase_order_event_does_not_call_connector() -> None:
    state = SwarmState()
    _seed(state, status="authorized")
    connector = SpyConnector()
    po_agent = PurchaseOrderAgent(base_connector=connector)
    # Non-replayed create + track -> one submit, one status check.
    await drive(po_agent, state, _approval_granted_event())
    agent = ExecutionTrackingAgent(base_connector=connector)
    await drive(agent, state, _po_created_event())
    assert len(connector.status_calls) == 1
    # Re-dispatching the purchase-order-created event *replayed* must NOT call
    # the connector again (replay stays strictly read-only).
    replayed = _po_created_event().model_copy(update={"replayed": True})
    await drive(agent, state, replayed)
    assert len(connector.status_calls) == 1  # unchanged
    # And no extra external_call artifact was recorded.
    assert len(state.find_artifacts(kind="external_call")) == 2
