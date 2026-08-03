"""Unit tests for the Phase 7 PurchaseOrderAgent and ExecutionTrackingAgent."""

import pytest

from swarm import Event, SwarmState
from swarm.core.artifact import Artifact
from swarm.domain.agents import ExecutionTrackingAgent, PurchaseOrderAgent
from swarm.domain.artifacts import (
    DECISION_ARTIFACT_NAME,
    EXECUTION_AUTHORIZATION_ARTIFACT_NAME,
    ExecutionAuthorizationArtifact,
    PurchaseOrderArtifact,
    QuoteArtifact,
)
from swarm.domain.events import ProcurementEventType
from swarm.domain.order import PurchaseStatus
from tests.unit.procurement_helpers import drive


class SpyConnector:
    """Deterministic connector that records calls for assertion."""

    def __init__(self) -> None:
        self.submitted: list = []
        self.tracked: list = []

    def submit_order(self, order) -> PurchaseStatus:
        self.submitted.append(order)
        return PurchaseStatus.SUBMITTED

    def track_order(self, order) -> PurchaseStatus:
        self.tracked.append(order)
        return PurchaseStatus.DELIVERED

    def order_lifecycle(self, order) -> list[str]:
        return ["SUBMITTED", "CONFIRMED", "SHIPPED", "DELIVERED"]


def seed_flow(
    state: SwarmState,
    *,
    authorization_status: str = "authorized",
    supplier_id: str = "MinerCorp_A",
    unit_price: float = 984.0,
    quantity: int = 1000,
) -> Artifact:
    state.put_artifact(
        Artifact(
            id="dec-1",
            kind=DECISION_ARTIFACT_NAME,
            name="decision",
            data={"selected_supplier": supplier_id, "reasoning": {"ranked": []}},
            correlation_id="REQ-EXEC-01",
        )
    )
    state.put_artifact(
        Artifact(
            id="req-1",
            kind="requirement",
            name="requirement",
            data={
                "constraints": {
                    "material": "aluminum",
                    "quantity": quantity,
                    "budget": 3_500_000.0,
                    "target_lead_time_days": 30,
                }
            },
            correlation_id="REQ-EXEC-01",
        )
    )
    state.put_artifact(
        QuoteArtifact(
            id="quote-1",
            name=f"quote_{supplier_id}",
            data={
                "supplier_id": supplier_id,
                "price": unit_price,
                "currency": "USD",
                "metadata": {"quantity": quantity, "lead_time_days": 30},
            },
            tags={"supplier": supplier_id},
            correlation_id="REQ-EXEC-01",
        )
    )
    return state.put_artifact(
        ExecutionAuthorizationArtifact(
            id="auth-1",
            name=EXECUTION_AUTHORIZATION_ARTIFACT_NAME,
            data={
                "decision_id": "dec-1",
                "authorization_status": authorization_status,
                "approved_by": "governance_sim" if authorization_status == "authorized" else None,
            },
            correlation_id="REQ-EXEC-01",
        )
    )


def approval_granted_event() -> Event:
    return Event(
        type=ProcurementEventType.APPROVAL_GRANTED,
        source="approval_agent",
        payload={
            "artifact": EXECUTION_AUTHORIZATION_ARTIFACT_NAME,
            "decision_id": "dec-1",
            "authorization_status": "authorized",
            "approved_by": "governance_sim",
        },
        correlation_id="REQ-EXEC-01",
    )


def purchase_order_created_event(order_id: str = "PO-dec-1") -> Event:
    return Event(
        type=ProcurementEventType.PURCHASE_ORDER_CREATED,
        source="purchase_order_agent",
        payload={
            "artifact": "purchase_order",
            "order_id": order_id,
            "decision_id": "dec-1",
        },
        correlation_id="REQ-EXEC-01",
    )


async def drive_order(state: SwarmState, status: str = "authorized") -> SpyConnector:
    connector = SpyConnector()
    agent = PurchaseOrderAgent(connector=connector)
    await drive(agent, state, approval_granted_event())
    return connector


async def drive_execution(state: SwarmState) -> SpyConnector:
    connector = SpyConnector()
    agent = ExecutionTrackingAgent(connector=connector)
    await drive(agent, state, purchase_order_created_event())
    return connector


@pytest.mark.asyncio
async def test_approval_granted_creates_purchase_order() -> None:
    state = SwarmState()
    auth = seed_flow(state, authorization_status="authorized")
    connector = await drive_order(state)

    assert connector.submitted[0].order_id == "PO-dec-1"
    # The connector receives the order in its pre-submission state.
    assert connector.submitted[0].status == PurchaseStatus.CREATED

    orders = state.find_artifacts(kind="purchase_order")
    assert len(orders) == 1
    order = orders[0]
    assert isinstance(order, PurchaseOrderArtifact)
    assert order.parent_ids == [auth.id]
    assert order.data["status"] == "SUBMITTED"
    assert order.data["supplier_id"] == "MinerCorp_A"
    assert order.data["total_amount"] == 984_000.0


@pytest.mark.asyncio
async def test_approval_granted_publishes_purchase_order_created() -> None:
    state = SwarmState()
    seed_flow(state, authorization_status="authorized")
    agent = PurchaseOrderAgent(connector=SpyConnector())
    bus = await drive(agent, state, approval_granted_event())

    created = [e for e in bus.event_log() if e.type == ProcurementEventType.PURCHASE_ORDER_CREATED]
    assert len(created) == 1
    assert created[0].payload["order_id"] == "PO-dec-1"


@pytest.mark.asyncio
async def test_create_order_blocked_when_pending() -> None:
    state = SwarmState()
    seed_flow(state, authorization_status="pending")
    agent = PurchaseOrderAgent(connector=SpyConnector())
    assert agent.create_order(state) is None
    assert state.find_artifacts(kind="purchase_order") == []


@pytest.mark.asyncio
async def test_create_order_blocked_when_rejected() -> None:
    state = SwarmState()
    seed_flow(state, authorization_status="rejected")
    agent = PurchaseOrderAgent(connector=SpyConnector())
    assert agent.create_order(state) is None
    assert state.find_artifacts(kind="purchase_order") == []


@pytest.mark.asyncio
async def test_create_order_blocked_without_authorization() -> None:
    state = SwarmState()
    agent = PurchaseOrderAgent(connector=SpyConnector())
    assert agent.create_order(state) is None
    assert state.find_artifacts(kind="purchase_order") == []


@pytest.mark.asyncio
async def test_create_order_is_idempotent() -> None:
    state = SwarmState()
    seed_flow(state, authorization_status="authorized")
    agent = PurchaseOrderAgent(connector=SpyConnector())
    first = agent.create_order(state)
    second = agent.create_order(state)
    assert first is not None and second is not None
    assert first.id == second.id
    assert len(state.find_artifacts(kind="purchase_order")) == 1


@pytest.mark.asyncio
async def test_purchase_order_agent_skips_replayed_approval() -> None:
    state = SwarmState()
    seed_flow(state, authorization_status="authorized")
    agent = PurchaseOrderAgent(connector=SpyConnector())
    replayed = approval_granted_event()
    replayed = Event(
        type=replayed.type,
        source=replayed.source,
        payload=replayed.payload,
        correlation_id=replayed.correlation_id,
        replayed=True,
    )
    await drive(agent, state, replayed)
    assert state.find_artifacts(kind="purchase_order") == []


@pytest.mark.asyncio
async def test_execution_tracking_creates_status_on_order_created() -> None:
    state = SwarmState()
    seed_flow(state, authorization_status="authorized")
    po_agent = PurchaseOrderAgent(connector=SpyConnector())
    order = po_agent.create_order(state)
    assert order is not None

    connector = await drive_execution(state)

    assert connector.tracked[0].order_id == "PO-dec-1"
    statuses = state.find_artifacts(kind="execution_status")
    assert len(statuses) == 1
    status = statuses[0]
    assert status.parent_ids == [order.id]
    assert status.data["status"] == "DELIVERED"
    assert status.data["lifecycle"] == ["SUBMITTED", "CONFIRMED", "SHIPPED", "DELIVERED"]


@pytest.mark.asyncio
async def test_execution_tracking_publishes_status_updated() -> None:
    state = SwarmState()
    seed_flow(state, authorization_status="authorized")
    PurchaseOrderAgent(connector=SpyConnector()).create_order(state)
    agent = ExecutionTrackingAgent(connector=SpyConnector())
    bus = await drive(agent, state, purchase_order_created_event())

    updated = [
        e
        for e in bus.event_log()
        if e.type == ProcurementEventType.EXECUTION_STATUS_UPDATED
    ]
    assert len(updated) == 1
    assert updated[0].payload["status"] == "DELIVERED"


@pytest.mark.asyncio
async def test_track_returns_none_without_order() -> None:
    state = SwarmState()
    agent = ExecutionTrackingAgent(connector=SpyConnector())
    assert agent.track(state) is None
    assert state.find_artifacts(kind="execution_status") == []


@pytest.mark.asyncio
async def test_track_is_idempotent() -> None:
    state = SwarmState()
    seed_flow(state, authorization_status="authorized")
    PurchaseOrderAgent(connector=SpyConnector()).create_order(state)
    agent = ExecutionTrackingAgent(connector=SpyConnector())
    first = agent.track(state)
    second = agent.track(state)
    assert first is not None and second is not None
    assert first.id == second.id
    assert len(state.find_artifacts(kind="execution_status")) == 1
