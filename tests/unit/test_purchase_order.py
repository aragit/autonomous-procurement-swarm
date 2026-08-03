"""Unit tests for the Phase 7 deterministic purchase-order domain model."""

from datetime import UTC, datetime

from swarm import SwarmState
from swarm.core.artifact import Artifact
from swarm.domain.artifacts import (
    EXECUTION_AUTHORIZATION_ARTIFACT_NAME,
    PURCHASE_ORDER_ARTIFACT_NAME,
    PurchaseOrderArtifact,
)
from swarm.domain.order import (
    MockSupplierConnector,
    PurchaseOrder,
    PurchaseStatus,
    order_id_for,
    order_to_dict,
)


def _seed_artifacts(
    state: SwarmState,
    *,
    decision_id: str = "dec-1",
    authorization_id: str = "auth-1",
    supplier_id: str = "MinerCorp_A",
    unit_price: float = 984.0,
    quantity: int = 1000,
) -> dict[str, Artifact]:
    decision = state.put_artifact(
        Artifact(
            id=decision_id,
            kind="decision",
            name="decision",
            data={"selected_supplier": supplier_id, "reasoning": {"ranked": []}},
            correlation_id="REQ-PO-01",
        )
    )
    authorization = state.put_artifact(
        Artifact(
            id=authorization_id,
            kind="execution_authorization",
            name=EXECUTION_AUTHORIZATION_ARTIFACT_NAME,
            data={
                "decision_id": decision_id,
                "authorization_status": "authorized",
                "approved_by": "governance_sim",
            },
            correlation_id="REQ-PO-01",
        )
    )
    requirement = state.put_artifact(
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
            correlation_id="REQ-PO-01",
        )
    )
    quote = state.put_artifact(
        Artifact(
            id=f"quote-{supplier_id}",
            kind="quote",
            name=f"quote_{supplier_id}",
            data={
                "supplier_id": supplier_id,
                "price": unit_price,
                "currency": "USD",
                "metadata": {"quantity": quantity, "lead_time_days": 30},
            },
            tags={"supplier": supplier_id},
            correlation_id="REQ-PO-01",
        )
    )
    return {
        "decision": decision,
        "authorization": authorization,
        "requirement": requirement,
        "quote": quote,
    }


def test_order_id_for_is_stable_and_deterministic() -> None:
    assert order_id_for("dec-1") == "PO-dec-1"
    assert order_id_for("dec-1") == order_id_for("dec-1")


def test_purchase_order_statuses_round_trip() -> None:
    assert PurchaseStatus("SUBMITTED").value == "SUBMITTED"
    assert PurchaseStatus.DELIVERED.value == "DELIVERED"
    assert {s.value for s in PurchaseStatus} == {
        "CREATED",
        "SUBMITTED",
        "CONFIRMED",
        "SHIPPED",
        "DELIVERED",
        "CANCELLED",
    }


def test_purchase_order_from_artifacts_uses_decision_and_quote() -> None:
    state = SwarmState()
    artifacts = _seed_artifacts(state, unit_price=984.0, quantity=1000)
    order = PurchaseOrder.from_artifacts(
        order_id=order_id_for(artifacts["decision"].id),
        decision_artifact=artifacts["decision"],
        authorization_artifact=artifacts["authorization"],
        requirement_artifact=artifacts["requirement"],
        quote_artifact=artifacts["quote"],
    )
    assert order.order_id == "PO-dec-1"
    assert order.decision_id == "dec-1"
    assert order.authorization_id == "auth-1"
    assert order.supplier_id == "MinerCorp_A"
    assert order.quantity == 1000
    assert order.total_amount == 984_000.0
    assert order.currency == "USD"
    assert order.status == PurchaseStatus.CREATED


def test_purchase_order_from_artifacts_without_quote_defaults_price() -> None:
    state = SwarmState()
    artifacts = _seed_artifacts(state)
    order = PurchaseOrder.from_artifacts(
        order_id=order_id_for(artifacts["decision"].id),
        decision_artifact=artifacts["decision"],
        authorization_artifact=artifacts["authorization"],
        requirement_artifact=artifacts["requirement"],
        quote_artifact=None,
    )
    assert order.total_amount == 0.0
    assert order.currency == "USD"


def test_order_to_dict_round_trips_fields() -> None:
    state = SwarmState()
    artifacts = _seed_artifacts(state, unit_price=1870.4, quantity=3000)
    order = PurchaseOrder.from_artifacts(
        order_id=order_id_for(artifacts["decision"].id),
        decision_artifact=artifacts["decision"],
        authorization_artifact=artifacts["authorization"],
        requirement_artifact=artifacts["requirement"],
        quote_artifact=artifacts["quote"],
    )
    data = order_to_dict(order)
    assert data["order_id"] == "PO-dec-1"
    assert data["supplier_id"] == "MinerCorp_A"
    assert data["total_amount"] == 1870.4 * 3000
    assert data["quantity"] == 3000
    assert data["status"] == "CREATED"
    assert data["items"][0]["material"] == "aluminum"


def test_purchase_order_artifact_name_and_kind() -> None:
    artifact = PurchaseOrderArtifact(data=order_to_dict(_build_minimal_order()))
    assert artifact.kind == "purchase_order"
    assert artifact.name == PURCHASE_ORDER_ARTIFACT_NAME


def _build_minimal_order() -> PurchaseOrder:
    return PurchaseOrder(
        order_id="PO-x",
        decision_id="dec-1",
        authorization_id="auth-1",
        supplier_id="MinerCorp_A",
        currency="USD",
        items=[],
        total_amount=0.0,
        status=PurchaseStatus.CREATED,
        created_at=datetime.now(UTC).isoformat(),
    )


class _RecordingConnector:
    """Determinism probe used by artifact tests."""

    def __init__(self) -> None:
        self.submitted: list[str] = []
        self.tracked: list[str] = []

    def submit_order(self, order: PurchaseOrder) -> PurchaseStatus:
        self.submitted.append(order.order_id)
        return PurchaseStatus.SUBMITTED

    def track_order(self, order: PurchaseOrder) -> PurchaseStatus:
        self.tracked.append(order.order_id)
        return PurchaseStatus.DELIVERED

    def order_lifecycle(self, order: PurchaseOrder) -> list[str]:
        return ["SUBMITTED", "CONFIRMED", "SHIPPED", "DELIVERED"]


def test_mock_connector_is_deterministic() -> None:
    connector = MockSupplierConnector()
    order = _build_minimal_order()
    assert connector.submit_order(order) == PurchaseStatus.SUBMITTED
    assert connector.track_order(order) == PurchaseStatus.DELIVERED
    lifecycle = connector.order_lifecycle(order)
    assert lifecycle == ["SUBMITTED", "CONFIRMED", "SHIPPED", "DELIVERED"]
