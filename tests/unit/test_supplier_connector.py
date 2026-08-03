"""Unit tests for the Phase 7 deterministic supplier connector."""

import pytest

from swarm.domain.order import (
    MockSupplierConnector,
    PurchaseOrder,
    PurchaseStatus,
    order_id_for,
)


def _order(supplier_id: str = "MinerCorp_A", quantity: int = 1000) -> PurchaseOrder:
    return PurchaseOrder(
        order_id=order_id_for("dec-1"),
        decision_id="dec-1",
        authorization_id="auth-1",
        supplier_id=supplier_id,
        currency="USD",
        items=[],
        total_amount=quantity * 984.0,
        status=PurchaseStatus.CREATED,
    )


def test_submit_order_returns_submitted_status() -> None:
    connector = MockSupplierConnector()
    assert connector.submit_order(_order()) == PurchaseStatus.SUBMITTED


def test_track_order_returns_delivered_status() -> None:
    connector = MockSupplierConnector()
    assert connector.track_order(_order()) == PurchaseStatus.DELIVERED


def test_order_lifecycle_is_deterministic_and_complete() -> None:
    connector = MockSupplierConnector()
    lifecycle = connector.order_lifecycle(_order())
    assert lifecycle == ["SUBMITTED", "CONFIRMED", "SHIPPED", "DELIVERED"]
    assert lifecycle[0] == "SUBMITTED"
    assert lifecycle[-1] == "DELIVERED"


@pytest.mark.parametrize("quantity", [1, 1000, 50_000])
def test_connector_is_pure_function_of_order(quantity: int) -> None:
    """The same order always yields the same status sequence (replay-safe)."""
    connector = MockSupplierConnector()
    order = _order(quantity=quantity)
    first = connector.order_lifecycle(order)
    second = connector.order_lifecycle(order)
    assert first == second
    assert first == connector.order_lifecycle(_order(quantity=quantity))


def test_custom_connector_can_implement_failure_path() -> None:
    """A non-default connector can model a cancelled order deterministically."""

    class FailingConnector:
        def submit_order(self, order: PurchaseOrder) -> PurchaseStatus:
            return PurchaseStatus.CANCELLED

        def track_order(self, order: PurchaseOrder) -> PurchaseStatus:
            return PurchaseStatus.CANCELLED

        def order_lifecycle(self, order: PurchaseOrder) -> list[str]:
            return ["SUBMITTED", "CANCELLED"]

    connector = FailingConnector()
    assert connector.submit_order(_order()) == PurchaseStatus.CANCELLED
