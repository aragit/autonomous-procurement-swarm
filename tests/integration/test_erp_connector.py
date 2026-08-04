"""Tests for the Phase 8 deterministic connectors and idempotency contract.

Covers the BaseConnector implementations (Mock, SupplierAPI, SAP, Oracle,
Coupa): determinism, pure-function-of-input behavior, validate_supplier, and
that every connector satisfies the shared interface contract.
"""

import pytest

from swarm.domain.order import PurchaseOrder, PurchaseStatus, order_id_for
from swarm.integrations.base import BaseConnector
from swarm.integrations.erp import ConnectorConfig, CoupaConnector, OracleConnector, SAPConnector
from swarm.integrations.mock import MockConnector
from swarm.integrations.supplier_api import SupplierAPIConnector


def _order(supplier_id: str = "MinerCorp_A") -> PurchaseOrder:
    return PurchaseOrder(
        order_id=order_id_for("dec-1"),
        decision_id="dec-1",
        authorization_id="auth-1",
        supplier_id=supplier_id,
        currency="USD",
        items=[],
        total_amount=98_400.0,
        status=PurchaseStatus.CREATED,
    )


@pytest.mark.parametrize(
    "connector",
    [
        MockConnector(),
        SupplierAPIConnector(),
        SAPConnector(ConnectorConfig(provider="sap", endpoint="https://sap.local")),
        OracleConnector(ConnectorConfig(provider="oracle", endpoint="https://oracle.local")),
        CoupaConnector(ConnectorConfig(provider="coupa", endpoint="https://coupa.local")),
    ],
)
def test_connector_satisfies_base_contract(connector: BaseConnector) -> None:
    response = connector.submit_order(_order())
    status = connector.get_order_status(response.order_id)
    assert response.success is True
    assert status.status == status.lifecycle[-1]
    assert connector.validate_supplier("MinerCorp_A") is True


def test_mock_connector_lifecycle_is_deterministic() -> None:
    connector = MockConnector()
    order = _order()
    first = connector.get_order_status(order.order_id)
    second = connector.get_order_status(order.order_id)
    assert first == second
    assert first.lifecycle == ["SUBMITTED", "CONFIRMED", "SHIPPED", "DELIVERED"]
    assert first.status == "DELIVERED"


def test_supplier_api_connector_simulates_deterministic_progression() -> None:
    connector = SupplierAPIConnector()
    order = _order()
    response = connector.submit_order(order)
    status = connector.get_order_status(order.order_id)
    assert response.reference_id == f"SUPPLIER-API-{order.order_id}"
    assert status.lifecycle[0] == "SUBMITTED"
    assert status.lifecycle[-1] == status.status
    # deterministic: same order id -> same terminal status
    assert connector.get_order_status(order.order_id) == status


def test_supplier_api_rejects_invalid_prefix() -> None:
    connector = SupplierAPIConnector()
    assert connector.validate_supplier("invalid_supplier") is False
    assert connector.validate_supplier("MinerCorp_A") is True


def test_erp_connectors_are_deterministic_and_distinct() -> None:
    sap = SAPConnector(ConnectorConfig(provider="sap", endpoint="https://sap.local"))
    oracle = OracleConnector(ConnectorConfig(provider="oracle", endpoint="https://oracle.local"))
    order = _order()
    sap_status = sap.get_order_status(order.order_id)
    oracle_status = oracle.get_order_status(order.order_id)
    # Both deterministic on the same input.
    assert sap.get_order_status(order.order_id) == sap_status
    assert oracle.get_order_status(order.order_id) == oracle_status
    # The reference id encodes the system for audit lineage.
    assert sap.submit_order(order).reference_id.startswith("sap-")
    assert oracle.submit_order(order).reference_id.startswith("oracle-")


def test_erp_connector_config_live_flag() -> None:
    sim = SAPConnector(ConnectorConfig(provider="sap", endpoint="https://sap.local"))
    live = SAPConnector(
        ConnectorConfig(
            provider="sap",
            endpoint="https://sap.local",
            credentials={"client_id": "x", "client_secret": "y"},
        )
    )
    assert sim.config.live is False
    assert live.config.live is True
