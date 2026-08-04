"""Unit tests for the config-driven connector factory (Phase 8.1)."""

import pytest

from swarm.domain.order import PurchaseOrder, PurchaseStatus, order_id_for
from swarm.integrations import (
    ConnectorConfig,
    MockConnector,
    SupplierAPIConnector,
    build_connector,
    build_connector_from_env,
    get_connector_config_from_env,
)
from swarm.integrations.erp import CoupaConnector, OracleConnector, SAPConnector


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
    "provider,expected_type",
    [
        ("mock", MockConnector),
        ("supplier_api", SupplierAPIConnector),
        ("sap", SAPConnector),
        ("oracle", OracleConnector),
        ("coupa", CoupaConnector),
    ],
)
def test_build_connector_returns_supported_adapter(provider: str, expected_type: type) -> None:
    connector = build_connector(ConnectorConfig(provider=provider, mode="sandbox"))
    assert isinstance(connector, expected_type)


def test_mock_provider_returns_mock_connector() -> None:
    connector = build_connector(ConnectorConfig(provider="mock"))
    assert isinstance(connector, MockConnector)
    assert connector.validate_supplier("anyone") is True


def test_supplier_api_provider_is_stateless_and_rejects_invalid() -> None:
    connector = build_connector(ConnectorConfig(provider="supplier_api"))
    assert isinstance(connector, SupplierAPIConnector)
    assert connector.validate_supplier("MinerCorp_A") is True
    assert connector.validate_supplier("invalid_supplier") is False


@pytest.mark.parametrize(
    "provider,expected_type",
    [("sap", SAPConnector), ("oracle", OracleConnector), ("coupa", CoupaConnector)],
)
def test_erp_providers_map_to_correct_type(provider: str, expected_type: type) -> None:
    connector = build_connector(ConnectorConfig(provider=provider, mode="sandbox"))
    assert isinstance(connector, expected_type)
    resp = connector.submit_order(_order())
    assert resp.reference_id.startswith(f"{provider}-")
    resp = connector.submit_order(_order())
    assert resp.reference_id.startswith(f"{provider}-")


def test_erp_sandbox_is_not_live() -> None:
    config = ConnectorConfig(provider="coupa", mode="sandbox")
    assert config.environment == "sandbox"
    assert config.live is False
    connector = build_connector(config)
    assert connector.config.live is False


def test_erp_prod_with_credentials_is_live() -> None:
    config = ConnectorConfig(
        provider="sap",
        mode="prod",
        endpoint="https://sap.example",
        credentials={"client_id": "x", "client_secret": "y"},
    )
    assert config.environment == "production"
    assert config.live is True


def test_unknown_provider_config_raises_value_error() -> None:
    with pytest.raises(ValueError):
        ConnectorConfig(provider="unknown_erp")


def test_unknown_provider_via_build_raises_value_error() -> None:
    with pytest.raises(ValueError):
        build_connector(ConnectorConfig(provider="bogus"))


def test_invalid_mode_raises_value_error() -> None:
    with pytest.raises(ValueError):
        ConnectorConfig(provider="mock", mode="live")


def test_build_connector_from_env_defaults_to_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROCUREMENT_CONNECTOR_PROVIDER", raising=False)
    monkeypatch.delenv("PROCUREMENT_CONNECTOR_MODE", raising=False)
    connector = build_connector_from_env()
    assert isinstance(connector, MockConnector)
    config = get_connector_config_from_env()
    assert config.provider == "mock"
    assert config.mode == "sandbox"


def test_build_connector_from_env_reads_operator_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROCUREMENT_CONNECTOR_PROVIDER", "supplier_api")
    monkeypatch.setenv("PROCUREMENT_CONNECTOR_MODE", "sandbox")
    config = get_connector_config_from_env()
    assert config.provider == "supplier_api"
    assert config.mode == "sandbox"
    connector = build_connector_from_env()
    assert isinstance(connector, SupplierAPIConnector)


def test_staging_to_prod_env_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same swarm code path, different environments select different adapters."""
    monkeypatch.setenv("PROCUREMENT_CONNECTOR_PROVIDER", "coupa")
    monkeypatch.setenv("PROCUREMENT_CONNECTOR_MODE", "prod")
    connector = build_connector_from_env()
    assert isinstance(connector, CoupaConnector)
    assert connector.config.environment == "production"
