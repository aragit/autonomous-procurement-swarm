"""Config-driven connector selection for the procurement swarm.

This is the runtime switch that lets the *same* swarm target a different external
system per environment — without code changes:

    DEV  → ConnectorConfig(provider="mock", mode="sandbox")      → MockConnector
    STAGE→ ConnectorConfig(provider="supplier_api", mode="sandbox") → SupplierAPIConnector
    PROD → ConnectorConfig(provider="coupa", mode="prod")        → CoupaConnector (live)

``provider`` picks the adapter; ``mode`` (``sandbox`` | ``prod``) selects the
operating environment. When ``mode == "prod"`` and ``credentials`` are present the
adapter performs a live API call — credentials are *configuration data* supplied
by the operator and never hardcoded, so the swarm trace and tests stay
deterministic and reproducible. When no credentials are present (or mode is
``sandbox``) every adapter simulates a deterministic response, keeping execution
replay-safe and auditable.

The factory reuses the lower-level ERP :class:`ConnectorConfig` internally — ERP
adapters remain unchanged and fully backward compatible.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

from swarm.integrations.base import BaseConnector
from swarm.integrations.erp.erp import (
    ConnectorConfig as ERPConnectorConfig,
)
from swarm.integrations.erp.erp import (
    CoupaConnector,
    OracleConnector,
    SAPConnector,
)
from swarm.integrations.mock import MockConnector
from swarm.integrations.supplier_api import SupplierAPIConnector

#: Supported connector providers.
PROVIDERS: tuple[str, ...] = ("mock", "supplier_api", "sap", "oracle", "coupa")

#: Supported operating modes.
MODES: tuple[str, ...] = ("sandbox", "prod")


@dataclass(frozen=True)
class ConnectorConfig:
    """Runtime configuration for selecting an external connector.

    ``provider`` selects the adapter (``mock``, ``supplier_api``, ``sap``,
    ``oracle`` or ``coupa``). ``mode`` selects the operating environment
    (``sandbox`` = deterministic simulation, ``prod`` = live API when
    ``credentials`` are supplied). ``credentials`` is an opaque, operator-supplied
    mapping — it is configuration *data*, never hardcoded by the swarm.
    """

    provider: str
    mode: str = "sandbox"
    endpoint: str | None = None
    credentials: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValueError(
                f"Unsupported connector provider {self.provider!r}; expected one of {PROVIDERS}"
            )
        if self.mode not in MODES:
            raise ValueError(
                f"Unsupported connector mode {self.mode!r}; expected one of {MODES}"
            )

    @property
    def environment(self) -> str:
        """ERP-style environment name derived from ``mode``."""
        return "production" if self.mode == "prod" else "sandbox"

    @property
    def live(self) -> bool:
        """Whether a live (non-simulated) API call is expected."""
        return self.mode == "prod" and bool(self.credentials)


def _erp_config(config: ConnectorConfig) -> ERPConnectorConfig:
    """Translate a unified config into the ERP-adapter config."""
    return ERPConnectorConfig(
        provider=config.provider,
        endpoint=config.endpoint or "",
        credentials=config.credentials,
        environment=config.environment,
    )


_ERP_BUILDERS: dict[str, Callable[[ERPConnectorConfig], BaseConnector]] = {
    "sap": SAPConnector,
    "oracle": OracleConnector,
    "coupa": CoupaConnector,
}


def build_connector(config: ConnectorConfig) -> BaseConnector:
    """Construct the :class:`BaseConnector` selected by ``config``.

    Same swarm, different environment, no code changes. Raises ``ValueError`` if
    ``config.provider`` is not one of :data:`PROVIDERS`.
    """
    provider = config.provider
    if provider == "mock":
        return MockConnector()
    if provider == "supplier_api":
        api_key = (config.credentials or {}).get("api_key")
        return SupplierAPIConnector(endpoint=config.endpoint, api_key=api_key)
    if provider in _ERP_BUILDERS:
        return _ERP_BUILDERS[provider](_erp_config(config))
    raise ValueError(
        f"Unsupported connector provider {provider!r}; expected one of {PROVIDERS}"
    )


def get_connector_config_from_env() -> ConnectorConfig:
    """Build a :class:`ConnectorConfig` from `PROCUREMENT_` environment variables.

    Environment variables:
        PROCUREMENT_CONNECTOR_PROVIDER (default ``mock``)
        PROCUREMENT_CONNECTOR_MODE     (default ``sandbox``)
    """
    provider = os.environ.get("PROCUREMENT_CONNECTOR_PROVIDER", "mock")
    mode = os.environ.get("PROCUREMENT_CONNECTOR_MODE", "sandbox")
    return ConnectorConfig(provider=provider, mode=mode)


def build_connector_from_env() -> BaseConnector:
    """Construct the connector selected by the runtime environment."""
    return build_connector(get_connector_config_from_env())


__all__ = [
    "ConnectorConfig",
    "PROVIDERS",
    "MODES",
    "build_connector",
    "build_connector_from_env",
    "get_connector_config_from_env",
]
