"""Connector integration layer for the autonomous procurement swarm.

Public surface: :class:`BaseConnector` (the deterministic port every
ERP/supplier adapter satisfies), :class:`MockConnector` (default in-memory
adapter), the enterprise adapters (:class:`SupplierAPIConnector`,
``SAPConnector``, ``OracleConnector``, ``CoupaConnector``), and the
:func:`build_connector` factory driven by a :class:`ConnectorConfig` for
runtime-selectable, environment-driven connector wiring.
"""

from swarm.integrations.base import BaseConnector, ExternalResponse, ExternalStatus
from swarm.integrations.erp import CoupaConnector, OracleConnector, SAPConnector
from swarm.integrations.factory import (
    MODES,
    PROVIDERS,
    ConnectorConfig,
    build_connector,
    build_connector_from_env,
    get_connector_config_from_env,
)
from swarm.integrations.mock import MockConnector
from swarm.integrations.supplier_api import SupplierAPIConnector

__all__ = [
    "BaseConnector",
    "ExternalResponse",
    "ExternalStatus",
    "ConnectorConfig",
    "PROVIDERS",
    "MODES",
    "SAPConnector",
    "OracleConnector",
    "CoupaConnector",
    "MockConnector",
    "SupplierAPIConnector",
    "build_connector",
    "build_connector_from_env",
    "get_connector_config_from_env",
]
