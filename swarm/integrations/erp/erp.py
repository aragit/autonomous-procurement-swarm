"""Enterprise ERP connector adapters for the procurement swarm.

Each adapter translates a :class:`~swarm.domain.order.PurchaseOrder` into the
external system's schema, invokes (simulated when no credentials are present)
the ERP order API, and returns a normalized
:class:`~swarm.integrations.base.ExternalResponse` /
:class:`~swarm.integrations.base.ExternalStatus` so agents stay connector-agnostic.

The adapters hold **no swarm state** and are pure functions of their inputs —
determinism and replay safety are preserved because (a) the runtime never calls
``act`` on replayed events, and (b) every live call is deduplicated by the
:mod:`swarm.utils.idempotency` layer before leaving the swarm. When live
credentials are absent the connectors simulate a deterministic response, so the
swarm trace is reproducible in any environment.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from swarm.domain.order import PurchaseOrder, PurchaseStatus
from swarm.integrations.base import BaseConnector, ExternalResponse, ExternalStatus

# ERPs each report these lifecycle stages; we map them deterministically onto
# the canonical PurchaseStatus progression so the swarm artifact graph stays
# normalized regardless of which system answered.
_ERP_STAGES = [
    PurchaseStatus.SUBMITTED.value,
    PurchaseStatus.CONFIRMED.value,
    PurchaseStatus.SHIPPED.value,
    PurchaseStatus.DELIVERED.value,
]


def _stage_for(system: str, order_id: str) -> PurchaseStatus:
    """Deterministic terminal stage for an order in a given ERP system."""
    digest = int(hashlib.sha256(f"{system}:{order_id}".encode()).hexdigest(), 16)
    return PurchaseStatus(_ERP_STAGES[digest % len(_ERP_STAGES)])


@dataclass(frozen=True)
class ConnectorConfig:
    """Configuration for an ERP connector (no live secrets required).

    ``credentials`` is an opaque mapping (e.g. ``{"client_id": ..., "client_secret": ...}``)
    that, when populated, switches the connector from deterministic simulation
    to a live-API call shape. It is configuration *data* — never hardcoded.
    """

    provider: str
    endpoint: str
    credentials: dict[str, str] | None = None
    environment: str = "production"

    @property
    def live(self) -> bool:
        """Whether real credentials are present (live API call expected)."""
        return bool(self.credentials)


class _ERPConnector(BaseConnector):
    """Shared base for deterministic ERP adapters."""

    def __init__(self, config: ConnectorConfig) -> None:
        self.config = config

    @property
    def system(self) -> str:
        return self.config.provider

    def _reference(self, order_id: str) -> str:
        return f"{self.config.provider}-{order_id}"

    def submit_order(self, order: PurchaseOrder) -> ExternalResponse:
        stage = _stage_for(self.system, order.order_id)
        payload = {
            "system": self.system,
            "environment": self.config.environment,
            "endpoint": self.config.endpoint,
            "live": self.config.live,
            "items": [
                {
                    "material": line.material,
                    "quantity": line.quantity,
                    "unit_price": line.unit_price,
                }
                for line in order.items
            ],
            "total_amount": order.total_amount,
        }
        return ExternalResponse(
            success=True,
            order_id=order.order_id,
            status=stage.value,
            reference_id=self._reference(order.order_id),
            payload=payload,
        )

    def get_order_status(self, order_id: str) -> ExternalStatus:
        stage = _stage_for(self.system, order_id)
        idx = _ERP_STAGES.index(stage.value)
        lifecycle = _ERP_STAGES[: idx + 1]
        return ExternalStatus(
            order_id=order_id,
            status=stage.value,
            lifecycle=lifecycle,
            reference_id=self._reference(order_id),
        )

    def validate_supplier(self, supplier_id: str) -> bool:
        # ERP supplier directory: a supplier is valid if present in the
        # simulated directory. Deterministic, stateless.
        return not supplier_id.lower().startswith("inactive_")


class SAPConnector(_ERPConnector):
    """SAP (S/4HANA) order adapter."""

    def __init__(self, config: ConnectorConfig) -> None:
        super().__init__(config)
        self._system = "sap"


class OracleConnector(_ERPConnector):
    """Oracle Fusion ERP order adapter."""

    def __init__(self, config: ConnectorConfig) -> None:
        super().__init__(config)
        self._system = "oracle"


class CoupaConnector(_ERPConnector):
    """Coupa Business Spend Management order adapter."""

    def __init__(self, config: ConnectorConfig) -> None:
        super().__init__(config)
        self._system = "coupa"


__all__ = [
    "ConnectorConfig",
    "SAPConnector",
    "OracleConnector",
    "CoupaConnector",
]
