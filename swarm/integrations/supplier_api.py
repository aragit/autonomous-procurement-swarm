"""SupplierAPIConnector — a stateless external supplier-system adapter.

This is the first "real" :class:`~swarm.integrations.base.BaseConnector` and is
intentionally simpler than the ERP adapters: it talks to a (simulated) supplier
order API and records every call. Even when no live credentials are configured
it produces deterministic responses, so the execution trace stays reproducible
and replay-safe.

Design rules enforced here (inherited from :class:`BaseConnector`):
- pure function of inputs (deterministic);
- no hidden mutable state between calls (stateless);
- simulated response shape when unconfigured;
- returns normalized :class:`ExternalResponse` / :class:`ExternalStatus`.
"""

from __future__ import annotations

import hashlib

from swarm.domain.order import PurchaseOrder, PurchaseStatus
from swarm.integrations.base import BaseConnector, ExternalResponse, ExternalStatus

# Deterministic simulated supplier order statuses keyed by a hash of the order
# id, so the same order always progresses the same way.
_DETERMINED_STATUSES = [
    PurchaseStatus.SUBMITTED.value,
    PurchaseStatus.CONFIRMED.value,
    PurchaseStatus.SHIPPED.value,
    PurchaseStatus.DELIVERED.value,
]


def _status_for(order_id: str) -> PurchaseStatus:
    """Deterministically map an order id onto a terminal lifecycle stage."""
    digest = int(hashlib.sha256(order_id.encode()).hexdigest(), 16)
    return PurchaseStatus(_DETERMINED_STATUSES[digest % len(_DETERMINED_STATUSES)])


class SupplierAPIConnector(BaseConnector):
    """Stateless supplier-order connector with a simulated HTTP API surface.

    ``endpoint``/``api_key`` are accepted as configuration but, when unset, the
    connector answers deterministically so no live system is required to keep the
    swarm reproducible.
    """

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key

    @property
    def system(self) -> str:
        return "supplier_api"

    def _reference(self, order_id: str) -> str:
        return f"SUPPLIER-API-{order_id}"

    def submit_order(self, order: PurchaseOrder) -> ExternalResponse:
        """Simulate an HTTP ``POST /orders`` call returning a normalized response."""
        return ExternalResponse(
            success=True,
            order_id=order.order_id,
            status=_status_for(order.order_id).value,
            reference_id=self._reference(order.order_id),
            payload={"submitted": True, "supplier_id": order.supplier_id},
        )

    def get_order_status(self, order_id: str) -> ExternalStatus:
        """Simulate an HTTP ``GET /orders/{id}`` returning the realized status."""
        reference = self._reference(order_id)
        current = _status_for(order_id)
        idx = _DETERMINED_STATUSES.index(current.value)
        lifecycle = _DETERMINED_STATUSES[: idx + 1]
        return ExternalStatus(
            order_id=order_id,
            status=current.value,
            lifecycle=lifecycle,
            reference_id=reference,
        )

    def validate_supplier(self, supplier_id: str) -> bool:
        """Simulate a supplier-directory lookup; rejects a reserved test prefix."""
        return not supplier_id.lower().startswith("invalid_")
