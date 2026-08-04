"""MockConnector — deterministic in-memory adapter for the procurement swarm.

Replaces the legacy ``MockSupplierConnector``: the same order id and supplier
always yield the same submission reference and status sequence, so the execution
trace stays deterministic and replay-safe even without a live system. When a
real connector is wired in, it simply implements the same
:class:`~swarm.integrations.base.BaseConnector` interface.
"""

from __future__ import annotations

import hashlib

from swarm.domain.order import PurchaseOrder, PurchaseStatus
from swarm.integrations.base import BaseConnector, ExternalResponse, ExternalStatus

MOCK_LIFECYCLE = [
    PurchaseStatus.SUBMITTED.value,
    PurchaseStatus.CONFIRMED.value,
    PurchaseStatus.SHIPPED.value,
    PurchaseStatus.DELIVERED.value,
]


class MockConnector(BaseConnector):
    """Deterministic, in-memory :class:`BaseConnector` implementation.

    Healthy suppliers progress to delivery; status is a pure function of the
    order so the same input always produces the same lifecycle.
    """

    system = "mock"

    def submit_order(self, order: PurchaseOrder) -> ExternalResponse:
        reference = f"MOCK-{order.order_id}"
        return ExternalResponse(
            success=True,
            order_id=order.order_id,
            status=PurchaseStatus.SUBMITTED.value,
            reference_id=reference,
            payload={"currency": order.currency, "total_amount": order.total_amount},
        )

    def get_order_status(self, order_id: str) -> ExternalStatus:
        return ExternalStatus(
            order_id=order_id,
            status=PurchaseStatus.DELIVERED.value,
            lifecycle=list(MOCK_LIFECYCLE),
            reference_id=f"MOCK-{order_id}",
        )

    def validate_supplier(self, supplier_id: str) -> bool:
        _ = hashlib.sha256(supplier_id.encode()).hexdigest()
        return True

    def order_lifecycle(self, order: PurchaseOrder) -> list[str]:
        """Backwards-compatible lifecycle accessor for legacy callers."""
        return list(MOCK_LIFECYCLE)


__all__ = ["MockConnector"]
