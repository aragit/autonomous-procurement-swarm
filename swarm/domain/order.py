"""Deterministic procurement order & supplier-connection domain (Phase 7).

This layer is the *Action* tier of the swarm: it turns an authorized decision
into a concrete purchase order and tracks its execution lifecycle — with no LLM
and no autonomous decision making. Risk, governance and approval still own
*safety & authorization*; this layer only ever runs once a decision is
``authorized`` and only ever records what a deterministic supplier connector
reports back.

The :class:`SupplierConnector` is a small protocol so the swarm stays real:
:meth:`submit_order` hands a :class:`PurchaseOrder` to a supplier system and
:meth:`track_order` reports the realized status. :class:`MockSupplierConnector`
is a deterministic, in-memory implementation suitable for the demo and tests;
the enterprise ``SAPConnector`` / ``OracleConnector`` / ``CoupaConnector``
adapters plug in later without changing the agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

from swarm.core.artifact import Artifact


class PurchaseStatus(StrEnum):
    """Deterministic lifecycle stages of a purchase order."""

    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    CONFIRMED = "CONFIRMED"
    SHIPPED = "SHIPPED"
    DELIVERED = "DELIVERED"
    CANCELLED = "CANCELLED"


class SupplierConnector(Protocol):
    """Minimal contract every ERP/supplier adapter must satisfy."""

    def submit_order(self, order: PurchaseOrder) -> PurchaseStatus: ...

    def track_order(self, order: PurchaseOrder) -> PurchaseStatus: ...

    def order_lifecycle(self, order: PurchaseOrder) -> list[str]: ...


class MockSupplierConnector:
    """Deterministic, in-memory supplier connector (Phase 7 default).

    Healthy suppliers progress to delivery; status is a pure function of the
    order so the same input always produces the same lifecycle — making the
    execution trace replay-safe and auditable.
    """

    def submit_order(self, order: PurchaseOrder) -> PurchaseStatus:
        return PurchaseStatus.SUBMITTED

    def track_order(self, order: PurchaseOrder) -> PurchaseStatus:
        return PurchaseStatus.DELIVERED

    def order_lifecycle(self, order: PurchaseOrder) -> list[str]:
        return [
            PurchaseStatus.SUBMITTED.value,
            PurchaseStatus.CONFIRMED.value,
            PurchaseStatus.SHIPPED.value,
            PurchaseStatus.DELIVERED.value,
        ]


#: Shared default connector (mirrors the module-level ``default_store`` pattern).
default_connector: SupplierConnector = MockSupplierConnector()


@dataclass(frozen=True)
class OrderLine:
    """One line of a purchase order."""

    material: str
    quantity: int
    unit_price: float


@dataclass(frozen=True)
class PurchaseOrder:
    """A deterministic purchase order anchored to an authorized decision.

    Built by :class:`swarm.domain.agents.purchase_order_agent.PurchaseOrderAgent`
    from the originating decision, authorization, requirement and quote
    artifacts — so the order is always reproducible from the recorded swarm trace.
    """

    order_id: str
    decision_id: str
    authorization_id: str
    supplier_id: str
    currency: str
    items: list[OrderLine] = field(default_factory=list)
    total_amount: float = 0.0
    status: PurchaseStatus = PurchaseStatus.CREATED
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    submitted_at: str | None = None

    @property
    def quantity(self) -> int:
        return sum(line.quantity for line in self.items)

    @classmethod
    def from_artifacts(
        cls,
        *,
        order_id: str,
        decision_artifact: Artifact,
        authorization_artifact: Artifact,
        requirement_artifact: Artifact,
        quote_artifact: Artifact | None = None,
    ) -> PurchaseOrder:
        """Build a :class:`PurchaseOrder` from the artifacts of a completed flow."""
        decision_data = decision_artifact.data
        constraints = requirement_artifact.data.get("constraints", {})
        material = str(constraints.get("material", ""))
        quantity = int(constraints.get("quantity", 0))
        quote_data: dict[str, Any] = (
            quote_artifact.data if quote_artifact is not None else {}
        )
        unit_price = float(quote_data.get("price", 0.0))
        currency = str(quote_data.get("currency", "USD"))
        total_amount = unit_price * quantity
        return cls(
            order_id=order_id,
            decision_id=str(decision_artifact.id),
            authorization_id=str(authorization_artifact.id),
            supplier_id=str(decision_data.get("selected_supplier", "")),
            currency=currency,
            items=[OrderLine(material=material, quantity=quantity, unit_price=unit_price)],
            total_amount=total_amount,
            status=PurchaseStatus.CREATED,
        )


def order_id_for(decision_id: str) -> str:
    """Stable, deterministic order id derived from the originating decision."""
    return f"PO-{decision_id}"


def order_to_dict(order: PurchaseOrder) -> dict[str, Any]:
    """Serialize a :class:`PurchaseOrder` into a JSON-safe dict for artifacts/API."""
    return {
        "order_id": order.order_id,
        "decision_id": order.decision_id,
        "authorization_id": order.authorization_id,
        "supplier_id": order.supplier_id,
        "currency": order.currency,
        "items": [
            {
                "material": line.material,
                "quantity": line.quantity,
                "unit_price": line.unit_price,
            }
            for line in order.items
        ],
        "total_amount": order.total_amount,
        "quantity": order.quantity,
        "status": order.status.value,
        "created_at": order.created_at,
        "submitted_at": order.submitted_at,
    }
