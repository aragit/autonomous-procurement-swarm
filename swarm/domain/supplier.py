"""Deterministic supplier intelligence models for the Phase 5 procurement swarm.

A :class:`SupplierPerformance` record is a cumulative, append-only snapshot of
how a supplier has performed across completed procurements. Every metric is a
running average updated in a single deterministic pass from an
:class:`OutcomeArtifact`; there is no learning, no LLM and no external state —
the same sequence of outcomes always yields the same record.
"""

from __future__ import annotations

from datetime import UTC, datetime


class SupplierPerformance:
    """Cumulative, deterministic performance record for a single supplier.

    ``average_*`` fields are running means over every recorded outcome so far;
    ``average_delivery_score`` and the delivery-reliability convention both derive
    from the same on-time signal, keeping the record internally consistent.
    """

    def __init__(self, supplier_id: str, *, now: datetime | None = None) -> None:
        self.supplier_id = supplier_id
        self.total_orders = 0
        self.successful_orders = 0
        self.average_delivery_score = 0.0
        self.average_quality_score = 0.0
        self.average_price_competitiveness = 1.0
        self.average_carbon_score = 0.0
        self.last_updated = now or datetime.now(UTC)

    def apply_outcome(
        self,
        *,
        delivered_on_time: bool,
        quality_score: float,
        price_competitiveness: float,
        carbon_score: float,
        now: datetime | None = None,
    ) -> SupplierPerformance:
        """Return a new record with ``outcome`` folded in (immutable update)."""
        next_count = self.total_orders + 1
        new = SupplierPerformance(self.supplier_id, now=now or datetime.now(UTC))
        new.total_orders = next_count
        new.successful_orders = self.successful_orders + (1 if delivered_on_time else 0)
        new.average_delivery_score = self._running_mean(
            self.average_delivery_score, self.total_orders, 1.0 if delivered_on_time else 0.0
        )
        new.average_quality_score = self._running_mean(
            self.average_quality_score, self.total_orders, quality_score
        )
        new.average_price_competitiveness = self._running_mean(
            self.average_price_competitiveness, self.total_orders, price_competitiveness
        )
        new.average_carbon_score = self._running_mean(
            self.average_carbon_score, self.total_orders, carbon_score
        )
        return new

    @property
    def delivery_reliability(self) -> float:
        """Fraction of orders delivered on time (0.0 when no orders recorded)."""
        if self.total_orders == 0:
            return 0.0
        return self.successful_orders / self.total_orders

    def to_summary(self) -> dict[str, float | int | str]:
        """Serializable summary used by the performance query API."""
        return {
            "supplier_id": self.supplier_id,
            "total_orders": self.total_orders,
            "successful_orders": self.successful_orders,
            "delivery_score": round(self.average_delivery_score, 4),
            "quality_score": round(self.average_quality_score, 4),
            "price_competitiveness": round(self.average_price_competitiveness, 4),
            "carbon_score": round(self.average_carbon_score, 4),
        }

    @staticmethod
    def _running_mean(current: float, count: int, value: float) -> float:
        """Deterministic running mean of ``count`` samples plus ``value``."""
        if count <= 0:
            return float(value)
        return (current * count + float(value)) / (count + 1)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"SupplierPerformance(supplier_id={self.supplier_id!r}, "
            f"total_orders={self.total_orders}, "
            f"delivery_reliability={self.delivery_reliability:.4f})"
        )
