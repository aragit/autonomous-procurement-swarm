"""Supplier performance memory for the Phase 5 procurement swarm.

:mod:`swarm.memory.supplier` provides a deterministic, in-memory
:class:`SupplierMemoryStore` that records and retrieves cumulative
:class:`~swarm.domain.supplier.SupplierPerformance` records. It is the only
persistence surface for supplier intelligence — no external database or LLM is
introduced. The store is intentionally simple: the same sequence of recorded
outcomes always produces the same performance records, so swarm behavior stays
fully reproducible.

A module-level :data:`default_store` is provided for long-lived processes (such
as the API) that must share supplier history across independent swarm runs.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import UTC, datetime
from typing import Any

from swarm.domain.supplier import SupplierPerformance

# Determinism knobs. A supplier's reliability below this is "poor" history and a
# poor reliability above this is "strong" history; the adjustment is interpolated
# linearly in between and clamped to [-MAX_HISTORY_ADJUSTMENT, +MAX_HISTORY_ADJUSTMENT].
POOR_RELIABILITY = 0.40
STRONG_RELIABILITY = 0.90
MAX_HISTORY_ADJUSTMENT = 0.05


class SupplierMemoryStore:
    """Deterministic, in-memory supplier performance store (insertion-ordered)."""

    def __init__(self) -> None:
        self._performance: OrderedDict[str, SupplierPerformance] = OrderedDict()

    def get_supplier_performance(self, supplier_id: str) -> SupplierPerformance | None:
        """The cumulative performance record for ``supplier_id``, or None."""
        return self._performance.get(supplier_id)

    def save_performance(self, performance: SupplierPerformance) -> SupplierPerformance:
        """Store (or replace) a performance record for a supplier."""
        self._performance[performance.supplier_id] = performance
        return performance

    def update_from_outcome(
        self,
        outcome: dict[str, Any],
        *,
        reference_price: float | None = None,
    ) -> SupplierPerformance:
        """Fold one :class:`OutcomeArtifact`-shaped outcome into supplier memory.

        ``outcome`` must contain ``supplier_id``, ``delivered_on_time``,
        ``quality_score``, ``actual_price`` and ``carbon_score``.
        ``reference_price`` (the supplier's decided quote unit price) is used to
        derive price competitiveness; when absent, competitiveness defaults to the
        neutral ``1.0``.
        """
        supplier_id = str(outcome["supplier_id"])
        delivered_on_time = bool(outcome.get("delivered_on_time"))
        quality_score = float(outcome.get("quality_score") or 0.0)
        actual_price = float(outcome.get("actual_price") or 0.0)
        carbon_score = float(outcome.get("carbon_score") or 0.0)

        price_competitiveness = self._price_competitiveness(reference_price, actual_price)

        existing = self.get_supplier_performance(supplier_id)
        record = SupplierPerformance(supplier_id) if existing is None else existing
        updated = record.apply_outcome(
            delivered_on_time=delivered_on_time,
            quality_score=quality_score,
            price_competitiveness=price_competitiveness,
            carbon_score=carbon_score,
        )
        self.save_performance(updated)
        return updated

    @staticmethod
    def _price_competitiveness(reference_price: float | None, actual_price: float) -> float:
        """Ratio of quoted price to actual price, clamped to [0, 1].

        ``actual_price`` at or below the quoted price is perfect competitiveness
        (1.0); actual above quoted degrades toward 0.
        """
        if not reference_price or actual_price <= 0:
            return 1.0
        ratio = reference_price / actual_price
        return min(1.0, max(0.0, ratio))

    @staticmethod
    def history_adjustment(performance: SupplierPerformance | None) -> float:
        """Deterministic score multiplier for a supplier's reliability.

        ``+MAX_HISTORY_ADJUSTMENT`` for strong reliability (>= 0.9), ``-MAX_HISTORY_ADJUSTMENT``
        for poor reliability (<= 0.4), linear in between. No record → ``0.0`` (Phase 4
        behavior unchanged).
        """
        if performance is None or performance.total_orders == 0:
            return 0.0
        reliability = performance.delivery_reliability
        if reliability >= STRONG_RELIABILITY:
            return MAX_HISTORY_ADJUSTMENT
        if reliability <= POOR_RELIABILITY:
            return -MAX_HISTORY_ADJUSTMENT
        ratio = (reliability - POOR_RELIABILITY) / (STRONG_RELIABILITY - POOR_RELIABILITY)
        return (ratio * 2.0 - 1.0) * MAX_HISTORY_ADJUSTMENT

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Deterministic, serializable view of all records (for API/state dumps)."""
        return {sid: perf.to_summary() for sid, perf in self._performance.items()}


#: Long-lived, process-wide store shared across swarm runs (used by the API so
#: supplier memory persists across requests without a database.
default_store = SupplierMemoryStore()


def now_utc() -> datetime:
    """UTC now helper, isolated for testability."""
    return datetime.now(UTC)
