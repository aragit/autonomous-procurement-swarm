"""Unit tests for the deterministic supplier performance memory store."""

from datetime import UTC, datetime

import pytest

from swarm.domain.supplier import SupplierPerformance
from swarm.memory import SupplierMemoryStore

FIXED_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def make_outcome(
    *,
    supplier_id: str = "MinerCorp_A",
    delivered_on_time: bool = True,
    quality_score: float = 0.92,
    actual_price: float = 1000.0,
    carbon_score: float = 1800.0,
) -> dict[str, object]:
    return {
        "supplier_id": supplier_id,
        "delivered_on_time": delivered_on_time,
        "quality_score": quality_score,
        "actual_price": actual_price,
        "carbon_score": carbon_score,
    }


def test_store_returns_none_for_unknown_supplier():
    store = SupplierMemoryStore()
    assert store.get_supplier_performance("Ghost") is None


def test_save_and_retrieve_performance():
    store = SupplierMemoryStore()
    perf = SupplierPerformance("MinerCorp_A", now=FIXED_NOW)
    perf.total_orders = 3
    perf.successful_orders = 2
    retrieved = store.save_performance(perf)
    assert store.get_supplier_performance("MinerCorp_A") is retrieved


def test_update_from_outcome_initializes_record():
    store = SupplierMemoryStore()
    outcome = make_outcome(
        delivered_on_time=True, quality_score=0.92, actual_price=1000.0, carbon_score=1800.0
    )
    perf = store.update_from_outcome(outcome)

    assert perf.supplier_id == "MinerCorp_A"
    assert perf.total_orders == 1
    assert perf.successful_orders == 1
    assert perf.average_delivery_score == 1.0
    assert perf.average_quality_score == 0.92
    assert perf.average_carbon_score == 1800.0
    assert perf.delivery_reliability == 1.0


def test_update_from_outcome_running_averages():
    store = SupplierMemoryStore()
    store.update_from_outcome(make_outcome(quality_score=0.8, carbon_score=1800.0))
    store.update_from_outcome(
        make_outcome(quality_score=0.92, carbon_score=2000.0, delivered_on_time=False)
    )

    perf = store.get_supplier_performance("MinerCorp_A")
    assert perf.total_orders == 2
    assert perf.successful_orders == 1
    assert perf.delivery_reliability == 0.5
    assert perf.average_quality_score == pytest.approx((0.8 + 0.92) / 2)
    assert perf.average_carbon_score == 1900.0


def test_update_is_deterministic_across_calls():
    store_a = SupplierMemoryStore()
    store_b = SupplierMemoryStore()
    for outcome in [
        make_outcome(quality_score=0.8, carbon_score=1800.0, delivered_on_time=True),
        make_outcome(quality_score=0.9, carbon_score=2000.0, delivered_on_time=False),
    ]:
        store_a.update_from_outcome(outcome)
        store_b.update_from_outcome(outcome)

    a = store_a.get_supplier_performance("MinerCorp_A")
    b = store_b.get_supplier_performance("MinerCorp_A")
    assert a.total_orders == b.total_orders == 2
    assert a.average_quality_score == pytest.approx(b.average_quality_score)
    assert a.delivery_reliability == b.delivery_reliability == 0.5
    # last_updated may differ in wall-clock but the metrics are identical
    assert a.delivery_reliability == 0.5


def test_history_adjustment_strong_reliability_boosts():
    store = SupplierMemoryStore()
    perf = store.update_from_outcome(make_outcome(delivered_on_time=True))
    assert SupplierMemoryStore.history_adjustment(perf) == 0.05


def test_history_adjustment_poor_reliability_penalizes():
    store = SupplierMemoryStore()
    store.update_from_outcome(make_outcome(delivered_on_time=True))
    store.update_from_outcome(make_outcome(delivered_on_time=True))
    store.update_from_outcome(make_outcome(delivered_on_time=True))
    store.update_from_outcome(make_outcome(delivered_on_time=True))
    store.update_from_outcome(make_outcome(delivered_on_time=False))
    perf = store.get_supplier_performance("MinerCorp_A")
    assert perf.delivery_reliability == 0.8
    # 0.8 is between 0.4 and 0.9: ratio=(0.8-0.4)/0.5=0.8, adj=(0.8*2-1)*0.05 = 0.03
    assert abs(SupplierMemoryStore.history_adjustment(perf) - 0.03) < 1e-9


def test_history_adjustment_poor_below_threshold_penalizes_max():
    store = SupplierMemoryStore()
    store.update_from_outcome(make_outcome(delivered_on_time=False))
    store.update_from_outcome(make_outcome(delivered_on_time=False))
    perf = store.get_supplier_performance("MinerCorp_A")
    assert perf.delivery_reliability == 0.0
    assert SupplierMemoryStore.history_adjustment(perf) == -0.05


def test_history_adjustment_none_is_zero():
    assert SupplierMemoryStore.history_adjustment(None) == 0.0


def test_history_adjustment_zero_orders_is_zero():
    perf = SupplierPerformance("NoOrders")
    assert SupplierMemoryStore.history_adjustment(perf) == 0.0


def test_price_competitiveness_clamped():
    store = SupplierMemoryStore()
    # actual == reference -> 1.0; actual below reference -> clamped to 1.0
    comp = store._price_competitiveness(reference_price=984.0, actual_price=984.0)
    assert comp == 1.0
    comp_better = store._price_competitiveness(reference_price=984.0, actual_price=900.0)
    assert comp_better == 1.0
    comp_worse = store._price_competitiveness(reference_price=984.0, actual_price=1200.0)
    assert comp_worse == round(984.0 / 1200.0, 10)
