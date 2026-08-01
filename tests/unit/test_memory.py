"""Unit tests for the memory & heuristics layer (Sprint 5)."""

import pytest
import pytest_asyncio

from core.memory.heuristics import HeuristicReservationEstimator
from core.memory.semantic import PgVectorMemoryStore

TEST_DB_URL = "postgresql+asyncpg://procurement:procurement@localhost:5433/procurement"


class TestHeuristicReservationEstimator:
    """Pure in-memory heuristics — no DB needed."""

    def test_estimate_reservation_price_no_history(self):
        est = HeuristicReservationEstimator()
        # No history: spot minus default 8% margin
        price = est.estimate_reservation_price("MinerCorp_A", 120.0)
        assert price == pytest.approx(120.0 * 0.92)
        assert price > 0

    def test_record_result_builds_profile(self):
        est = HeuristicReservationEstimator()
        est.record_auction_result(
            supplier_id="MinerCorp_A",
            original_bid_price=100.0,
            final_price=85.0,
            spot_price=120.0,
            turns_taken=5,
            won=True,
        )
        profile = est.get_profile("MinerCorp_A")
        assert profile is not None
        assert profile.auctions_participated == 1
        assert profile.auctions_won == 1
        assert profile.avg_concession_slope == pytest.approx(3.0)  # (100-85)/5
        assert profile.concession_speed == "medium"  # 3.0 > 120*0.01=1.2, < 120*0.05=6.0

    def test_wins_and_losses_tracked(self):
        est = HeuristicReservationEstimator()
        for won in [True, False, True]:
            est.record_auction_result(
                supplier_id="X",
                original_bid_price=100.0,
                final_price=90.0 if won else None,
                spot_price=120.0,
                turns_taken=3,
                won=won,
            )
        profile = est.get_profile("X")
        assert profile.auctions_participated == 3
        assert profile.auctions_won == 2
        assert profile.avg_final_price_vs_spot > 0

    def test_estimate_uses_history_when_available(self):
        est = HeuristicReservationEstimator()
        # Win with final_price == spot (ratio 1.0), so estimate == spot
        est.record_auction_result(
            supplier_id="Y",
            original_bid_price=200.0,
            final_price=120.0,
            spot_price=120.0,
            turns_taken=4,
            won=True,
        )
        # concession slope = (200-120)/4 = 20 > 120*0.05=6 => "fast"
        est.record_auction_result(
            supplier_id="Y",
            original_bid_price=200.0,
            final_price=110.0,
            spot_price=120.0,
            turns_taken=4,
            won=True,
        )
        profile = est.get_profile("Y")
        assert profile.concession_speed == "fast"
        # fast conceders: estimated *= 0.95; floor at 60% of spot
        estimated = est.estimate_reservation_price("Y", 120.0)
        assert estimated <= 120.0 * 1.0 * 0.95 + 1e-6
        assert estimated >= 120.0 * 0.6

    def test_floor_never_below_60pct(self):
        est = HeuristicReservationEstimator()
        est.record_auction_result(
            supplier_id="Z",
            original_bid_price=1000.0,
            final_price=50.0,
            spot_price=100.0,
            turns_taken=1,
            won=True,
        )
        est.record_auction_result(
            supplier_id="Z",
            original_bid_price=1000.0,
            final_price=45.0,
            spot_price=100.0,
            turns_taken=1,
            won=True,
        )
        estimated = est.estimate_reservation_price("Z", 100.0)
        assert estimated >= 100.0 * 0.6

    def test_all_profiles(self):
        est = HeuristicReservationEstimator()
        assert est.all_profiles() == []
        est.record_auction_result("A", 10.0, 9.0, 100.0, 2, True)
        est.record_auction_result("B", 10.0, 9.0, 100.0, 2, True)
        assert len(est.all_profiles()) == 2


@pytest_asyncio.fixture
async def vector_store():
    """Provide a pgvector store against the local PostgreSQL."""
    store = PgVectorMemoryStore(TEST_DB_URL)
    await store.init_schema()
    yield store
    await store.engine.dispose()


@pytest.mark.asyncio
async def test_vector_store_index_and_query(vector_store):
    """Round-trip: index two suppliers, then query for similarity."""
    await vector_store.index_supplier(
        "MinerCorp_A",
        {
            "supplier_id": "MinerCorp_A",
            "avg_margin_at_win": 0.20,
            "reliability_score": 0.85,
        },
    )
    await vector_store.index_supplier(
        "DistribCorp_B",
        {
            "supplier_id": "DistribCorp_B",
            "avg_margin_at_win": 0.12,
            "reliability_score": 0.90,
        },
    )

    results = await vector_store.query_similar_suppliers(
        {"supplier_id": "MinerCorp_A", "avg_margin_at_win": 0.20, "reliability_score": 0.85},
        n_results=2,
    )
    assert len(results) >= 1
    assert results[0]["supplier_id"] == "MinerCorp_A"
    assert results[0]["distance"] == pytest.approx(0.0)
    assert "metadata" in results[0]


@pytest.mark.asyncio
async def test_vector_store_get_profile(vector_store):
    await vector_store.index_supplier("TraderCorp_D", {"name": "TraderCorp_D", "reliability": 0.7})
    profile = await vector_store.get_profile("TraderCorp_D")
    assert profile == {"name": "TraderCorp_D", "reliability": 0.7}
    assert await vector_store.get_profile("Nope") is None
