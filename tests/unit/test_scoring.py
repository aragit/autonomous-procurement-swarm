"""Unit tests for multi-criteria scoring engine."""

import pytest

from core.evaluator.scoring import EvaluationWeights, MultiCriteriaEvaluator
from core.protocol.schema import BidPayload


@pytest.fixture
def evaluator():
    return MultiCriteriaEvaluator(
        weights=EvaluationWeights(),
        esg_baselines={
            "steel": 1800.0,
            "aluminum": 12000.0,
        },
    )


def test_price_score_at_spot(evaluator):
    score = evaluator._score_price(500.0, 500.0)
    assert score == 0.5  # 1.0 - (500/500)/2 = 0.5


def test_price_score_below_spot(evaluator):
    score = evaluator._score_price(250.0, 500.0)
    assert score == 0.75  # 1.0 - (250/500)/2 = 0.75


def test_price_score_at_double_spot(evaluator):
    score = evaluator._score_price(1000.0, 500.0)
    assert score == 0.0


def test_lead_time_at_target(evaluator):
    score = evaluator._score_lead_time(30, 30)
    assert score == 1.0


def test_lead_time_double_target(evaluator):
    score = evaluator._score_lead_time(60, 30)
    assert score == 0.0


def test_esg_perfect(evaluator):
    score = evaluator._score_esg(0.0, "steel")
    assert score == 1.0


def test_esg_at_baseline(evaluator):
    score = evaluator._score_esg(1800.0, "steel")
    assert score == 0.0


def test_esg_above_baseline(evaluator):
    score = evaluator._score_esg(3600.0, "steel")
    assert score == 0.0


def test_esg_normalization_with_quantity(evaluator):
    # 1800 kg total carbon for 1000 units = 1.8 kg per unit, well below 1800 baseline
    score = evaluator._score_esg(1800.0, "steel", quantity=1000)
    assert score == pytest.approx(0.999, abs=0.001)

    # 1,800,000 kg total for 1000 units = 1800 per unit, exactly at baseline
    score = evaluator._score_esg(1_800_000.0, "steel", quantity=1000)
    assert score == 0.0


def test_reliability_score(evaluator):
    score = evaluator._score_reliability(0.85)
    assert score == 0.85


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        EvaluationWeights(
            price_weight=0.5,
            lead_time_weight=0.5,
            esg_weight=0.2,
            reliability_weight=0.2,
        )


def test_full_composite_score(evaluator):
    bid = BidPayload(
        session_id="test",
        supplier_id="TestSupplier",
        unit_price=400.0,  # Spot=500, ratio=0.8, price_score=0.6
        lead_time_days=30,  # Target=30, lt_score=1.0
        carbon_footprint_kg=900.0,  # Baseline=1800, esg_score=0.5
        reliability_score=0.8,
        bid_bond_amount=100.0,
        delivery_date="2026-08-15",
        justification="Test",
    )

    score = evaluator.score_bid(bid, market_spot_price=500.0, target_lead_time=30, material="steel")
    # Expected: 0.4*0.6 + 0.25*1.0 + 0.2*0.5 + 0.15*0.8 = 0.24 + 0.25 + 0.10 + 0.12 = 0.71
    assert score == pytest.approx(0.71, abs=0.01)


def test_rank_bids(evaluator):
    bids = [
        BidPayload(
            session_id="t",
            supplier_id="CheapSlow",
            unit_price=300.0,
            lead_time_days=60,
            carbon_footprint_kg=1800.0,
            reliability_score=0.5,
            bid_bond_amount=100.0,
            delivery_date="2026-08-15",
        ),
        BidPayload(
            session_id="t",
            supplier_id="ExpensiveFast",
            unit_price=500.0,
            lead_time_days=15,
            carbon_footprint_kg=0.0,
            reliability_score=1.0,
            bid_bond_amount=100.0,
            delivery_date="2026-08-15",
        ),
    ]
    ranked = evaluator.rank_bids(
        bids, market_spot_price=500.0, target_lead_time=30, material="steel"
    )
    # ExpensiveFast should win due to perfect ESG + reliability + lead time
    assert ranked[0][1].supplier_id == "ExpensiveFast"
    assert ranked[1][1].supplier_id == "CheapSlow"
