"""
Tests for market simulation.
"""

import pytest
from core.market_simulator import MarketSimulator, GeopoliticalRiskModel, RiskRegime


def test_market_simulator_creation():
    sim = MarketSimulator(seed=42)
    assert sim.day == 0


def test_market_step():
    sim = MarketSimulator(seed=42)
    states = sim.step()
    assert len(states) == 6  # All materials
    for material, state in states.items():
        assert state.spot_price > 0
        assert 0 <= state.geo_risk <= 1


def test_price_process():
    sim = MarketSimulator(seed=42)
    states1 = sim.step()
    states2 = sim.step()
    # Prices should change
    assert states1["steel"].spot_price != states2["steel"].spot_price


def test_risk_regime_transitions():
    model = GeopoliticalRiskModel(seed=42)
    regimes = [model.step() for _ in range(100)]
    # Should see multiple regimes
    assert len(set(regimes)) > 1


def test_risk_score_range():
    model = GeopoliticalRiskModel(seed=42)
    for _ in range(50):
        model.step()
        score = model.get_risk_score()
        assert 0 <= score <= 1
