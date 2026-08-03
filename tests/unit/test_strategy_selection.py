"""Unit tests for Phase 4 strategy model and deterministic selection rules."""

import pytest
from pydantic import ValidationError

from swarm.domain.strategy import (
    BALANCED_STRATEGY,
    DEFAULT_STRATEGIES,
    Strategy,
    select_strategy,
)


def test_default_strategies_expose_expected_weights():
    assert set(DEFAULT_STRATEGIES) == {"cost_optimized", "balanced", "low_carbon"}
    assert DEFAULT_STRATEGIES["cost_optimized"].as_weights() == {
        "price_weight": 0.65,
        "score_weight": 0.25,
        "carbon_weight": 0.10,
    }
    assert DEFAULT_STRATEGIES["balanced"].as_weights() == {
        "price_weight": 0.40,
        "score_weight": 0.40,
        "carbon_weight": 0.20,
    }
    assert DEFAULT_STRATEGIES["low_carbon"].as_weights() == {
        "price_weight": 0.20,
        "score_weight": 0.25,
        "carbon_weight": 0.55,
    }


def test_strategy_weights_always_sum_to_one():
    for strategy in DEFAULT_STRATEGIES.values():
        total = strategy.price_weight + strategy.score_weight + strategy.carbon_weight
        assert abs(total - 1.0) < 1e-6


def test_strategy_rejects_weights_that_do_not_sum_to_one():
    with pytest.raises(ValidationError):
        Strategy(
            name="broken",
            price_weight=0.5,
            score_weight=0.5,
            carbon_weight=0.5,
        )


def test_balanced_strategy_is_the_legacy_fallback():
    assert BALANCED_STRATEGY is DEFAULT_STRATEGIES["balanced"]
    assert BALANCED_STRATEGY.as_weights() == {
        "price_weight": 0.40,
        "score_weight": 0.40,
        "carbon_weight": 0.20,
    }


def test_select_strategy_prefers_low_carbon_on_carbon_constraint():
    strategy = select_strategy(
        {
            "material": "aluminum",
            "quantity": 1000,
            "budget": 2_000_000.0,
            "max_unit_price": 2640.0,
            "max_carbon_per_unit": 800.0,
        }
    )
    assert strategy.name == "low_carbon"


def test_select_strategy_prefers_cost_optimized_on_tight_budget():
    strategy = select_strategy(
        {
            "material": "aluminum",
            "quantity": 1000,
            "budget": 500_000.0,
            "max_unit_price": 2640.0,
            "max_carbon_per_unit": None,
        }
    )
    assert strategy.name == "cost_optimized"


def test_select_strategy_picks_balanced_when_relaxed():
    strategy = select_strategy(
        {
            "material": "aluminum",
            "quantity": 1000,
            "budget": 2_000_000.0,
            "max_unit_price": 2640.0,
            "max_carbon_per_unit": None,
        }
    )
    assert strategy.name == "balanced"


def test_select_strategy_is_deterministic():
    constraints = {
        "material": "aluminum",
        "quantity": 1000,
        "budget": 2_000_000.0,
        "max_unit_price": 2640.0,
        "max_carbon_per_unit": 800.0,
    }
    assert [select_strategy(constraints).name for _ in range(3)] == [
        "low_carbon",
        "low_carbon",
        "low_carbon",
    ]
