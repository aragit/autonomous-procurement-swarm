"""Tests for Contextual Bandit (LinUCB) implementation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from mesh.neuro.bandits import (
    BanditContext,
    LinUCBBandit,
    NegotiationStrategy,
    compute_reward,
    get_default_bandit,
)


class TestBanditContext:
    """Tests for BanditContext feature extraction."""

    def test_to_vector_returns_correct_shape(self):
        ctx = BanditContext(
            urgency=0.5,
            budget_margin=0.3,
            supplier_rating=0.8,
            material_complexity=0.2,
            historical_win_rate=0.6,
            round_number=0.1,
        )
        vec = ctx.to_vector()
        assert vec.shape == (6,)
        expected = [0.5, 0.3, 0.8, 0.2, 0.6, 0.1]
        for actual, exp in zip(vec, expected, strict=True):
            assert abs(actual - exp) < 1e-6

    def test_from_requirement_builds_context(self):
        requirement = {
            "constraints": {
                "budget": 100000.0,
                "target_lead_time_days": 15,
                "material": "steel",
            }
        }
        supplier = {
            "supplier_id": "TestCorp",
            "reliability_score": 0.85,
            "esg_carbon_per_unit": 1500.0,
        }
        pool_data = {
            "spot_price": 500.0,
            "quantity": 1000,
        }

        ctx = BanditContext.from_requirement(requirement, supplier, pool_data, round_num=2)

        assert ctx.urgency > 0.5  # 15 days is urgent
        assert 0.0 <= ctx.budget_margin <= 1.0
        assert ctx.supplier_rating == 0.85
        assert 0.0 <= ctx.material_complexity <= 1.0
        assert ctx.round_number == 0.2  # 2/10

    def test_from_requirement_handles_missing_data(self):
        """Test graceful handling of missing optional fields."""
        requirement = {"constraints": {}}
        supplier = {}
        pool_data = {}

        ctx = BanditContext.from_requirement(requirement, supplier, pool_data)

        assert 0.0 <= ctx.urgency <= 1.0
        assert 0.0 <= ctx.budget_margin <= 1.0
        assert 0.0 <= ctx.supplier_rating <= 1.0
        assert 0.0 <= ctx.material_complexity <= 1.0
        assert 0.0 <= ctx.historical_win_rate <= 1.0
        assert ctx.round_number == 0.0


class TestLinUCBBandit:
    """Tests for LinUCB contextual bandit."""

    def test_initialization_default_strategies(self):
        bandit = LinUCBBandit()
        assert len(bandit.strategies) == 5
        assert NegotiationStrategy.BALANCED_CONCESSION in bandit.strategies

    def test_initialization_custom_strategies(self):
        strategies = [
            NegotiationStrategy.AGGRESSIVE_ANCHOR,
            NegotiationStrategy.BALANCED_CONCESSION,
        ]
        bandit = LinUCBBandit(strategies=strategies)
        assert len(bandit.strategies) == 2

    def test_select_action_returns_valid_strategy(self):
        bandit = LinUCBBandit()
        ctx = BanditContext(
            urgency=0.5, budget_margin=0.5, supplier_rating=0.5,
            material_complexity=0.5, historical_win_rate=0.5, round_number=0.0
        )
        action = bandit.select_action(ctx)
        assert action in bandit.strategies

    def test_select_action_explores_initially(self):
        """Initially all actions should have similar scores, so selection varies."""
        bandit = LinUCBBandit(alpha=1.0)
        ctx = BanditContext(
            urgency=0.5, budget_margin=0.5, supplier_rating=0.5,
            material_complexity=0.5, historical_win_rate=0.5, round_number=0.0
        )
        # Run multiple times - should eventually pick different actions due to exploration
        actions = {bandit.select_action(ctx) for _ in range(20)}
        assert len(actions) > 1  # Should explore multiple strategies

    def test_update_increments_counts(self):
        bandit = LinUCBBandit()
        ctx = BanditContext(
            urgency=0.5, budget_margin=0.5, supplier_rating=0.5,
            material_complexity=0.5, historical_win_rate=0.5, round_number=0.0
        )
        action = bandit.select_action(ctx)
        initial_count = bandit.action_counts[action]
        bandit.update(action, ctx, 0.8)
        assert bandit.action_counts[action] == initial_count + 1

    def test_update_changes_theta(self):
        """Theta should change after update."""
        bandit = LinUCBBandit()
        ctx = BanditContext(
            urgency=0.5, budget_margin=0.5, supplier_rating=0.5,
            material_complexity=0.5, historical_win_rate=0.5, round_number=0.0
        )
        action = bandit.select_action(ctx)
        theta_before = bandit.theta[action].copy()
        bandit.update(action, ctx, 1.0)
        theta_after = bandit.theta[action]
        # Theta should have changed
        assert not np.array_equal(theta_before, theta_after)

    def test_get_action_stats(self):
        bandit = LinUCBBandit()
        ctx = BanditContext(
            urgency=0.5, budget_margin=0.5, supplier_rating=0.5,
            material_complexity=0.5, historical_win_rate=0.5, round_number=0.0
        )
        action = bandit.select_action(ctx)
        bandit.update(action, ctx, 0.7)
        bandit.update(action, ctx, 0.9)

        stats = bandit.get_action_stats()
        assert action.value in stats
        assert stats[action.value]["count"] == 2
        assert abs(stats[action.value]["avg_reward"] - 0.8) < 0.01

    def test_persistence_roundtrip(self):
        """Test save_state and load_state preserve all parameters."""
        bandit = LinUCBBandit(alpha=0.5, lambda_reg=2.0)
        ctx = BanditContext(
            urgency=0.5, budget_margin=0.5, supplier_rating=0.5,
            material_complexity=0.5, historical_win_rate=0.5, round_number=0.0
        )
        action = bandit.select_action(ctx)
        bandit.update(action, ctx, 0.8)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bandit_state.json"
            bandit.save_state(path)
            loaded = LinUCBBandit.load_state(path)

        assert loaded.alpha == bandit.alpha
        assert loaded.lambda_reg == bandit.lambda_reg
        assert loaded.context_dim == bandit.context_dim
        assert loaded.strategies == bandit.strategies
        assert loaded.action_counts == bandit.action_counts
        assert loaded.total_rewards == bandit.total_rewards
        for a in bandit.strategies:
            assert np.allclose(loaded.A[a], bandit.A[a])
            assert np.allclose(loaded.b[a], bandit.b[a])
            assert np.allclose(loaded.theta[a], bandit.theta[a])

    def test_cold_start_fallback_uniform(self):
        """With no history, selection should be roughly uniform over many trials."""
        bandit = LinUCBBandit(alpha=1.0)
        ctx = BanditContext(
            urgency=0.5, budget_margin=0.5, supplier_rating=0.5,
            material_complexity=0.5, historical_win_rate=0.5, round_number=0.0
        )
        counts = dict.fromkeys(bandit.strategies, 0)
        for _ in range(1000):
            action = bandit.select_action(ctx)
            counts[action] += 1
        # Each strategy should be selected at least some times
        for count in counts.values():
            assert count > 50  # At least 5% each


class TestComputeReward:
    """Tests for reward computation."""

    def test_reward_bounds(self):
        decision = {"composite_score": 0.8}
        quote = {
            "price": 100.0,
            "terms": "net_30",
            "metadata": {"quantity": 1000},
        }
        requirement = {"constraints": {"budget": 200000.0}}

        reward = compute_reward(decision, quote, requirement, negotiation_rounds=1)
        assert 0.0 <= reward <= 1.0

    def test_reward_cost_reduction(self):
        """Lower cost relative to budget should give higher reward."""
        decision = {"composite_score": 0.8}
        quote = {"price": 50.0, "terms": "net_30", "metadata": {"quantity": 1000}}
        requirement = {"constraints": {"budget": 100000.0}}

        reward_low_cost = compute_reward(decision, quote, requirement, negotiation_rounds=1)

        quote_high = {"price": 150.0, "terms": "net_30", "metadata": {"quantity": 1000}}
        reward_high_cost = compute_reward(decision, quote_high, requirement, negotiation_rounds=1)

        assert reward_low_cost > reward_high_cost

    def test_reward_payment_terms(self):
        """Better payment terms should increase reward."""
        decision = {"composite_score": 0.8}
        requirement = {"constraints": {"budget": 100000.0}}

        quote_cod = {"price": 100.0, "terms": "cod", "metadata": {"quantity": 1000}}
        quote_net60 = {"price": 100.0, "terms": "net_60", "metadata": {"quantity": 1000}}

        reward_cod = compute_reward(decision, quote_cod, requirement, negotiation_rounds=1)
        reward_net60 = compute_reward(decision, quote_net60, requirement, negotiation_rounds=1)

        assert reward_cod > reward_net60

    def test_reward_convergence_speed(self):
        """Fewer rounds should give higher reward."""
        decision = {"composite_score": 0.8}
        quote = {"price": 100.0, "terms": "net_30", "metadata": {"quantity": 1000}}
        requirement = {"constraints": {"budget": 100000.0}}

        reward_fast = compute_reward(decision, quote, requirement, negotiation_rounds=1)
        reward_slow = compute_reward(decision, quote, requirement, negotiation_rounds=5)

        assert reward_fast > reward_slow


class TestGetDefaultBandit:
    """Tests for bandit factory."""

    def test_returns_bandit_instance(self):
        bandit = get_default_bandit()
        assert isinstance(bandit, LinUCBBandit)
        assert len(bandit.strategies) == 5

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
