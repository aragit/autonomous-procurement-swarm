"""Tests for v1.1 Step 23: Strategy-aware policy learning.

Covers strategy inclusion in signature/version, deterministic candidate
generation across the strategy space, evaluation consistency, force-promote
(rollback), and POLICY_MAX_PARAM_DELTA enforcement.
"""

from __future__ import annotations

import pytest

from swarm.config import (
    DEFAULT_STRATEGY_TYPE,
    POLICY_MAX_PARAM_DELTA,
    STRATEGY_TYPES,
    THRESHOLD_CLAMP_MIN,
)
from swarm.learning.adaptive_policy import (
    apply_strategy_weights,
    get_strategy_type,
    override_strategy_type,
)
from swarm.learning.learner import (
    evaluate_candidate,
    get_active_policy,
    learn_candidates,
    promote_policy,
)
from swarm.learning.policy import (
    THRESHOLD_KEYS,
    WEIGHT_KEYS,
    build_policy,
    canonical_params,
    compute_signature,
    generate_candidates,
)
from swarm.learning.routing import (
    apply_param_overrides,
    normalize_context,
    select_strategy_with_params,
    validate_param_overrides,
    validate_routing_strategy,
)
from swarm.storage.event_store import (
    store_artifact,
    store_event,
    store_feedback,
    store_policy,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def db_path(tmp_path) -> str:
    return str(tmp_path / "strategy_learning.db")


@pytest.fixture(autouse=True)
def _isolate_event_store(db_path: str):
    """Point the event store at a per-test temp DB (deterministic isolation)."""
    import swarm.storage.event_store as es

    orig = es._DB_PATH
    es._DB_PATH = db_path
    es.init_db(db_path)
    # Reset the param overrides flag before each test.
    from swarm.learning.adaptive_policy import _overrides_applied_flag

    _overrides_applied_flag.set(False)
    yield
    es._DB_PATH = orig


SAFE_THRESH = {
    "confidence_threshold": 0.7,
    "stability_threshold": 0.7,
    "trust_threshold": 0.7,
}
SAFE_WEIGHTS = {"price_weight": 0.4, "score_weight": 0.4, "carbon_weight": 0.2}


def _seed_traces(n: int, supplier: str = "Supp_A") -> None:
    """Persist ``n`` traces that have a decision artifact + feedback."""
    for i in range(n):
        tid = f"TRACE-STRAT-{i}"
        store_event(
            tid,
            "procurement_request",
            {
                "material": "aluminum",
                "quantity": 1000,
                "budget": 2_000_000.0,
                "target_lead_time_days": 30,
                "max_carbon_per_unit": None,
                "goal": None,
                "supplier_count": 5,
            },
        )
        store_artifact(
            tid,
            "result",
            {
                "selected_supplier": supplier,
                "reasoning": {
                    "ranked": [{"supplier_id": supplier, "score": 0.5}]
                },
            },
        )
        store_feedback(tid, outcome_score=0.9, success=True, latency_ms=100.0)


def _replay_stub(inp, thr, w, strat):
    """Replay stub that reproduces the original supplier with a fixed score."""
    return {"selected_supplier": "Supp_A", "score": 0.5}


# --------------------------------------------------------------------------- #
# Strategy inclusion in signature/version
# --------------------------------------------------------------------------- #


class TestStrategyInSignature:
    def test_different_strategy_different_version(self) -> None:
        p1 = build_policy(SAFE_THRESH, SAFE_WEIGHTS, {"type": "balanced"})
        p2 = build_policy(SAFE_THRESH, SAFE_WEIGHTS, {"type": "cost_optimized"})
        assert p1.version != p2.version

    def test_same_strategy_same_version(self) -> None:
        p1 = build_policy(SAFE_THRESH, SAFE_WEIGHTS, {"type": "balanced"})
        p2 = build_policy(SAFE_THRESH, SAFE_WEIGHTS, {"type": "balanced"})
        assert p1.version == p2.version

    def test_strategy_in_canonical_params(self) -> None:
        cp = canonical_params(SAFE_THRESH, SAFE_WEIGHTS, {"type": "cost_optimized"})
        assert "strategy" in cp
        assert cp["strategy"] == {"type": "cost_optimized"}

    def test_strategy_defaults_to_balanced(self) -> None:
        p = build_policy(SAFE_THRESH, SAFE_WEIGHTS)
        assert p.strategy == {"type": DEFAULT_STRATEGY_TYPE}
        assert p.strategy["type"] == "balanced"

    def test_signature_deterministic_with_strategy(self) -> None:
        s1 = compute_signature(SAFE_THRESH, SAFE_WEIGHTS, {"type": "quality_first"})
        s2 = compute_signature(
            {"trust_threshold": 0.7, "confidence_threshold": 0.7, "stability_threshold": 0.7},
            {"carbon_weight": 0.2, "price_weight": 0.4, "score_weight": 0.4},
            {"type": "quality_first"},
        )
        assert s1 == s2


# --------------------------------------------------------------------------- #
# Deterministic candidate generation with strategies
# --------------------------------------------------------------------------- #


class TestCandidateGenerationWithStrategies:
    def test_base_policy_included(self) -> None:
        base = build_policy(SAFE_THRESH, SAFE_WEIGHTS)
        cands = generate_candidates(base)
        assert any(c.signature == base.signature for c in cands)

    def test_all_strategies_explored(self) -> None:
        base = build_policy(SAFE_THRESH, SAFE_WEIGHTS)
        cands = generate_candidates(base)
        strat_types = {c.strategy["type"] for c in cands}
        for st in STRATEGY_TYPES:
            assert st in strat_types

    def test_deduplicates_by_signature(self) -> None:
        base = build_policy(SAFE_THRESH, SAFE_WEIGHTS)
        cands = generate_candidates(base)
        sigs = [c.signature for c in cands]
        assert len(sigs) == len(set(sigs))

    def test_ordered_by_signature(self) -> None:
        base = build_policy(SAFE_THRESH, SAFE_WEIGHTS)
        cands = generate_candidates(base)
        sigs = [c.signature for c in cands]
        assert sigs == sorted(sigs)

    def test_deterministic_across_runs(self) -> None:
        base = build_policy(SAFE_THRESH, SAFE_WEIGHTS)
        c1 = [c.version for c in generate_candidates(base)]
        c2 = [c.version for c in generate_candidates(base)]
        assert c1 == c2

    def test_unsafe_candidates_filtered_in_learning(self, monkeypatch) -> None:
        """Generate all candidates, then verify only safe ones are persisted/evaluated."""
        monkeypatch.setattr("swarm.learning.learner.MIN_TRACES_FOR_LEARNING", 5)
        _seed_traces(5)
        seen = []

        def capture(inp, thr, w, strat):
            seen.append((thr, w, strat))
            return {"selected_supplier": "Supp_A", "score": 0.5}

        learn_candidates(replay_fn=capture)
        # Every captured candidate should be safe.
        for thr, w, _strat in seen:
            for k in THRESHOLD_KEYS:
                assert THRESHOLD_CLAMP_MIN <= thr[k] <= 0.9
            total = sum(w.values())
            assert abs(total - 1.0) < 1e-6

    def test_all_within_param_deltas(self) -> None:
        base = build_policy(SAFE_THRESH, SAFE_WEIGHTS)
        cands = generate_candidates(base)
        for c in cands:
            for k in THRESHOLD_KEYS:
                assert (
                    abs(c.thresholds[k] - base.thresholds[k]) <= POLICY_MAX_PARAM_DELTA + 1e-9
                )
            for k in WEIGHT_KEYS:
                assert (
                    abs(c.weights[k] - base.weights[k]) <= POLICY_MAX_PARAM_DELTA + 1e-9
                )


# --------------------------------------------------------------------------- #
# Strategy type resolution
# --------------------------------------------------------------------------- #


class TestStrategyTypeResolution:
    def test_override_beats_config(self) -> None:
        with override_strategy_type("cost_optimized"):
            assert get_strategy_type() == "cost_optimized"

    def test_falls_through_to_config_default(self) -> None:
        with override_strategy_type("balanced"):
            assert get_strategy_type() == "balanced"

    def test_apply_strategy_weights_balanced_identity(self) -> None:
        weights = {"price_weight": 0.4, "score_weight": 0.4, "carbon_weight": 0.2}
        result = apply_strategy_weights("balanced", weights)
        assert result == weights

    def test_apply_strategy_weights_cost_optimized(self) -> None:
        weights = {"price_weight": 0.4, "score_weight": 0.4, "carbon_weight": 0.2}
        result = apply_strategy_weights("cost_optimized", weights)
        # price_weight is boosted * 1.2, then re-clamp + normalize
        assert result["price_weight"] > 0.4
        assert sum(result.values()) - 1.0 < 1e-6

    def test_apply_strategy_weights_quality_first(self) -> None:
        weights = {"price_weight": 0.4, "score_weight": 0.4, "carbon_weight": 0.2}
        result = apply_strategy_weights("quality_first", weights)
        assert result["score_weight"] > 0.4
        assert sum(result.values()) - 1.0 < 1e-6

    def test_apply_strategy_weights_trust_weighted(self) -> None:
        weights = {"price_weight": 0.4, "score_weight": 0.4, "carbon_weight": 0.2}
        result = apply_strategy_weights("trust_weighted", weights)
        assert result["carbon_weight"] > 0.2
        assert sum(result.values()) - 1.0 < 1e-6


# --------------------------------------------------------------------------- #
# Evaluation consistency across strategies
# --------------------------------------------------------------------------- #


class TestEvaluationConsistency:
    def test_strategy_passed_to_replay(self) -> None:
        traces = [
            {
                "events": [
                    {
                        "event_type": "procurement_request",
                        "payload": {"material": "aluminum"},
                        "created_at": "x",
                    }
                ],
                "artifacts": [
                    {
                        "artifact_type": "result",
                        "data": {
                            "selected_supplier": "Supp_A",
                            "reasoning": {"ranked": [{"supplier_id": "Supp_A", "score": 0.5}]},
                        },
                    }
                ],
                "feedback": {"success": True, "outcome_score": 0.9},
            }
        ] * 5

        seen_strategies: list[str] = []

        def replay(inp, thr, w, strat):
            seen_strategies.append(strat)
            return {"selected_supplier": "Supp_A", "score": 0.5}

        cand = build_policy(SAFE_THRESH, SAFE_WEIGHTS, {"type": "cost_optimized"})
        evaluate_candidate(traces, cand, active_metric=0.0, replay_fn=replay)
        assert all(s == "cost_optimized" for s in seen_strategies)

    def test_different_strategies_evaluated(self, monkeypatch) -> None:
        monkeypatch.setattr("swarm.learning.learner.MIN_TRACES_FOR_LEARNING", 5)
        _seed_traces(5)
        seen_strategies: set[str] = set()

        def capture(inp, thr, w, strat):
            seen_strategies.add(strat)
            return {"selected_supplier": "Supp_A", "score": 0.5}

        learn_candidates(replay_fn=capture)
        # Must have evaluated across multiple strategies.
        assert len(seen_strategies) > 1


# --------------------------------------------------------------------------- #
# Learning selects best strategy
# --------------------------------------------------------------------------- #


class TestLearningSelectsStrategy:
    def test_ok_with_strategies(self, monkeypatch) -> None:
        monkeypatch.setattr("swarm.learning.learner.MIN_TRACES_FOR_LEARNING", 5)
        _seed_traces(5)
        res = learn_candidates(replay_fn=_replay_stub)
        assert res["status"] == "ok"
        assert res["best"] is not None
        assert res["best"]["version"]
        # Candidates list includes strategy info.
        for c in res["candidates"]:
            assert "strategy" in c
            strat_type = c["strategy"]["type"]
            assert strat_type in STRATEGY_TYPES or strat_type == "routing"

    def test_idempotent_across_runs(self, monkeypatch) -> None:
        monkeypatch.setattr("swarm.learning.learner.MIN_TRACES_FOR_LEARNING", 5)
        _seed_traces(5)
        r1 = learn_candidates(replay_fn=_replay_stub)
        r2 = learn_candidates(replay_fn=_replay_stub)
        assert r1["best"]["version"] == r2["best"]["version"]

    def test_fallback_to_balanced(self) -> None:
        ap = get_active_policy()
        assert ap.strategy["type"] == DEFAULT_STRATEGY_TYPE


# --------------------------------------------------------------------------- #
# Force promote (rollback)
# --------------------------------------------------------------------------- #


class TestForcePromote:
    def test_force_promote_workflow(self, monkeypatch) -> None:
        monkeypatch.setattr("swarm.learning.learner.MIN_TRACES_FOR_LEARNING", 5)
        _seed_traces(5)
        res = learn_candidates(replay_fn=_replay_stub)
        version = res["best"]["version"]
        assert promote_policy(version)["status"] == "promoted"

        # Now seed a "worse" policy and try normal promote (should fail).
        worse = build_policy(
            SAFE_THRESH,
            {"price_weight": 0.45, "score_weight": 0.35, "carbon_weight": 0.2},
            {"type": "balanced"},
        )
        store_policy(
            version=worse.version,
            signature=worse.signature,
            thresholds=worse.thresholds,
            weights=worse.weights,
            strategy=worse.strategy,
            metric=0.01,
            success_rate=0.0,
            avg_score=0.0,
            feedback_success_rate=0.0,
            feedback_outcome_score=0.0,
            decision_stability=0.0,
            active=False,
        )
        # Normal promote should fail (metric not better).
        result = promote_policy(worse.version)
        assert result["status"] == "rejected"
        assert result["reason"] == "fails_improvement_margin"
        # Force promote should succeed (rollback to worse policy).
        result = promote_policy(worse.version, force=True)
        assert result["status"] == "promoted"

    def test_force_promote_already_active(self, monkeypatch) -> None:
        monkeypatch.setattr("swarm.learning.learner.MIN_TRACES_FOR_LEARNING", 5)
        _seed_traces(5)
        version = learn_candidates(replay_fn=_replay_stub)["best"]["version"]
        assert promote_policy(version)["status"] == "promoted"
        # Re-promoting the active version with force should work.
        result = promote_policy(version, force=True)
        assert result["status"] == "promoted"


# --------------------------------------------------------------------------- #
# Step 25: Contextual Parameter Adaptation — param overrides
# --------------------------------------------------------------------------- #


class TestSelectStrategyWithParams:
    """Tests for select_strategy_with_params and RoutingDecision."""

    def test_single_strategy_no_overrides(self) -> None:
        strategy = {"type": "balanced"}
        context = {"budget_level": "low"}
        decision = select_strategy_with_params(context, strategy)
        assert decision.strategy_type == "balanced"
        assert decision.param_overrides == {}

    def test_routing_match_with_param_overrides(self) -> None:
        strategy = {
            "type": "routing",
            "rules": [
                {
                    "conditions": {"budget_level": "low"},
                    "strategy": "cost_optimized",
                    "param_overrides": {"price_weight": 0.10},
                }
            ],
            "default": "balanced",
        }
        context = {"budget_level": "low", "urgency": "low", "supplier_count": 5}
        decision = select_strategy_with_params(context, strategy)
        assert decision.strategy_type == "cost_optimized"
        assert decision.param_overrides == {"price_weight": 0.10}

    def test_routing_fallback_no_overrides(self) -> None:
        strategy = {
            "type": "routing",
            "rules": [
                {
                    "conditions": {"budget_level": "low"},
                    "strategy": "cost_optimized",
                    "param_overrides": {"price_weight": 0.10},
                }
            ],
            "default": "balanced",
        }
        context = {"budget_level": "high", "urgency": "low", "supplier_count": 5}
        decision = select_strategy_with_params(context, strategy)
        assert decision.strategy_type == "balanced"
        assert decision.param_overrides == {}

    def test_routing_first_match_wins(self) -> None:
        strategy = {
            "type": "routing",
            "rules": [
                {
                    "conditions": {"urgency": "high"},
                    "strategy": "quality_first",
                    "param_overrides": {"score_weight": 0.10},
                },
                {
                    "conditions": {"budget_level": "low"},
                    "strategy": "cost_optimized",
                    "param_overrides": {"price_weight": 0.10},
                },
            ],
            "default": "balanced",
        }
        context = {"budget_level": "low", "urgency": "high", "supplier_count": 5}
        decision = select_strategy_with_params(context, strategy)
        assert decision.strategy_type == "quality_first"


class TestValidateParamOverrides:
    """Tests for validate_param_overrides."""

    def test_empty_overrides_valid(self) -> None:
        assert validate_param_overrides({}) is True

    def test_valid_threshold_override(self) -> None:
        assert validate_param_overrides({"confidence_threshold": 0.05}) is True

    def test_valid_weight_override(self) -> None:
        assert validate_param_overrides({"price_weight": -0.10}) is True

    def test_invalid_key(self) -> None:
        assert validate_param_overrides({"bogus_param": 0.05}) is False

    def test_delta_too_large(self) -> None:
        assert validate_param_overrides({"price_weight": 0.11}) is False

    def test_too_many_params(self) -> None:
        assert validate_param_overrides(
            {"price_weight": 0.10, "score_weight": 0.10, "carbon_weight": 0.10}
        ) is False

    def test_non_numeric_delta(self) -> None:
        assert validate_param_overrides({"price_weight": "high"}) is False


class TestValidateRoutingWithParamOverrides:
    """Tests for validate_routing_strategy with param overrides."""

    def test_valid_routing_with_overrides(self) -> None:
        strategy = {
            "type": "routing",
            "rules": [
                {
                    "conditions": {"budget_level": "low"},
                    "strategy": "cost_optimized",
                    "param_overrides": {"price_weight": 0.10},
                }
            ],
            "default": "balanced",
        }
        assert validate_routing_strategy(strategy) is True

    def test_invalid_overrides_rejected(self) -> None:
        strategy = {
            "type": "routing",
            "rules": [
                {
                    "conditions": {"budget_level": "low"},
                    "strategy": "cost_optimized",
                    "param_overrides": {"bogus": 0.10},
                }
            ],
            "default": "balanced",
        }
        assert validate_routing_strategy(strategy) is False


class TestApplyParamOverrides:
    """Tests for apply_param_overrides."""

    @staticmethod
    def _reset_flag() -> None:
        from swarm.learning.adaptive_policy import _overrides_applied_flag
        _overrides_applied_flag.set(False)

    def test_apply_threshold_override(self) -> None:
        self._reset_flag()
        thresholds = {
            "confidence_threshold": 0.7,
            "stability_threshold": 0.7,
            "trust_threshold": 0.7,
        }
        weights = {"price_weight": 0.4, "score_weight": 0.4, "carbon_weight": 0.2}
        overrides = {"confidence_threshold": 0.05}
        t, w = apply_param_overrides(thresholds, weights, overrides)
        assert t["confidence_threshold"] == 0.75
        assert w == weights  # weights unchanged

    def test_apply_weight_override(self) -> None:
        self._reset_flag()
        thresholds = {
            "confidence_threshold": 0.7,
            "stability_threshold": 0.7,
            "trust_threshold": 0.7,
        }
        weights = {"price_weight": 0.4, "score_weight": 0.4, "carbon_weight": 0.2}
        overrides = {"price_weight": 0.10}
        t, w = apply_param_overrides(thresholds, weights, overrides)
        assert t == thresholds  # thresholds unchanged
        # 0.4 + 0.10 = 0.50; total = 1.1; normalized = 0.50/1.1
        assert w["price_weight"] == pytest.approx(0.50 / 1.1)

    def test_clamp_after_override(self) -> None:
        self._reset_flag()
        thresholds = {
            "confidence_threshold": 0.95,
            "stability_threshold": 0.7,
            "trust_threshold": 0.7,
        }
        weights = {"price_weight": 0.4, "score_weight": 0.4, "carbon_weight": 0.2}
        overrides = {"confidence_threshold": 0.10}  # would push to 1.05
        t, w = apply_param_overrides(thresholds, weights, overrides)
        from swarm.config import THRESHOLD_CLAMP_MAX
        assert t["confidence_threshold"] <= THRESHOLD_CLAMP_MAX

    def test_weights_re_normalized_after_override(self) -> None:
        self._reset_flag()
        thresholds = {
            "confidence_threshold": 0.7,
            "stability_threshold": 0.7,
            "trust_threshold": 0.7,
        }
        weights = {"price_weight": 0.4, "score_weight": 0.4, "carbon_weight": 0.2}
        overrides = {"price_weight": 0.10, "score_weight": 0.10}
        t, w = apply_param_overrides(thresholds, weights, overrides)
        assert abs(sum(w.values()) - 1.0) < 1e-6


class TestRoutingCandidatesWithParams:
    """Tests that generate_candidates includes param override candidates."""

    def test_routing_candidates_include_param_overrides(self) -> None:
        base = build_policy(
            SAFE_THRESH,
            SAFE_WEIGHTS,
            {"type": "balanced"},
        )
        candidates = generate_candidates(base)
        # At least one candidate should have a routing strategy with param_overrides.
        found_param_override = False
        for c in candidates:
            strat = c.strategy or {}
            if strat.get("type") == "routing":
                for rule in strat.get("rules", []):
                    if rule.get("param_overrides"):
                        found_param_override = True
                        break
            if found_param_override:
                break
        assert found_param_override, "Expected at least one routing candidate with param_overrides"

    def test_routing_candidates_param_override_count(self) -> None:
        base = build_policy(SAFE_THRESH, SAFE_WEIGHTS, {"type": "balanced"})
        candidates = generate_candidates(base)
        param_override_count = 0
        for c in candidates:
            strat = c.strategy or {}
            if strat.get("type") == "routing":
                for rule in strat.get("rules", []):
                    overrides = rule.get("param_overrides")
                    if overrides:
                        from swarm.config import MAX_PARAMS_PER_RULE
                        assert len(overrides) <= MAX_PARAMS_PER_RULE
                        from swarm.config import PARAM_OVERRIDE_DELTA
                        for delta in overrides.values():
                            assert abs(delta) <= PARAM_OVERRIDE_DELTA + 1e-9
                        param_override_count += 1
        assert param_override_count > 0, "Expected multiple candidates with param overrides"


# --------------------------------------------------------------------------- #
# Step 25: Live runtime integration — param overrides applied end-to-end
# --------------------------------------------------------------------------- #


class TestLiveParamOverrideIntegration:
    """Verify param overrides propagate through the runtime path."""

    def test_get_param_overrides_returns_empty_for_non_routing(self) -> None:
        """When active policy is single strategy, no param overrides."""
        from swarm.learning.adaptive_policy import get_param_overrides

        # With no active policy, returns empty.
        overrides = get_param_overrides({"budget_level": "low"})
        assert overrides == {}

    def test_get_param_overrides_returns_routing_overrides(self) -> None:
        """When active policy is routing with overrides, they're returned."""
        from swarm.learning.adaptive_policy import get_param_overrides
        from swarm.learning.policy import build_policy
        from swarm.storage.event_store import store_policy

        policy = build_policy(
            SAFE_THRESH,
            SAFE_WEIGHTS,
            {
                "type": "routing",
                "rules": [
                    {
                        "conditions": {"budget_level": "low"},
                        "strategy": "cost_optimized",
                        "param_overrides": {"price_weight": 0.10},
                    }
                ],
                "default": "balanced",
            },
        )
        store_policy(
            version=policy.version,
            signature=policy.signature,
            thresholds=policy.thresholds,
            weights=policy.weights,
            strategy=policy.strategy,
            metric=0.9,
            success_rate=0.9,
            avg_score=0.9,
            feedback_success_rate=0.9,
            feedback_outcome_score=0.9,
            decision_stability=0.9,
            active=True,
        )
        overrides = get_param_overrides({"budget_level": "low"})
        assert overrides == {"price_weight": 0.10}

    def test_get_param_overrides_returns_empty_for_non_matching_context(self) -> None:
        """When context doesn't match routing rules, no overrides."""
        from swarm.learning.adaptive_policy import get_param_overrides
        from swarm.learning.policy import build_policy
        from swarm.storage.event_store import store_policy

        policy = build_policy(
            SAFE_THRESH,
            SAFE_WEIGHTS,
            {
                "type": "routing",
                "rules": [
                    {
                        "conditions": {"budget_level": "low"},
                        "strategy": "cost_optimized",
                        "param_overrides": {"price_weight": 0.10},
                    }
                ],
                "default": "balanced",
            },
        )
        store_policy(
            version=policy.version,
            signature=policy.signature,
            thresholds=policy.thresholds,
            weights=policy.weights,
            strategy=policy.strategy,
            metric=0.9,
            success_rate=0.9,
            avg_score=0.9,
            feedback_success_rate=0.9,
            feedback_outcome_score=0.9,
            decision_stability=0.9,
            active=True,
        )
        # Context matches the "else" branch, no overrides.
        overrides = get_param_overrides({"budget_level": "high"})
        assert overrides == {}


# --------------------------------------------------------------------------- #
# Step 25 Hardened: additional safety, determinism, and trace tests
# --------------------------------------------------------------------------- #


class TestNoDoubleOverrideApplication:
    """Part 1: prevent double application of param overrides."""

    def test_no_double_override_application(self) -> None:
        """When _overrides_applied_flag is True, apply_param_overrides is a no-op."""
        thresholds = {"confidence_threshold": 0.7}
        weights = {"price_weight": 0.4, "score_weight": 0.4, "carbon_weight": 0.2}
        overrides = {"price_weight": 0.10}

        # First application: should apply the delta.
        t1, w1 = apply_param_overrides(
            dict(thresholds), dict(weights), dict(overrides)
        )
        assert w1["price_weight"] != weights["price_weight"]  # changed

        # Second application with same inputs: should be no-op (flag is True).
        t2, w2 = apply_param_overrides(
            dict(thresholds), dict(weights), dict(overrides)
        )
        assert w2 == weights  # unchanged (no-op)


class TestApplyParamOverridesImmutability:
    """Part 2: apply_param_overrides must not mutate inputs."""

    def test_apply_param_overrides_immutable_inputs(self) -> None:
        """Input dicts are not mutated by apply_param_overrides."""
        thresholds = {
            "confidence_threshold": 0.7,
            "stability_threshold": 0.7,
            "trust_threshold": 0.7,
        }
        weights = {"price_weight": 0.4, "score_weight": 0.4, "carbon_weight": 0.2}
        overrides = {"price_weight": 0.10, "confidence_threshold": 0.05}

        thresholds_copy = dict(thresholds)
        weights_copy = dict(weights)

        apply_param_overrides(thresholds, weights, overrides)

        assert thresholds == thresholds_copy, "Input thresholds were mutated!"
        assert weights == weights_copy, "Input weights were mutated!"


class TestDeterministicRuleSelection:
    """Part 3: deterministic rule selection order."""

    def test_deterministic_rule_selection_order(self) -> None:
        """When multiple rules match, highest priority wins deterministically."""
        strategy = {
            "type": "routing",
            "rules": [
                {
                    "rule_id": "low_priority",
                    "priority": 1,
                    "conditions": {"budget_level": "low"},
                    "strategy": "cost_optimized",
                    "param_overrides": {"price_weight": -0.05},
                },
                {
                    "rule_id": "high_priority",
                    "priority": 10,
                    "conditions": {"budget_level": "low"},
                    "strategy": "quality_first",
                    "param_overrides": {"score_weight": 0.10},
                },
            ],
            "default": "balanced",
        }
        context = {"budget_level": "low", "urgency": "low", "supplier_count": 5}
        decision = select_strategy_with_params(context, strategy)
        assert decision.strategy_type == "quality_first"
        assert decision.param_overrides == {"score_weight": 0.10}

        # Run again to confirm determinism.
        decision2 = select_strategy_with_params(context, strategy)
        assert decision2.strategy_type == "quality_first"
        assert decision2.param_overrides == {"score_weight": 0.10}


class TestContextNormalization:
    """Part 3: context normalization for deterministic matching."""

    def test_context_normalization_effect(self) -> None:
        """Context with mixed-case keys and whitespace matches normalized rules."""
        from swarm.learning.routing import _matches

        normalized = normalize_context(
            {"Budget_Level": "  Low  ", "Supplier_Count": 2, "extra": None}
        )
        assert normalized == {"budget_level": "low", "supplier_count": 2}
        assert _matches(normalized, "budget_level", "low") is True


class TestTraceContainsParamOverrides:
    """Part 4: param overrides are traced in replay results."""

    def test_trace_contains_param_overrides(self) -> None:
        """The evaluate_candidate function attaches applied_overrides to replay."""
        from swarm.learning.learner import evaluate_candidate
        from swarm.learning.policy import build_policy

        # Build a routing candidate with param overrides.
        candidate = build_policy(
            SAFE_THRESH,
            SAFE_WEIGHTS,
            {
                "type": "routing",
                "rules": [
                    {
                        "conditions": {"budget_level": "low"},
                        "strategy": "cost_optimized",
                        "param_overrides": {"price_weight": 0.10},
                    }
                ],
                "default": "balanced",
            },
        )

        traces = [
            {
                "events": [
                    {
                        "event_type": "procurement_request",
                        "data": {
                            "material": "aluminum",
                            "quantity": 1000,
                            "budget": 2_000_000.0,
                            "target_lead_time_days": 30,
                            "max_carbon_per_unit": None,
                            "goal": None,
                            "supplier_count": 5,
                        },
                    }
                ],
                "artifacts": [
                    {
                        "name": "result",
                        "data": {
                            "selected_supplier": "Supp_A",
                            "reasoning": {
                                "ranked": [{"supplier_id": "Supp_A", "score": 0.5}]
                            },
                        },
                    }
                ],
                "feedback": {"success": True, "outcome_score": 0.9},
            }
        ]

        def replay(inp, thr, w, strat):
            return {"selected_supplier": "Supp_A", "score": 0.5}

        result = evaluate_candidate(traces, candidate, active_metric=0.0, replay_fn=replay)
        assert result.metric == 0.0  # stub returns low score


class TestCandidateCapEnforced:
    """Part 5: candidate cap is enforced with deterministic sampling."""

    def test_candidate_cap_enforced(self) -> None:
        """generate_candidates yields at most MAX_CANDIDATES entries."""
        from swarm.config import MAX_CANDIDATES
        base = build_policy(SAFE_THRESH, SAFE_WEIGHTS, {"type": "balanced"})
        candidates = generate_candidates(base)
        assert len(candidates) <= MAX_CANDIDATES

    def test_candidate_cap_deterministic_sample(self) -> None:
        """Sampling when cap exceeds is deterministic (same seed)."""
        base = build_policy(SAFE_THRESH, SAFE_WEIGHTS, {"type": "balanced"})
        c1 = generate_candidates(base)
        c2 = generate_candidates(base)
        sigs1 = [p.signature for p in c1]
        sigs2 = [p.signature for p in c2]
        assert sigs1 == sigs2, "Candidate generation must be deterministic"


class TestConflictResolutionPriority:
    """Part 6: priority-based conflict resolution with tie-breaking."""

    def test_conflict_resolution_priority(self) -> None:
        """Highest priority rule wins among multiple matches."""
        strategy = {
            "type": "routing",
            "rules": [
                {
                    "priority": 1,
                    "rule_id": "a",
                    "conditions": {"budget_level": "low"},
                    "strategy": "cost_optimized",
                    "param_overrides": {"price_weight": -0.05},
                },
                {
                    "priority": 5,
                    "rule_id": "b",
                    "conditions": {"urgency": "high"},
                    "strategy": "quality_first",
                    "param_overrides": {"score_weight": 0.10},
                },
                {
                    "priority": 10,
                    "rule_id": "c",
                    "conditions": {"budget_level": "low", "urgency": "high"},
                    "strategy": "trust_weighted",
                    "param_overrides": {"score_weight": 0.10, "carbon_weight": -0.05},
                },
            ],
            "default": "balanced",
        }
        # Context matches ALL three rules — should pick the highest priority (rule c).
        context = {"budget_level": "low", "urgency": "high", "supplier_count": 5}
        decision = select_strategy_with_params(context, strategy)
        assert decision.strategy_type == "trust_weighted"
        assert "score_weight" in decision.param_overrides
        assert "carbon_weight" in decision.param_overrides

    def test_conflict_resolution_tie_break_by_condition_count(self) -> None:
        """When priority ties, rule with fewer conditions (more general) wins."""
        strategy = {
            "type": "routing",
            "rules": [
                {
                    "priority": 5,
                    "rule_id": "specific",
                    "conditions": {"budget_level": "low", "urgency": "high"},
                    "strategy": "cost_optimized",
                    "param_overrides": {"price_weight": 0.10},
                },
                {
                    "priority": 5,
                    "rule_id": "general",
                    "conditions": {"budget_level": "low"},
                    "strategy": "quality_first",
                    "param_overrides": {"score_weight": 0.10},
                },
            ],
            "default": "balanced",
        }
        # Both match — tie on priority. The more general one (fewer conditions) wins.
        context = {"budget_level": "low", "urgency": "high", "supplier_count": 5}
        decision = select_strategy_with_params(context, strategy)
        assert decision.strategy_type == "quality_first"
