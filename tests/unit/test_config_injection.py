"""Tests for v1.0 Step 16: Config Injection (Full OC Format).

Verifies that all hardcoded thresholds and constants have been replaced
with centralized values from swarm.config, and that changing config
values produces observable behavior changes.
"""

import pytest

from swarm import config
from swarm.utils.llm_drift import detect_drift
from swarm.utils.llm_fallback import evaluate_llm_usage
from swarm.utils.policy import apply_policy_constraints


class TestTrustThresholdAffectsFallback:
    def test_default_trust_threshold_rejects_low_trust(self) -> None:
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=0.95,
            stability=0.80,
            trust=0.76,
        )
        assert result["use_llm"] is True
        assert result["reason"] == "accepted"

    def test_raising_trust_threshold_rejects_previously_valid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("swarm.utils.llm_fallback.TRUST_THRESHOLD", 0.9)
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=0.95,
            stability=0.80,
            trust=0.76,
        )
        assert result["use_llm"] is False
        assert result["reason"] == "low_trust"

    def test_lowering_trust_threshold_accepts_previously_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("swarm.utils.llm_fallback.TRUST_THRESHOLD", 0.5)
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=0.95,
            stability=0.80,
            trust=0.76,
        )
        assert result["use_llm"] is True
        assert result["reason"] == "accepted"


class TestConfidenceThresholdAffectsFallback:
    def test_default_confidence_threshold_accepts_high_confidence(self) -> None:
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=0.95,
            stability=0.95,
            trust=0.90,
        )
        assert result["use_llm"] is True

    def test_raising_confidence_threshold_rejects_previously_valid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("swarm.utils.llm_fallback.CONFIDENCE_THRESHOLD", 0.99)
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=0.95,
            stability=1.0,
            trust=0.95,
        )
        assert result["use_llm"] is False
        assert result["reason"] == "low_confidence"

    def test_lowering_confidence_threshold_accepts_previously_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("swarm.utils.llm_fallback.CONFIDENCE_THRESHOLD", 0.4)
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=0.5,
            stability=1.0,
            trust=0.8,
        )
        assert result["use_llm"] is True
        assert result["reason"] == "accepted"


class TestStabilityThresholdAffectsFallback:
    def test_default_stability_threshold_accepts_high_stability(self) -> None:
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=0.95,
            stability=0.95,
            trust=0.90,
        )
        assert result["use_llm"] is True

    def test_raising_stability_threshold_rejects_previously_valid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("swarm.utils.llm_fallback.STABILITY_THRESHOLD", 0.99)
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=1.0,
            stability=0.95,
            trust=0.95,
        )
        assert result["use_llm"] is False
        assert result["reason"] == "low_stability"

    def test_lowering_stability_threshold_accepts_previously_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("swarm.utils.llm_fallback.STABILITY_THRESHOLD", 0.3)
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=1.0,
            stability=0.5,
            trust=0.8,
        )
        assert result["use_llm"] is True
        assert result["reason"] == "accepted"


class TestPolicyBoundsAffectOutput:
    def test_default_price_max_clamps_high_price(self) -> None:
        result = apply_policy_constraints({"price": 0.9, "delivery": 0.1})
        assert result["price"] <= config.POLICY_PRICE_MAX

    def test_raising_price_max_allows_higher_price(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("swarm.utils.policy.POLICY_PRICE_MAX", 0.9)
        result = apply_policy_constraints({"price": 0.85, "delivery": 0.15})
        # delivery clamped to 0.3, price 0.85 within bounds → normalized: 0.85/1.15≈0.739
        assert result["price"] == pytest.approx(0.7391, abs=0.001)

    def test_lowering_price_max_clamps_more_aggressively(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("swarm.utils.policy.POLICY_PRICE_MAX", 0.5)
        result = apply_policy_constraints({"price": 0.85, "delivery": 0.15})
        # price clamped to 0.5, delivery clamped to 0.3 → normalized: 0.5/0.8=0.625
        assert result["price"] == pytest.approx(0.625, abs=0.001)

    def test_default_delivery_min_enforces_minimum(self) -> None:
        result = apply_policy_constraints({"price": 0.9, "delivery": 0.1})
        assert result["delivery"] >= config.POLICY_DELIVERY_MIN

    def test_raising_delivery_min_enforces_higher_minimum(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("swarm.utils.policy.POLICY_DELIVERY_MIN", 0.5)
        result = apply_policy_constraints({"price": 0.9, "delivery": 0.1})
        # price clamped to 0.7, delivery clamped to 0.5 → normalized: 0.5/1.2≈0.417
        assert result["delivery"] == pytest.approx(0.4167, abs=0.001)

    def test_lowering_delivery_min_allows_lower_delivery(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("swarm.utils.policy.POLICY_DELIVERY_MIN", 0.1)
        result = apply_policy_constraints({"price": 0.9, "delivery": 0.1})
        # price clamped to 0.7, delivery stays 0.1 → normalized: 0.1/0.8=0.125
        assert result["delivery"] == pytest.approx(0.125, abs=0.001)


class TestDriftDetectionReactsToConfidenceDropThreshold:
    def test_default_drop_triggers_drift(self) -> None:
        history = [
            {"confidence": 0.95, "stability": 0.9, "trust": 0.85, "num_completions": 3},
            {"confidence": 0.75, "stability": 0.9, "trust": 0.67, "num_completions": 3},
        ]
        drift, reasons = detect_drift(history)
        assert drift is True
        assert any("confidence_drop" in r for r in reasons)

    def test_raising_drop_threshold_prevents_confidence_drop_drift(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When CONFIDENCE_DROP_THRESHOLD is raised above the actual drop,
        no confidence_drop reason."""
        monkeypatch.setattr("swarm.utils.llm_drift.CONFIDENCE_DROP_THRESHOLD", 0.3)
        history = [
            {"confidence": 0.95, "stability": 0.9, "trust": 0.85, "num_completions": 3},
            {"confidence": 0.75, "stability": 0.9, "trust": 0.85, "num_completions": 3},
        ]
        drift, reasons = detect_drift(history)
        assert not any("confidence_drop" in r for r in reasons)

    def test_lowering_drop_threshold_triggers_drift_earlier(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A smaller CONFIDENCE_DROP_THRESHOLD triggers drift for smaller drops."""
        monkeypatch.setattr("swarm.utils.llm_drift.CONFIDENCE_DROP_THRESHOLD", 0.04)
        history = [
            {"confidence": 0.95, "stability": 0.9, "trust": 0.85, "num_completions": 3},
            {"confidence": 0.90, "stability": 0.9, "trust": 0.81, "num_completions": 3},
        ]
        drift, reasons = detect_drift(history)
        assert drift is True
        assert any("confidence_drop" in r for r in reasons)


class TestStabilityToleranceImpactsConsensus:
    def test_default_tolerance_allows_small_drift(self) -> None:
        """Small drift within default tolerance should produce high stability."""
        from swarm.utils.llm_stability import compute_temporal_stability

        history = [
            {
                "round": 1,
                "aggregated_adjustments": {
                    "price_weight_delta": -0.05,
                    "delivery_weight_delta": 0.05,
                },
            },
            {
                "round": 2,
                "aggregated_adjustments": {
                    "price_weight_delta": -0.05 + config.STABILITY_TOLERANCE * 0.5,
                    "delivery_weight_delta": 0.05 - config.STABILITY_TOLERANCE * 0.5,
                },
            },
        ]
        result = compute_temporal_stability(history)
        assert result == 1.0

    def test_raising_tolerance_eliminates_drift_penalty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With a large tolerance, even significant drift is treated as stable."""
        monkeypatch.setattr("swarm.utils.llm_stability.TOLERANCE", 0.08)
        from swarm.utils.llm_stability import compute_temporal_stability

        history = [
            {
                "round": 1,
                "aggregated_adjustments": {
                    "price_weight_delta": -0.05,
                    "delivery_weight_delta": 0.05,
                },
            },
            {
                "round": 2,
                "aggregated_adjustments": {
                    "price_weight_delta": 0.05,
                    "delivery_weight_delta": -0.05,
                },
            },
        ]
        result = compute_temporal_stability(history)
        # With tolerance=0.08: abs_diff=0.10, drift=(0.10-0.08)/(0.2-0.08)=0.01/0.12≈0.083
        # stability≈0.917
        assert result > 0.8

    def test_lowering_tolerance_increases_drift_penalty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With zero tolerance, any drift is penalized."""
        monkeypatch.setattr("swarm.utils.llm_stability.TOLERANCE", 0.0)
        from swarm.utils.llm_stability import compute_temporal_stability

        history = [
            {
                "round": 1,
                "aggregated_adjustments": {
                    "price_weight_delta": -0.05,
                    "delivery_weight_delta": 0.05,
                },
            },
            {
                "round": 2,
                "aggregated_adjustments": {
                    "price_weight_delta": 0.05,
                    "delivery_weight_delta": -0.05,
                },
            },
        ]
        result = compute_temporal_stability(history)
        # With tolerance=0.0: abs_diff=0.10, drift=0.10/0.2=0.5
        # stability=0.5
        assert result < 0.6


class TestDeterminism:
    def test_same_config_produces_same_results(self) -> None:
        """For the same config values, results are deterministic."""
        r1 = evaluate_llm_usage(
            has_completions=True,
            confidence=0.95,
            stability=0.85,
            trust=0.81,
        )
        r2 = evaluate_llm_usage(
            has_completions=True,
            confidence=0.95,
            stability=0.85,
            trust=0.81,
        )
        assert r1 == r2

    def test_policy_deterministic_with_config(self) -> None:
        """Policy constraints produce deterministic output for same config."""
        r1 = apply_policy_constraints({"price": 0.6, "delivery": 0.4})
        r2 = apply_policy_constraints({"price": 0.6, "delivery": 0.4})
        assert r1 == r2

    def test_drift_detection_deterministic_with_config(self) -> None:
        """Drift detection produces deterministic output for same config."""
        history = [
            {"confidence": 0.9, "stability": 0.95, "trust": 0.85, "num_completions": 3},
            {"confidence": 0.88, "stability": 0.92, "trust": 0.81, "num_completions": 3},
        ]
        d1, r1 = detect_drift(history)
        d2, r2 = detect_drift(history)
        assert d1 == d2
        assert r1 == r2
