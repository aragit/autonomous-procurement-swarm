"""Unit tests for v0.9 Step 9: LLM Fallback and Failure Handling.

Tests ``evaluate_llm_usage`` — the deterministic ordered evaluation that
guarantees safe degradation when LLM signals are absent, weak, or invalid.

Evaluation order (first failing condition wins):
  1. No completions → no_llm_data
  2. Low confidence → low_confidence
  3. Low stability → low_stability
  4. Low trust → low_trust
  5. All pass → accepted
"""

import pytest

from swarm import config
from swarm.utils.llm_fallback import evaluate_llm_usage


class TestNoLLMData:
    def test_no_completions_returns_false_no_llm_data(self) -> None:
        result = evaluate_llm_usage(
            has_completions=False,
            confidence=0.95,
            stability=0.95,
            trust=0.90,
        )
        assert result["use_llm"] is False
        assert result["reason"] == "no_llm_data"

    def test_no_completions_overrides_all_high_metrics(self) -> None:
        """Even with perfect confidence/stability/trust, no completions → reject."""
        result = evaluate_llm_usage(
            has_completions=False,
            confidence=1.0,
            stability=1.0,
            trust=1.0,
        )
        assert result["use_llm"] is False
        assert result["reason"] == "no_llm_data"


class TestLowConfidence:
    def test_confidence_below_threshold_returns_low_confidence(self) -> None:
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=0.5,
            stability=0.95,
            trust=0.475,
        )
        assert result["use_llm"] is False
        assert result["reason"] == "low_confidence"

    def test_confidence_below_threshold_takes_priority_over_stability(self) -> None:
        """Confidence check fires before stability check."""
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=0.5,
            stability=0.3,
            trust=0.15,
        )
        assert result["reason"] == "low_confidence"


class TestLowStability:
    def test_stability_below_threshold_returns_low_stability(self) -> None:
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=0.95,
            stability=0.4,
            trust=0.4,
        )
        assert result["use_llm"] is False
        assert result["reason"] == "low_stability"

    def test_stability_below_threshold_takes_priority_over_trust(self) -> None:
        """Stability check fires before trust check."""
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=0.95,
            stability=0.4,
            trust=0.4,
        )
        assert result["reason"] == "low_stability"


class TestLowTrust:
    def test_trust_below_threshold_returns_low_trust(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("swarm.utils.llm_fallback.TRUST_THRESHOLD", 0.8)
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=0.95,
            stability=0.80,
            trust=0.76,
        )
        assert result["use_llm"] is False
        assert result["reason"] == "low_trust"

    def test_trust_below_threshold_when_confidence_and_stability_pass(self) -> None:
        """Confidence and stability both above threshold, but trust (their product) is below."""
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=0.9,
            stability=0.75,
            trust=0.675,
        )
        assert result["use_llm"] is False
        assert result["reason"] == "low_trust"


class TestAccepted:
    def test_valid_case_returns_true_accepted(self) -> None:
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=0.95,
            stability=0.95,
            trust=0.90,
        )
        assert result["use_llm"] is True
        assert result["reason"] == "accepted"

    def test_all_at_threshold_returns_accepted(self) -> None:
        """All values exactly at threshold → accepted (boundary)."""
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=config.TRUST_THRESHOLD,
            stability=config.TRUST_THRESHOLD,
            trust=config.TRUST_THRESHOLD,
        )
        assert result["use_llm"] is True
        assert result["reason"] == "accepted"


class TestBoundaryCases:
    def test_confidence_at_threshold_is_not_rejected(self) -> None:
        """Confidence exactly at threshold → not low_confidence."""
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=config.TRUST_THRESHOLD,
            stability=1.0,
            trust=0.7,
        )
        assert result["reason"] != "low_confidence"

    def test_stability_at_threshold_is_not_rejected(self) -> None:
        """Stability exactly at threshold → not low_stability."""
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=1.0,
            stability=config.TRUST_THRESHOLD,
            trust=0.7,
        )
        assert result["reason"] != "low_stability"

    def test_trust_at_threshold_is_accepted(self) -> None:
        """Trust exactly at threshold → accepted."""
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=1.0,
            stability=1.0,
            trust=config.TRUST_THRESHOLD,
        )
        assert result["reason"] == "accepted"


class TestDeterminism:
    def test_same_inputs_produce_same_output(self) -> None:
        e1 = evaluate_llm_usage(
            has_completions=True,
            confidence=0.95,
            stability=0.85,
            trust=0.81,
        )
        e2 = evaluate_llm_usage(
            has_completions=True,
            confidence=0.95,
            stability=0.85,
            trust=0.81,
        )
        assert e1 == e2

    def test_output_has_required_keys(self) -> None:
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=0.95,
            stability=0.85,
            trust=0.81,
        )
        assert "use_llm" in result
        assert "reason" in result


class TestPriorityOrder:
    """Verify that the first failing condition wins."""

    def test_no_data_takes_priority_over_low_confidence(self) -> None:
        """No completions is checked first, even if confidence is also low."""
        result = evaluate_llm_usage(
            has_completions=False,
            confidence=0.5,
            stability=0.95,
            trust=0.475,
        )
        assert result["reason"] == "no_llm_data"

    def test_low_confidence_takes_priority_over_low_stability(self) -> None:
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=0.5,
            stability=0.3,
            trust=0.15,
        )
        assert result["reason"] == "low_confidence"

    def test_low_stability_takes_priority_over_low_trust(self) -> None:
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=0.95,
            stability=0.3,
            trust=0.285,
        )
        assert result["reason"] == "low_stability"


class TestEdgeCases:
    def test_zero_trust_with_completions_returns_low_confidence(self) -> None:
        """Completions exist but confidence is 0 → low_confidence."""
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=0.0,
            stability=0.0,
            trust=0.0,
        )
        assert result["reason"] == "low_confidence"

    def test_confidence_below_config_threshold_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When CONFIDENCE_THRESHOLD is raised, lower confidence is rejected."""
        monkeypatch.setattr("swarm.utils.llm_fallback.CONFIDENCE_THRESHOLD", 0.9)
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=0.8,
            stability=1.0,
            trust=0.8,
        )
        assert result["reason"] == "low_confidence"

    def test_stability_below_config_threshold_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When STABILITY_THRESHOLD is raised, lower stability is rejected."""
        monkeypatch.setattr("swarm.utils.llm_fallback.STABILITY_THRESHOLD", 0.9)
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=1.0,
            stability=0.8,
            trust=0.8,
        )
        assert result["reason"] == "low_stability"

    def test_trust_below_config_threshold_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When TRUST_THRESHOLD is raised, lower trust is rejected."""
        monkeypatch.setattr("swarm.utils.llm_fallback.TRUST_THRESHOLD", 0.9)
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=1.0,
            stability=1.0,
            trust=0.8,
        )
        assert result["reason"] == "low_trust"


class TestBehaviorChangesWithConfig:
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
        assert result["reason"] == "accepted"

    def test_raising_trust_threshold_rejects_previously_valid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("swarm.utils.llm_fallback.TRUST_THRESHOLD", 0.99)
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=1.0,
            stability=1.0,
            trust=0.95,
        )
        assert result["reason"] == "low_trust"

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
        assert result["reason"] == "low_stability"
