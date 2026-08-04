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

from swarm.utils.llm_fallback import evaluate_llm_usage
from swarm.utils.llm_stability import TRUST_THRESHOLD


class TestNoLLMData:
    def test_no_completions_returns_false_no_llm_data(self) -> None:
        result = evaluate_llm_usage(
            has_completions=False,
            confidence=0.95,
            stability=0.95,
            trust=0.90,
            threshold=TRUST_THRESHOLD,
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
            threshold=TRUST_THRESHOLD,
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
            threshold=TRUST_THRESHOLD,
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
            threshold=TRUST_THRESHOLD,
        )
        assert result["reason"] == "low_confidence"


class TestLowStability:
    def test_stability_below_threshold_returns_low_stability(self) -> None:
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=0.95,
            stability=0.5,
            trust=0.475,
            threshold=TRUST_THRESHOLD,
        )
        assert result["use_llm"] is False
        assert result["reason"] == "low_stability"

    def test_stability_below_threshold_takes_priority_over_trust(self) -> None:
        """Stability check fires before trust check."""
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=0.95,
            stability=0.5,
            trust=0.4,
            threshold=TRUST_THRESHOLD,
        )
        assert result["reason"] == "low_stability"


class TestLowTrust:
    def test_trust_below_threshold_returns_low_trust(self) -> None:
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=0.95,
            stability=0.80,
            trust=0.76,
            threshold=0.8,
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
            threshold=TRUST_THRESHOLD,
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
            threshold=TRUST_THRESHOLD,
        )
        assert result["use_llm"] is True
        assert result["reason"] == "accepted"

    def test_all_at_threshold_returns_accepted(self) -> None:
        """All values exactly at threshold → accepted (boundary)."""
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=TRUST_THRESHOLD,
            stability=TRUST_THRESHOLD,
            trust=TRUST_THRESHOLD,
            threshold=TRUST_THRESHOLD,
        )
        assert result["use_llm"] is True
        assert result["reason"] == "accepted"


class TestBoundaryCases:
    def test_confidence_at_threshold_is_not_rejected(self) -> None:
        """Confidence exactly at threshold → not low_confidence."""
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=TRUST_THRESHOLD,
            stability=1.0,
            trust=0.7,
            threshold=TRUST_THRESHOLD,
        )
        assert result["reason"] != "low_confidence"

    def test_stability_at_threshold_is_not_rejected(self) -> None:
        """Stability exactly at threshold → not low_stability."""
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=1.0,
            stability=TRUST_THRESHOLD,
            trust=0.7,
            threshold=TRUST_THRESHOLD,
        )
        assert result["reason"] != "low_stability"

    def test_trust_at_threshold_is_accepted(self) -> None:
        """Trust exactly at threshold → accepted."""
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=1.0,
            stability=1.0,
            trust=TRUST_THRESHOLD,
            threshold=TRUST_THRESHOLD,
        )
        assert result["reason"] == "accepted"


class TestDeterminism:
    def test_same_inputs_produce_same_output(self) -> None:
        e1 = evaluate_llm_usage(
            has_completions=True,
            confidence=0.95,
            stability=0.85,
            trust=0.81,
            threshold=TRUST_THRESHOLD,
        )
        e2 = evaluate_llm_usage(
            has_completions=True,
            confidence=0.95,
            stability=0.85,
            trust=0.81,
            threshold=TRUST_THRESHOLD,
        )
        assert e1 == e2

    def test_output_has_required_keys(self) -> None:
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=0.95,
            stability=0.85,
            trust=0.81,
            threshold=TRUST_THRESHOLD,
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
            threshold=TRUST_THRESHOLD,
        )
        assert result["reason"] == "no_llm_data"

    def test_low_confidence_takes_priority_over_low_stability(self) -> None:
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=0.5,
            stability=0.3,
            trust=0.15,
            threshold=TRUST_THRESHOLD,
        )
        assert result["reason"] == "low_confidence"

    def test_low_stability_takes_priority_over_low_trust(self) -> None:
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=0.95,
            stability=0.3,
            trust=0.285,
            threshold=TRUST_THRESHOLD,
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
            threshold=TRUST_THRESHOLD,
        )
        assert result["reason"] == "low_confidence"

    def test_negative_threshold_all_rejected(self) -> None:
        """With threshold=0, only trust=0 fails (0 >= 0 is True)."""
        # confidence=0, threshold=0 → 0 < 0 is False, so not low_confidence
        # stability=0, threshold=0 → 0 < 0 is False, so not low_stability
        # trust=0, threshold=0 → 0 < 0 is False, so not low_trust
        # → accepted
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=0.0,
            stability=0.0,
            trust=0.0,
            threshold=0.0,
        )
        assert result["use_llm"] is True
        assert result["reason"] == "accepted"

    def test_custom_threshold_respected(self) -> None:
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=0.8,
            stability=0.9,
            trust=0.72,
            threshold=0.8,
        )
        assert result["use_llm"] is False
        assert result["reason"] == "low_trust"

    def test_higher_custom_threshold_accepted(self) -> None:
        result = evaluate_llm_usage(
            has_completions=True,
            confidence=1.0,
            stability=1.0,
            trust=1.0,
            threshold=0.5,
        )
        assert result["use_llm"] is True
        assert result["reason"] == "accepted"
