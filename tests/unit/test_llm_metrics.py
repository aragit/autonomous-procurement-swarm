"""Tests for v0.9 Step 10: LLM metrics aggregation."""

from swarm.utils.llm_metrics import MAX_HISTORY, compute_llm_metrics


class TestComputeLLMMetrics:
    def test_empty_history(self) -> None:
        result = compute_llm_metrics([])
        assert result["acceptance_rate"] == 0.0
        assert result["avg_confidence"] == 0.0
        assert result["avg_stability"] == 0.0
        assert result["avg_trust"] == 0.0
        assert result["history_depth"] == 0
        assert result["max_history_depth"] == MAX_HISTORY

    def test_single_record(self) -> None:
        history = [
            {
                "confidence": 0.8,
                "stability": 0.9,
                "trust": 0.72,
                "decision_reason": "accepted",
            }
        ]
        result = compute_llm_metrics(history)
        assert result["history_depth"] == 1
        assert result["avg_confidence"] == 0.8
        assert result["avg_stability"] == 0.9
        assert result["avg_trust"] == 0.72

    def test_multiple_records_acceptance_rate(self) -> None:
        history = [
            {
                "confidence": 0.8,
                "stability": 0.9,
                "trust": 0.72,
                "decision_reason": "accepted",
            },
            {
                "confidence": 0.7,
                "stability": 0.8,
                "trust": 0.56,
                "decision_reason": "low_stability",
            },
            {
                "confidence": 0.6,
                "stability": 0.7,
                "trust": 0.42,
                "decision_reason": "accepted",
            },
        ]
        result = compute_llm_metrics(history)
        assert result["acceptance_rate"] == round(2 / 3, 4)
        assert result["avg_confidence"] == round((0.8 + 0.7 + 0.6) / 3, 4)
        assert result["avg_stability"] == round((0.9 + 0.8 + 0.7) / 3, 4)
        assert result["avg_trust"] == round((0.72 + 0.56 + 0.42) / 3, 4)
        assert result["history_depth"] == 3

    def test_all_rejected(self) -> None:
        history = [
            {
                "confidence": 0.5,
                "stability": 0.3,
                "trust": 0.15,
                "decision_reason": "low_confidence",
            },
            {
                "confidence": 0.4,
                "stability": 0.2,
                "trust": 0.08,
                "decision_reason": "low_stability",
            },
        ]
        result = compute_llm_metrics(history)
        assert result["acceptance_rate"] == 0.0

    def test_missing_fields_default_to_zero(self) -> None:
        history = [{"decision_reason": "accepted"}]
        result = compute_llm_metrics(history)
        assert result["avg_confidence"] == 0.0
        assert result["avg_stability"] == 0.0
        assert result["avg_trust"] == 0.0
