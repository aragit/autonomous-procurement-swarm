"""Tests for v0.9 Step 12: aggregated explainability."""

from swarm.utils.llm_explain import aggregate_explanations


class TestAggregateExplanations:
    def test_empty_history(self) -> None:
        result = aggregate_explanations([])
        assert result["rounds"] == []
        assert result["aggregate"]["total_rounds"] == 0
        assert result["drift"]["detected"] is False

    def test_single_record(self) -> None:
        history = [
            {
                "confidence": 0.8,
                "stability": 0.9,
                "trust": 0.72,
                "round": 1,
                "decision_reason": "accepted",
            }
        ]
        result = aggregate_explanations(history)
        assert result["aggregate"]["total_rounds"] == 1
        assert result["aggregate"]["avg_confidence"] == 0.8
        assert result["aggregate"]["avg_stability"] == 0.9
        assert result["aggregate"]["avg_trust"] == 0.72
        assert len(result["rounds"]) == 1

    def test_drift_detected(self) -> None:
        history = [
            {
                "confidence": 0.95,
                "stability": 0.9,
                "trust": 0.855,
                "round": 1,
                "num_completions": 3,
                "decision_reason": "accepted",
            },
            {
                "confidence": 0.70,
                "stability": 0.9,
                "trust": 0.63,
                "round": 2,
                "num_completions": 3,
                "decision_reason": "accepted",
            },
        ]
        result = aggregate_explanations(history)
        assert result["drift"]["detected"] is True

    def test_current_decision_included(self) -> None:
        history = [
            {"confidence": 0.8, "stability": 0.9, "trust": 0.72, "round": 1},
        ]
        current = {"decision": "accepted", "summary": "ok"}
        result = aggregate_explanations(history, current_decision=current)
        assert result["current"] == current

    def test_summary_string_deterministic(self) -> None:
        history = [
            {"confidence": 0.8, "stability": 0.9, "trust": 0.72, "round": 1},
        ]
        result1 = aggregate_explanations(history)
        result2 = aggregate_explanations(history)
        assert result1["summary"] == result2["summary"]
