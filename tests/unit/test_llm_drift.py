"""Tests for v0.9 Step 11: LLM drift detection."""

from swarm.utils.llm_drift import detect_drift


class TestDetectDrift:
    def test_empty_history(self) -> None:
        drift, reasons = detect_drift([])
        assert drift is False
        assert reasons == []

    def test_single_record_no_drift(self) -> None:
        history = [
            {
                "confidence": 0.9,
                "stability": 0.9,
                "trust": 0.81,
                "num_completions": 3,
                "decision_reason": "accepted",
            }
        ]
        drift, reasons = detect_drift(history)
        assert drift is False
        assert reasons == []

    def test_confidence_drop_triggers_drift(self) -> None:
        history = [
            {
                "confidence": 0.95,
                "stability": 0.9,
                "trust": 0.85,
                "num_completions": 3,
                "decision_reason": "accepted",
            },
            {
                "confidence": 0.75,
                "stability": 0.9,
                "trust": 0.67,
                "num_completions": 3,
                "decision_reason": "accepted",
            },
        ]
        drift, reasons = detect_drift(history)
        assert drift is True
        assert any("confidence_drop" in r for r in reasons)

    def test_stability_below_threshold(self) -> None:
        history = [
            {
                "confidence": 0.9,
                "stability": 0.3,
                "trust": 0.27,
                "num_completions": 3,
                "decision_reason": "accepted",
            }
        ]
        drift, reasons = detect_drift(history)
        assert drift is True
        assert "stability_below_threshold" in reasons

    def test_trust_below_threshold(self) -> None:
        history = [
            {
                "confidence": 0.9,
                "stability": 0.9,
                "trust": 0.1,
                "num_completions": 3,
                "decision_reason": "accepted",
            }
        ]
        drift, reasons = detect_drift(history)
        assert drift is True
        assert "trust_below_threshold" in reasons

    def test_insufficient_completions(self) -> None:
        history = [
            {
                "confidence": 0.9,
                "stability": 0.9,
                "trust": 0.81,
                "num_completions": 1,
                "decision_reason": "accepted",
            }
        ]
        drift, reasons = detect_drift(history)
        assert drift is True
        assert "insufficient_completions" in reasons

    def test_no_drift_with_healthy_metrics(self) -> None:
        history = [
            {
                "confidence": 0.9,
                "stability": 0.95,
                "trust": 0.85,
                "num_completions": 3,
                "decision_reason": "accepted",
            },
            {
                "confidence": 0.88,
                "stability": 0.92,
                "trust": 0.81,
                "num_completions": 3,
                "decision_reason": "accepted",
            },
        ]
        drift, reasons = detect_drift(history)
        assert drift is False
        assert reasons == []
