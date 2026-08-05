"""Tests for v0.9 Step 11: LLM drift detection."""

import pytest

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

    def test_confidence_drop_below_threshold_no_confidence_drop_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When CONFIDENCE_DROP_THRESHOLD is raised above the actual drop,
        no confidence_drop reason."""
        monkeypatch.setattr("swarm.utils.llm_drift.CONFIDENCE_DROP_THRESHOLD", 0.3)
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
                "trust": 0.85,
                "num_completions": 3,
                "decision_reason": "accepted",
            },
        ]
        drift, reasons = detect_drift(history)
        assert not any("confidence_drop" in r for r in reasons)

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

    def test_confidence_drop_exactly_at_threshold_is_drift(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A drop equal to or exceeding CONFIDENCE_DROP_THRESHOLD triggers drift."""
        monkeypatch.setattr("swarm.utils.llm_drift.CONFIDENCE_DROP_THRESHOLD", 0.19)
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

    def test_confidence_drop_just_below_threshold_no_drift(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A drop just below CONFIDENCE_DROP_THRESHOLD does not trigger drift."""
        monkeypatch.setattr("swarm.utils.llm_drift.CONFIDENCE_DROP_THRESHOLD", 0.21)
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
        assert not any("confidence_drop" in r for r in reasons)
