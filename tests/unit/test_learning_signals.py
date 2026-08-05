"""Tests for v1.0 Step 19: Learning Signals.

Tests ``compute_learning_signals`` in ``swarm.learning.signals``.
"""

import pytest

from swarm.learning.signals import compute_learning_signals


@pytest.fixture
def mock_history() -> list[dict]:
    return [
        {
            "round": 1,
            "confidence": 0.9,
            "stability": 0.85,
            "trust": 0.8,
            "decision_reason": "accepted",
            "payload": {},
            "num_completions": 3,
        },
        {
            "round": 2,
            "confidence": 0.88,
            "stability": 0.82,
            "trust": 0.78,
            "decision_reason": "accepted",
            "payload": {},
            "num_completions": 3,
        },
    ]


@pytest.fixture
def mock_feedback() -> dict:
    return {
        "outcome_score": 0.9,
        "success": True,
        "latency_ms": 1500.0,
        "user_feedback": "Good result",
    }


class TestLearningSignals:
    def test_returns_all_zero_signals_when_no_history(self) -> None:
        trace = {"llm_history": [], "feedback": None}
        signals = compute_learning_signals(trace)
        assert signals == {
            "confidence_vs_outcome_gap": 0.0,
            "stability_vs_success": 0.0,
            "trust_vs_outcome": 0.0,
            "drift_impact": 0.0,
        }

    def test_returns_all_zero_signals_when_no_feedback(self, mock_history) -> None:
        trace = {"llm_history": mock_history, "feedback": None}
        signals = compute_learning_signals(trace)
        assert signals == {
            "confidence_vs_outcome_gap": 0.0,
            "stability_vs_success": 0.0,
            "trust_vs_outcome": 0.0,
            "drift_impact": 0.0,
        }

    def test_high_confidence_low_outcome_shows_overconfidence(
        self, mock_history, mock_feedback
    ) -> None:
        # High confidence (~0.89) but outcome failure (0.0) → gap
        mock_feedback["success"] = False
        mock_feedback["outcome_score"] = 0.2
        trace = {"llm_history": mock_history, "feedback": mock_feedback}
        signals = compute_learning_signals(trace)
        # avg confidence ~ 0.89, outcome 0.0 → gap ~ 0.89
        assert signals["confidence_vs_outcome_gap"] > 0.5

    def test_stable_history_with_success_no_gap(self, mock_history, mock_feedback) -> None:
        mock_feedback["success"] = True
        mock_feedback["outcome_score"] = 0.95
        trace = {"llm_history": mock_history, "feedback": mock_feedback}
        signals = compute_learning_signals(trace)
        # avg confidence ~ 0.89, outcome 1.0 → no gap
        assert signals["confidence_vs_outcome_gap"] == 0.0

    def test_low_stability_failure_shows_instability(
        self, mock_history, mock_feedback
    ) -> None:
        # Set both stability values low so avg < 0.5
        mock_history[0]["stability"] = 0.3
        mock_history[1]["stability"] = 0.2
        mock_feedback["success"] = False
        mock_feedback["outcome_score"] = 0.1
        trace = {"llm_history": mock_history, "feedback": mock_feedback}
        signals = compute_learning_signals(trace)
        assert signals["stability_vs_success"] > 0.0

    def test_high_stability_success_no_instability(
        self, mock_history, mock_feedback
    ) -> None:
        mock_feedback["success"] = True
        mock_feedback["outcome_score"] = 0.9
        trace = {"llm_history": mock_history, "feedback": mock_feedback}
        signals = compute_learning_signals(trace)
        assert signals["stability_vs_success"] == 0.0

    def test_low_trust_with_outcome(self, mock_history, mock_feedback) -> None:
        # avg trust ~ 0.79, outcome failure → should have trust_vs_outcome > 0
        mock_feedback["success"] = False
        mock_feedback["outcome_score"] = 0.3
        mock_history[1]["trust"] = 0.4  # low trust
        trace = {"llm_history": mock_history, "feedback": mock_feedback}
        signals = compute_learning_signals(trace)
        assert signals["trust_vs_outcome"] > 0.0

    def test_drift_with_failure_shows_drift_impact(
        self, mock_history, mock_feedback
    ) -> None:
        # Confidence drop > 0.15 from round 1 to 2 triggers drift
        mock_history[1]["confidence"] = 0.5  # drop from 0.9 to 0.5 = 0.4
        mock_history[1]["stability"] = 0.4  # below threshold
        mock_history[1]["trust"] = 0.6  # below threshold
        mock_feedback["success"] = False
        mock_feedback["outcome_score"] = 0.2
        trace = {"llm_history": mock_history, "feedback": mock_feedback}
        signals = compute_learning_signals(trace)
        assert signals["drift_impact"] > 0.0

    def test_no_drift_with_failure_no_drift_impact(
        self, mock_history, mock_feedback
    ) -> None:
        # No drift conditions met
        mock_feedback["success"] = False
        mock_feedback["outcome_score"] = 0.3
        trace = {"llm_history": mock_history, "feedback": mock_feedback}
        signals = compute_learning_signals(trace)
        assert signals["drift_impact"] == 0.0

    def test_deterministic_output(self, mock_history, mock_feedback) -> None:
        trace = {"llm_history": mock_history, "feedback": mock_feedback}
        s1 = compute_learning_signals(trace)
        s2 = compute_learning_signals(trace)
        assert s1 == s2

    def test_signals_are_rounded(self, mock_history, mock_feedback) -> None:
        trace = {"llm_history": mock_history, "feedback": mock_feedback}
        signals = compute_learning_signals(trace)
        for _, val in signals.items():
            assert val == round(val, 4)
