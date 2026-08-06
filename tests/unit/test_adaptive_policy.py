"""Tests for v1.0 Step 20: Adaptive Policy (Auto-Tuning from Feedback).

Tests ``compute_adaptive_thresholds`` and ``get_adaptive_thresholds``
in ``swarm.learning.adaptive_policy``.
"""

from __future__ import annotations

from swarm.config import (
    CONFIDENCE_THRESHOLD,
    ENABLE_LEARNING,
    MIN_FEEDBACK_SAMPLES,
    STABILITY_THRESHOLD,
    TRUST_THRESHOLD,
)
from swarm.learning.adaptive_policy import (
    compute_adaptive_thresholds,
    get_adaptive_thresholds,
    override_thresholds,
)


def _make_feedback(
    success: bool,
    outcome_score: float,
    trust: float | None = None,
    trace_id: str = "T1",
    created_at: str = "2024-01-01T00:00:00",
) -> dict:
    fb = {
        "trace_id": trace_id,
        "outcome_score": outcome_score,
        "success": success,
        "latency_ms": 1000.0,
        "user_feedback": None,
        "created_at": created_at,
    }
    if trust is not None:
        fb["trust"] = trust
    return fb


def _make_many_feedback(n: int, success: bool, outcome_score: float) -> list[dict]:
    return [
        _make_feedback(
            success=success,
            outcome_score=outcome_score,
            trace_id=f"T{i}",
            created_at=f"2024-01-01T00:00:{i:02d}",
        )
        for i in range(n)
    ]


class TestComputeAdaptiveThresholds:
    def test_no_feedback_returns_defaults(self) -> None:
        result = compute_adaptive_thresholds([])
        assert result == {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "stability_threshold": STABILITY_THRESHOLD,
            "trust_threshold": TRUST_THRESHOLD,
        }

    def test_learning_disabled_returns_defaults(self) -> None:
        feedback = _make_many_feedback(10, success=True, outcome_score=0.9)
        original = ENABLE_LEARNING
        import swarm.config as cfg
        cfg.ENABLE_LEARNING = False
        try:
            result = compute_adaptive_thresholds(feedback)
        finally:
            cfg.ENABLE_LEARNING = original
        assert result == {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "stability_threshold": STABILITY_THRESHOLD,
            "trust_threshold": TRUST_THRESHOLD,
        }

    def test_below_min_samples_returns_defaults(self) -> None:
        feedback = _make_many_feedback(MIN_FEEDBACK_SAMPLES - 1, success=True, outcome_score=0.9)
        result = compute_adaptive_thresholds(feedback)
        assert result == {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "stability_threshold": STABILITY_THRESHOLD,
            "trust_threshold": TRUST_THRESHOLD,
        }

    def test_high_failure_rate_lowers_confidence(self) -> None:
        feedback = _make_many_feedback(10, success=False, outcome_score=0.2)
        result = compute_adaptive_thresholds(feedback)
        assert result["confidence_threshold"] < CONFIDENCE_THRESHOLD

    def test_low_failure_rate_does_not_lower_confidence(self) -> None:
        feedback = _make_many_feedback(10, success=True, outcome_score=0.9)
        result = compute_adaptive_thresholds(feedback)
        assert result["confidence_threshold"] == CONFIDENCE_THRESHOLD

    def test_high_failure_rate_increases_stability(self) -> None:
        feedback = _make_many_feedback(10, success=False, outcome_score=0.2)
        result = compute_adaptive_thresholds(feedback)
        assert result["stability_threshold"] > STABILITY_THRESHOLD

    def test_low_failure_rate_does_not_increase_stability(self) -> None:
        feedback = _make_many_feedback(10, success=True, outcome_score=0.9)
        result = compute_adaptive_thresholds(feedback)
        assert result["stability_threshold"] == STABILITY_THRESHOLD

    def test_high_trust_low_outcome_lowers_trust(self) -> None:
        feedback = [
            _make_feedback(
                success=False,
                outcome_score=0.3,
                trust=0.95,
                trace_id=f"T{i}",
                created_at=f"2024-01-01T00:00:{i:02d}",
            )
            for i in range(MIN_FEEDBACK_SAMPLES)
        ]
        result = compute_adaptive_thresholds(feedback)
        assert result["trust_threshold"] < TRUST_THRESHOLD

    def test_clamping_lower_bound(self) -> None:
        feedback = _make_many_feedback(20, success=False, outcome_score=0.0)
        result = compute_adaptive_thresholds(feedback)
        assert result["confidence_threshold"] >= 0.5
        assert result["stability_threshold"] >= 0.5
        assert result["trust_threshold"] >= 0.5

    def test_clamping_upper_bound(self) -> None:
        feedback = _make_many_feedback(20, success=True, outcome_score=0.9)
        result = compute_adaptive_thresholds(feedback)
        assert result["confidence_threshold"] <= 0.95
        assert result["stability_threshold"] <= 0.95
        assert result["trust_threshold"] <= 0.95

    def test_deterministic_output(self) -> None:
        feedback = _make_many_feedback(10, success=False, outcome_score=0.5)
        r1 = compute_adaptive_thresholds(feedback)
        r2 = compute_adaptive_thresholds(feedback)
        assert r1 == r2

    def test_output_keys(self) -> None:
        feedback = _make_many_feedback(10, success=True, outcome_score=0.9)
        result = compute_adaptive_thresholds(feedback)
        assert "confidence_threshold" in result
        assert "stability_threshold" in result
        assert "trust_threshold" in result

    def test_values_are_rounded(self) -> None:
        feedback = _make_many_feedback(10, success=False, outcome_score=0.3)
        result = compute_adaptive_thresholds(feedback)
        for v in result.values():
            assert v == round(v, 4)


class TestGetAdaptiveThresholds:
    """v1.1 Step 22 changed the runtime resolution to a strict priority:

    override_thresholds -> active_policy.thresholds -> config fallback.

    Feedback is no longer read at runtime (learning/execution separation).
    """

    def test_returns_defaults_when_no_db(self, monkeypatch) -> None:
        original = ENABLE_LEARNING
        import swarm.config as cfg
        cfg.ENABLE_LEARNING = False
        try:
            result = get_adaptive_thresholds()
        finally:
            cfg.ENABLE_LEARNING = original
        assert result == {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "stability_threshold": STABILITY_THRESHOLD,
            "trust_threshold": TRUST_THRESHOLD,
        }

    def test_no_feedback_in_db_returns_defaults(self, monkeypatch, tmp_path) -> None:
        import swarm.storage.event_store as es
        db_path = str(tmp_path / "empty_adapt.db")
        es._DB_PATH = db_path
        es.init_db(db_path)
        result = get_adaptive_thresholds()
        assert result == {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "stability_threshold": STABILITY_THRESHOLD,
            "trust_threshold": TRUST_THRESHOLD,
        }

    def test_feedback_does_not_affect_runtime_thresholds(self, monkeypatch, tmp_path) -> None:
        """Feedback is no longer read at runtime — runtime separation (Step 22)."""
        import swarm.storage.event_store as es
        from swarm.storage.event_store import store_feedback

        db_path = str(tmp_path / "adapt_test.db")
        es._DB_PATH = db_path
        es.init_db(db_path)

        for i in range(MIN_FEEDBACK_SAMPLES + 5):
            store_feedback(
                trace_id=f"T{i}",
                outcome_score=0.2,
                success=False,
                latency_ms=1000.0,
            )

        result = get_adaptive_thresholds()
        # No active policy -> config fallback, untouched by feedback.
        assert result == {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "stability_threshold": STABILITY_THRESHOLD,
            "trust_threshold": TRUST_THRESHOLD,
        }

    def test_active_policy_overrides_config(self, monkeypatch, tmp_path) -> None:
        """A promoted active policy supersedes the config fallback at runtime."""
        import swarm.storage.event_store as es
        db_path = str(tmp_path / "active_adapt.db")
        es._DB_PATH = db_path
        es.init_db(db_path)

        promoted = {
            "confidence_threshold": 0.75,
            "stability_threshold": 0.80,
            "trust_threshold": 0.85,
        }
        # Inject a promoted active policy directly into the store.
        es.store_policy(
            version="POL-ADTEST",
            signature="sig-ad-test",
            thresholds=promoted,
            weights={"price_weight": 0.4, "score_weight": 0.4, "carbon_weight": 0.2},
            active=True,
        )
        result = get_adaptive_thresholds()
        assert result == promoted

    def test_override_contextvar_beats_active_policy(self, monkeypatch, tmp_path) -> None:
        """The ContextVar override has the highest priority."""
        import swarm.storage.event_store as es
        db_path = str(tmp_path / "overlay_adapt.db")
        es._DB_PATH = db_path
        es.init_db(db_path)
        es.store_policy(
            version="POL-OVERLAY",
            signature="sig-overlay",
            thresholds={"confidence_threshold": 0.9, "stability_threshold": 0.9,
                        "trust_threshold": 0.9},
            weights={"price_weight": 0.5, "score_weight": 0.3, "carbon_weight": 0.2},
            active=True,
        )

        pinned = {"confidence_threshold": 0.61, "stability_threshold": 0.62,
                   "trust_threshold": 0.63}
        with override_thresholds(pinned):
            assert get_adaptive_thresholds() == pinned

    def test_deterministic_with_same_data(self, monkeypatch, tmp_path) -> None:
        import swarm.storage.event_store as es
        from swarm.storage.event_store import store_feedback

        db_path = str(tmp_path / "det_adapt.db")
        es._DB_PATH = db_path
        es.init_db(db_path)

        for i in range(MIN_FEEDBACK_SAMPLES + 3):
            store_feedback(
                trace_id=f"T{i}",
                outcome_score=0.5,
                success=True,
                latency_ms=1000.0,
            )

        r1 = get_adaptive_thresholds()
        r2 = get_adaptive_thresholds()
        assert r1 == r2
