"""Unit tests for v0.9 Step 6: Temporal Stability Metric.

Tests the ``compute_temporal_stability`` function that measures drift in LLM
adjustment suggestions across consensus rounds.
"""

from typing import Any

import pytest

from swarm.utils.llm_stability import (
    TOLERANCE,
    compute_temporal_stability,
)


def _history_round(
    round_num: int,
    price: float,
    delivery: float,
) -> dict[str, Any]:
    return {
        "round": round_num,
        "aggregated_adjustments": {
            "price_weight_delta": price,
            "delivery_weight_delta": delivery,
        },
    }


# --- Core stability calculation ---


def test_stability_empty_history_is_zero() -> None:
    assert compute_temporal_stability([]) == 0.0


def test_stability_single_record_is_zero() -> None:
    history = [_history_round(1, -0.05, 0.05)]
    assert compute_temporal_stability(history) == 0.0


def test_stability_identical_adjustments_is_one() -> None:
    """Identical adjustments across rounds → perfect stability."""
    history = [
        _history_round(1, -0.05, 0.05),
        _history_round(2, -0.05, 0.05),
        _history_round(3, -0.05, 0.05),
    ]
    assert compute_temporal_stability(history) == 1.0


def test_stability_within_tolerance_is_high() -> None:
    """Small drift within tolerance band → high stability."""
    history = [
        _history_round(1, -0.05, 0.05),
        _history_round(2, -0.05 + TOLERANCE * 0.5, 0.05 - TOLERANCE * 0.5),
    ]
    assert compute_temporal_stability(history) == 1.0


def test_stability_full_range_drift_is_zero() -> None:
    """Maximum drift (±0.1) → zero stability."""
    history = [
        _history_round(1, -0.1, 0.1),
        _history_round(2, 0.1, -0.1),
    ]
    assert compute_temporal_stability(history) == 0.0


def test_stability_partial_drift_is_partial() -> None:
    """Moderate drift → intermediate stability."""
    # Round 1: price -0.05, Round 2: price 0.05
    # abs_diff = 0.1, drift = (0.1 - 0.02) / (0.2 - 0.02) = 0.08/0.18 ≈ 0.444
    # stability = 1 - 0.444 ≈ 0.556
    history = [
        _history_round(1, -0.05, 0.05),
        _history_round(2, 0.05, -0.05),
    ]
    result = compute_temporal_stability(history)
    assert 0.4 < result < 0.7


# --- Multiple fields averaged ---


def test_stability_averages_across_fields() -> None:
    """Drift in one field but stability in another → averaged."""
    # price drift: -0.05 → 0.05 (abs_diff=0.1, drift≈0.444)
    # delivery: 0.05 → 0.05 (no drift)
    # mean drift = (0.444 + 0.0) / 2 = 0.222
    # stability = 1 - 0.222 ≈ 0.778
    history = [
        _history_round(1, -0.05, 0.05),
        _history_round(2, 0.05, 0.05),
    ]
    result = compute_temporal_stability(history)
    assert 0.7 < result < 0.9


# --- Multiple consecutive pairs ---


def test_stability_considers_all_consecutive_pairs() -> None:
    """3 rounds → 2 consecutive pairs, both averaged."""
    history = [
        _history_round(1, -0.05, 0.05),
        _history_round(2, -0.05, 0.05),  # stable: 1→2
        _history_round(3, 0.10, -0.10),  # divergent: 2→3
    ]
    result = compute_temporal_stability(history)
    # Pair 1-2: drift=0, Pair 2-3: drift = (0.15 - 0.02) / 0.18 ≈ 0.722
    # mean drift = (0 + 0.722) / 2 ≈ 0.361
    # stability ≈ 0.639
    assert 0.5 < result < 0.8


# --- Edge cases ---


def test_stability_sorts_by_round() -> None:
    """History records in any order → sorted by round before comparison."""
    history = [
        _history_round(3, -0.05, 0.05),
        _history_round(1, -0.05, 0.05),
        _history_round(2, -0.05, 0.05),
    ]
    result = compute_temporal_stability(history)
    assert result == 1.0


def test_stability_missing_fields_skipped() -> None:
    """Records without aggregated_adjustments don't break computation."""
    history = [
        {"round": 1, "aggregated_adjustments": {"price_weight_delta": -0.05}},
        {"round": 2, "aggregated_adjustments": {"price_weight_delta": -0.05}},
    ]
    result = compute_temporal_stability(history)
    assert result == 1.0  # only price field, no drift


def test_stability_unknown_fields_ignored() -> None:
    """Fields outside the recognised set are not considered."""
    history = [
        {
            "round": 1,
            "aggregated_adjustments": {
                "price_weight_delta": -0.05,
                "unknown_field": 0.5,
            },
        },
        {
            "round": 2,
            "aggregated_adjustments": {
                "price_weight_delta": 0.05,
                "unknown_field": -0.5,
            },
        },
    ]
    result = compute_temporal_stability(history)
    # Only price_weight_delta is considered: drift = (0.1 - 0.02) / 0.18 ≈ 0.444
    # stability ≈ 0.556
    assert 0.4 < result < 0.7


def test_stability_non_numeric_values_skipped() -> None:
    """Non-numeric adjustment values don't break computation."""
    history = [
        {
            "round": 1,
            "aggregated_adjustments": {"price_weight_delta": -0.05},
        },
        {
            "round": 2,
            "aggregated_adjustments": {"price_weight_delta": "fast"},
        },
    ]
    # Non-numeric value is skipped, no valid pairs for price → no drifts
    result = compute_temporal_stability(history)
    assert result == 0.0


def test_stability_value_is_rounded() -> None:
    """Stability is rounded to 4 decimal places."""
    # Craft values that produce a non-terminating decimal
    history = [
        _history_round(1, -0.05, 0.05),
        _history_round(2, 0.03, -0.03),
    ]
    result = compute_temporal_stability(history)
    # Price drift: abs_diff = 0.08, drift = (0.08 - 0.02) / 0.18 = 0.3333...
    # Delivery drift: abs_diff = 0.08, drift = 0.3333...
    # mean drift = 0.3333..., stability = 0.6667
    assert result == pytest.approx(0.6667, abs=0.001)


class TestStabilityToleranceConfigurable:
    def test_raising_tolerance_reduces_drift_penalty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With a larger tolerance, the same drift produces less penalty."""
        monkeypatch.setattr("swarm.utils.llm_stability.TOLERANCE", 0.08)
        history = [
            _history_round(1, -0.05, 0.05),
            _history_round(2, 0.05, -0.05),
        ]
        result = compute_temporal_stability(history)
        # With tolerance=0.08: abs_diff=0.10, drift=(0.10-0.08)/(0.2-0.08)=0.02/0.12≈0.167
        # stability≈0.833
        assert result > 0.8

    def test_lowering_tolerance_increases_drift_penalty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With a smaller tolerance, the same drift produces more penalty."""
        monkeypatch.setattr("swarm.utils.llm_stability.TOLERANCE", 0.0)
        history = [
            _history_round(1, -0.05, 0.05),
            _history_round(2, 0.05, -0.05),
        ]
        result = compute_temporal_stability(history)
        # With tolerance=0.0: abs_diff=0.10, drift=0.10/0.2=0.5
        # stability=0.5
        assert result < 0.6
