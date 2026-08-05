"""Unit tests for v0.9 Step 8: Deterministic Policy Constraints.

Tests ``apply_policy_constraints`` — the pure function that enforces hard
business rules (delivery >= POLICY_DELIVERY_MIN, price <= POLICY_PRICE_MAX)
on LLM-adjusted strategy weights.
"""

import pytest

from swarm import config
from swarm.utils.policy import apply_policy_constraints

# --- Delivery below minimum → corrected ---


def test_delivery_below_minimum_corrected_to_minimum() -> None:
    """Delivery weight below POLICY_DELIVERY_MIN is raised to POLICY_DELIVERY_MIN."""
    result = apply_policy_constraints({"price": 0.9, "delivery": 0.1})
    assert result["delivery"] >= config.POLICY_DELIVERY_MIN


def test_delivery_below_minimum_raises_after_normalization() -> None:
    """After clamping delivery to POLICY_DELIVERY_MIN and price to
    POLICY_PRICE_MAX, normalisation keeps delivery >= POLICY_DELIVERY_MIN."""
    result = apply_policy_constraints({"price": 0.95, "delivery": 0.05})
    # price clamped to POLICY_PRICE_MAX, delivery clamped to POLICY_DELIVERY_MIN
    assert result["delivery"] == config.POLICY_DELIVERY_MIN
    assert result["price"] == config.POLICY_PRICE_MAX


# --- Price above maximum → corrected ---


def test_price_above_maximum_corrected_to_maximum() -> None:
    """Price weight above POLICY_PRICE_MAX is lowered to POLICY_PRICE_MAX."""
    result = apply_policy_constraints({"price": 0.9, "delivery": 0.1})
    assert result["price"] <= config.POLICY_PRICE_MAX


def test_price_above_maximum_raises_after_normalization() -> None:
    """After clamping, price never exceeds POLICY_PRICE_MAX."""
    result = apply_policy_constraints({"price": 0.95, "delivery": 0.05})
    assert result["price"] == config.POLICY_PRICE_MAX
    assert result["delivery"] == config.POLICY_DELIVERY_MIN


# --- Normalization preserves sum == 1 ---


def test_normalization_keeps_sum_as_one() -> None:
    """After policy + normalisation, price + delivery == 1.0."""
    result = apply_policy_constraints({"price": 0.5, "delivery": 0.5})
    total = result["price"] + result["delivery"]
    assert abs(total - 1.0) < 1e-9


def test_normalization_keeps_sum_as_one_when_clamped() -> None:
    """Clamping + normalisation preserves sum == 1.0."""
    result = apply_policy_constraints({"price": 0.8, "delivery": 0.2})
    total = result["price"] + result["delivery"]
    assert abs(total - 1.0) < 1e-9


# --- Already valid weights unchanged ---


def test_already_valid_weights_unchanged() -> None:
    """Weights within bounds and summing to 1.0 after normalisation are unchanged."""
    # price=0.5, delivery=0.5 → both within bounds → normalisation: 0.5/1.0 = 0.5
    result = apply_policy_constraints({"price": 0.5, "delivery": 0.5})
    assert result["price"] == 0.5
    assert result["delivery"] == 0.5


def test_already_valid_weights_unchanged_boundary() -> None:
    """Weights at the exact boundary values are valid."""
    # price = POLICY_PRICE_MAX (max, OK), delivery = POLICY_DELIVERY_MIN (min, OK) → sum = 1.0
    result = apply_policy_constraints(
        {"price": config.POLICY_PRICE_MAX, "delivery": config.POLICY_DELIVERY_MIN}
    )
    assert result["price"] == config.POLICY_PRICE_MAX
    assert result["delivery"] == config.POLICY_DELIVERY_MIN


# --- Deterministic output ---


def test_output_is_deterministic() -> None:
    """Same input always produces the same output."""
    input_weights = {"price": 0.6, "delivery": 0.4}
    result1 = apply_policy_constraints(input_weights)
    result2 = apply_policy_constraints(input_weights)
    assert result1 == result2


def test_output_is_a_new_dict() -> None:
    """The function does not mutate the input dict."""
    input_weights = {"price": 0.6, "delivery": 0.4}
    original = dict(input_weights)
    apply_policy_constraints(input_weights)
    assert input_weights == original


# --- Edge cases ---


def test_both_constraints_triggered() -> None:
    """When price > max AND delivery < min simultaneously."""
    # price=0.95 → clamped to POLICY_PRICE_MAX, delivery=0.05 → clamped to POLICY_DELIVERY_MIN
    # total = 1.0, no further normalisation needed
    result = apply_policy_constraints({"price": 0.95, "delivery": 0.05})
    assert result["price"] == config.POLICY_PRICE_MAX
    assert result["delivery"] == config.POLICY_DELIVERY_MIN
    assert abs(result["price"] + result["delivery"] - 1.0) < 1e-9


def test_both_constraints_triggered_with_normalization() -> None:
    """Both constraints clamped, then normalised if sum != 1.0."""
    # price=0.8 → clamped to POLICY_PRICE_MAX, delivery=0.1 → clamped to POLICY_DELIVERY_MIN
    # sum = 1.0, already normalised
    result = apply_policy_constraints({"price": 0.8, "delivery": 0.1})
    assert result["price"] == config.POLICY_PRICE_MAX
    assert result["delivery"] == config.POLICY_DELIVERY_MIN


def test_missing_keys_use_defaults() -> None:
    """Missing keys default to 0.5."""
    result = apply_policy_constraints({})
    # price=0.5 (within bounds), delivery=0.5 (within bounds)
    assert result["price"] == 0.5
    assert result["delivery"] == 0.5


def test_partial_keys_uses_default_for_missing() -> None:
    """Only price provided → delivery defaults to 0.5."""
    result = apply_policy_constraints({"price": 0.2})
    # price=0.2, delivery=0.5 → within bounds → normalise to 0.2/0.7, 0.5/0.7
    assert result["price"] < 0.3  # normalised down
    assert result["delivery"] > 0.7  # normalised up


def test_output_rounded_to_four_decimals() -> None:
    """Output values are rounded to 4 decimal places."""
    result = apply_policy_constraints({"price": 0.123456, "delivery": 0.876544})
    # price=0.123456 (within bounds), delivery=0.876544 (within bounds)
    total = 0.123456 + 0.876544
    expected_price = 0.123456 / total
    expected_delivery = 0.876544 / total
    assert result["price"] == round(expected_price, 4)
    assert result["delivery"] == round(expected_delivery, 4)


def test_output_has_only_price_and_delivery_keys() -> None:
    """Output contains only 'price' and 'delivery' keys."""
    result = apply_policy_constraints({"price": 0.5, "delivery": 0.5})
    assert set(result.keys()) == {"price", "delivery"}


class TestPolicyBoundsConfigurable:
    def test_raising_price_max_lowers_clamped_price(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("swarm.utils.policy.POLICY_PRICE_MAX", 0.5)
        result = apply_policy_constraints({"price": 0.9, "delivery": 0.1})
        # price clamped to 0.5, delivery clamped to 0.3 → normalized: 0.5/0.8=0.625
        assert result["price"] == 0.625

    def test_lowering_price_max_allows_higher_price(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("swarm.utils.policy.POLICY_PRICE_MAX", 0.9)
        result = apply_policy_constraints({"price": 0.85, "delivery": 0.15})
        # price within bounds, delivery clamped to 0.3 → normalized: 0.85/1.15≈0.739
        assert result["price"] == pytest.approx(0.7391, abs=0.001)

    def test_raising_delivery_min_raises_clamped_delivery(
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
