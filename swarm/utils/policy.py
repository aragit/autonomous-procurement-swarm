"""Deterministic policy constraints for LLM-adjusted strategy weights (v0.9 Step 8).

Applies hard business rules to the LLM-influenced (audited) weights:

    delivery (score_weight) >= 0.3
    price  (price_weight)   <= 0.7

These are policy guardrails — they constrain the *audited* adjustments, never
the canonical strategy selection. The function is pure: same input always
produces the same output, with no side effects or LLM calls.
"""

from __future__ import annotations

#: Minimum allowed weight for delivery (score_weight).
MIN_DELIVERY_WEIGHT = 0.3

#: Maximum allowed weight for price (price_weight).
MAX_PRICE_WEIGHT = 0.7


def apply_policy_constraints(weights: dict[str, float]) -> dict[str, float]:
    """Apply deterministic business constraints to price/delivery weights.

    Enforces:
      - ``delivery >= MIN_DELIVERY_WEIGHT`` (0.3)
      - ``price <= MAX_PRICE_WEIGHT`` (0.7)

    After clamping, the two weights are normalised to sum to 1.0 so the
    output is always a valid 2-field distribution.

    Args:
        weights: A dict with ``price`` and ``delivery`` keys (floats).

    Returns:
        A new dict ``{"price": float, "delivery": float}`` with constraints
        applied and normalised.
    """
    price = float(weights.get("price", 0.5))
    delivery = float(weights.get("delivery", 0.5))

    # Step A: enforce bounds
    price = min(price, MAX_PRICE_WEIGHT)
    delivery = max(delivery, MIN_DELIVERY_WEIGHT)

    # Step C: normalise (ensures price + delivery == 1.0)
    total = price + delivery
    if total > 0.0:
        price = price / total
        delivery = delivery / total
    else:
        # Degenerate fallback: equal split
        price = 0.5
        delivery = 0.5

    return {"price": round(price, 4), "delivery": round(delivery, 4)}
