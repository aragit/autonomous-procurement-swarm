"""Bounded LLM adjustment validation (v0.9 Step 4).

This is the gatekeeper between LLM *suggestions* and deterministic *action*.
The function :func:`validate_strategy_adjustments` enforces strict bounds on
any weight deltas proposed by an LLM, rejecting excessive or unknown fields
so the deterministic system always stays in control.

Design principles:
- **Bounded**: every accepted delta is clamped to [-0.1, +0.1].
- **Typed**: non-numeric values are silently coerced to 0 (ignored).
- **Known fields only**: unknown keys are never accepted.
- **Pure**: this function has no side effects — it only transforms input.
"""

from __future__ import annotations

from typing import Any

#: Maximum absolute delta the LLM may propose for any single weight.
DELTA_BOUND = 0.1

#: The only adjustment fields the system recognizes.
#: ``delivery_weight_delta`` maps to ``score_weight`` (delivery time is a
#: component of the evaluation score).
_ALLOWED_ADJUSTMENT_FIELDS = frozenset(
    {"price_weight_delta", "delivery_weight_delta"}
)


def validate_strategy_adjustments(adjustments: dict[str, Any]) -> dict[str, float]:
    """Sanitize and bound an LLM's proposed strategy weight adjustments.

    Accepts a dict that may contain ``price_weight_delta`` and/or
    ``delivery_weight_delta``. Returns a dict containing only the validated
    entries — every accepted delta is within ``[-0.1, +0.1]``, unknown keys
    are dropped, and non-numeric / out-of-bound values are silently ignored.

    Unlike clamping, out-of-bound values are **rejected entirely** rather than
    coerced — this ensures excessive LLM suggestions never influence the
    deterministic path.

    Examples::

        {"price_weight_delta": -0.05}          # → {"price_weight_delta": -0.05}
        {"price_weight_delta": -0.2}            # → {}  (exceeds bound, rejected)
        {"unknown_field": 0.1}                  # → {}  (unknown key)
        {"price_weight_delta": "fast"}          # → {}  (non-numeric)
        {}                                      # → {}
    """
    if not isinstance(adjustments, dict):
        return {}

    result: dict[str, float] = {}
    for key, value in adjustments.items():
        if key not in _ALLOWED_ADJUSTMENT_FIELDS:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        # Reject (not clamp) values outside the bound
        if -DELTA_BOUND <= numeric <= DELTA_BOUND and abs(numeric) > 1e-12:
            result[key] = numeric
    return result
