"""Deterministic routing engine: context → strategy + param overrides (v1.1 Step 24+25).

The routing engine evaluates a policy's routing rules deterministically using
priority-based conflict resolution (highest priority wins; ties broken by
condition count, then rule_id) and falls back to the policy's default strategy.
No randomness, no dynamic rule creation.

v1.1 Step 25: rules can also specify ``param_overrides`` — deterministic deltas
applied to thresholds/weights when the rule matches.
v1.1 Step 25 Hardened: idempotency guard, immutability, deterministic ordering,
context normalization, trace observability, strict validation, candidate capping.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from swarm.learning.policy import Policy

from swarm.config import (
    CANDIDATE_SAMPLING_SEED,
    MAX_CANDIDATES,
    MAX_PARAMS_PER_RULE,
    MAX_ROUTING_RULES,
    PARAM_OVERRIDE_DELTA,
    PARAM_OVERRIDE_TOTAL_DELTA_CAP,
    ROUTING_CONDITION_KEYS,
    STRATEGY_TYPES,
)
from swarm.learning.context import extract_context

#: Predefined condition space for routing rule generation.
#: Each entry is (context_key, context_value).
ROUTING_CONDITIONS: list[tuple[str, Any]] = [
    ("budget_level", "low"),
    ("budget_level", "high"),
    ("urgency", "high"),
    ("supplier_count", "<3"),
]

#: Predefined parameter override space for routing rule generation (Step 25).
#: Each entry is (param_key, delta_value). Deltas are small and bounded.
PARAM_OVERRIDE_SPACE: list[tuple[str, float]] = [
    ("price_weight", -0.10),
    ("price_weight", +0.10),
    ("score_weight", -0.10),
    ("score_weight", +0.10),
    ("carbon_weight", -0.10),
    ("carbon_weight", +0.10),
    ("confidence_threshold", -0.05),
    ("confidence_threshold", +0.05),
    ("stability_threshold", -0.05),
    ("stability_threshold", +0.05),
    ("trust_threshold", -0.05),
    ("trust_threshold", +0.05),
]

#: Keys that can be overridden (thresholds + weight keys).
OVERRIDABLE_THRESHOLD_KEYS: tuple[str, ...] = (
    "confidence_threshold",
    "stability_threshold",
    "trust_threshold",
)
OVERRIDABLE_WEIGHT_KEYS: tuple[str, ...] = (
    "price_weight",
    "score_weight",
    "carbon_weight",
)


@dataclass
class RoutingDecision:
    """The result of a routing decision: strategy type + param overrides."""

    strategy_type: str
    param_overrides: dict[str, float]
    applied_overrides: dict[str, float]


def select_strategy(context: dict[str, Any], strategy: dict[str, Any]) -> str:
    """Select a strategy type given a context and a strategy dict.

    The strategy dict can be either:

    **Single strategy (Step 23 style)**::

        {"type": "balanced"}  — "type" IS the strategy type

    **Routing strategy (Step 24 style)**::

        {"type": "routing", "rules": [...], "default": "balanced"}

    For single strategies, the strategy type is returned directly.
    For routing strategies, rules are evaluated in order (first match wins),
    falling back to ``default``.

    Args:
        context: The extracted context dict from :func:`extract_context`.
        strategy: The strategy dict from the policy.

    Returns:
        A strategy type string from ``STRATEGY_TYPES``.

    Raises:
        ValueError: If the strategy is not a dict.
    """
    decision = select_strategy_with_params(context, strategy)
    return decision.strategy_type


def select_strategy_with_params(
    context: dict[str, Any], strategy: dict[str, Any]
) -> RoutingDecision:
    """Select a strategy type + param overrides given context and strategy.

    For single strategies, returns the strategy type with empty overrides.
    For routing strategies, returns the matched rule's strategy + param_overrides,
    or the default with empty overrides.

    Args:
        context: The extracted context dict from :func:`extract_context`.
        strategy: The strategy dict from the policy.

    Returns:
        A :class:`RoutingDecision` with strategy_type and param_overrides.
    """
    if not isinstance(strategy, dict):
        raise ValueError(f"strategy must be a dict, got {type(strategy)}")

    strat_type = str(strategy.get("type"))

    # Single strategy (Step 23): type IS the strategy name directly.
    if strat_type in STRATEGY_TYPES:
        return RoutingDecision(
            strategy_type=strat_type,
            param_overrides={},
            applied_overrides={},
        )

    # Routing strategy (Step 24/25).
    if strat_type != "routing":
        raise ValueError(f"Unknown strategy type: {strat_type}")

    rules = strategy.get("rules", [])
    default = str(strategy.get("default", "balanced"))

    if default not in STRATEGY_TYPES:
        default = "balanced"

    normalized = normalize_context(context)

    # Sort rules deterministically: priority DESC, then rule_id ASC.
    sorted_rules = _sort_rules(rules)

    # Collect all matching rules, then pick the best one.
    matching: list[tuple[dict[str, Any], int, str]] = []
    for idx, rule in enumerate(sorted_rules):
        conditions = rule.get("conditions", {})
        if all(_matches(normalized, key, val) for key, val in conditions.items()):
            priority = int(rule.get("priority", 0))
            rule_id = str(rule.get("rule_id", f"rule_{idx}"))
            matching.append((rule, priority, rule_id))

    if matching:
        # Tie-break: rule_id ASC (stable), condition_count ASC (more general),
        # priority DESC (highest priority wins).
        matching.sort(key=lambda m: m[2])  # rule_id ASC
        matching.sort(key=lambda m: len(m[0].get("conditions", {})))  # fewer conditions
        matching.sort(key=lambda m: -m[1])  # priority DESC

        rule, _, _ = matching[0]
        strat = str(rule.get("strategy", default))
        if strat not in STRATEGY_TYPES:
            strat = default
        overrides = rule.get("param_overrides", {}) or {}
        overrides_copy = dict(overrides)
        return RoutingDecision(
            strategy_type=strat,
            param_overrides=overrides_copy,
            applied_overrides=overrides_copy,
        )

    return RoutingDecision(
        strategy_type=default,
        param_overrides={},
        applied_overrides={},
    )


def apply_param_overrides(
    thresholds: dict[str, float],
    weights: dict[str, float],
    overrides: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Apply param override deltas to thresholds and weights (idempotent guard).

    Overrides are deltas (not absolute values). After applying:
    * Thresholds are clamped to ``[THRESHOLD_CLAMP_MIN, THRESHOLD_CLAMP_MAX]``.
    * Weights are clamped to ``[WEIGHT_CLAMP_MIN, WEIGHT_CLAMP_MAX]`` and
      re-normalized to sum to 1.0.

    **Idempotency**: if the ``_overrides_applied_flag`` ContextVar is already
    ``True``, this is a no-op — the caller is assumed to have already baked in
    the overrides. The flag is set to ``True`` after applying.

    **Immutability**: input dicts are never mutated; copies are used.

    Args:
        thresholds: The base threshold dict (not mutated).
        weights: The base weight dict (not mutated).
        overrides: A dict of param_key → delta.

    Returns:
        A tuple of (clamped_thresholds, clamped_normalized_weights).
    """
    from swarm.learning.adaptive_policy import _overrides_applied_flag
    from swarm.learning.policy import clamp_thresholds, clamp_weights

    # Idempotent guard: if overrides were already applied, no-op.
    if _overrides_applied_flag.get():
        return dict(thresholds), dict(weights)

    # Always work on copies — never mutate inputs (Part 2: immutability).
    new_thresholds = dict(thresholds)
    new_weights = dict(weights)

    for key, delta in overrides.items():
        if key in OVERRIDABLE_THRESHOLD_KEYS:
            new_thresholds[key] = new_thresholds.get(key, 0.7) + delta
        elif key in OVERRIDABLE_WEIGHT_KEYS:
            new_weights[key] = new_weights.get(key, 0.4) + delta

    result_thresholds = clamp_thresholds(new_thresholds)
    result_weights = clamp_weights(new_weights)

    # Mark that overrides have been applied for this execution context.
    _overrides_applied_flag.set(True)

    return result_thresholds, result_weights


def normalize_context(context: dict[str, Any]) -> dict[str, Any]:
    """Normalize a routing context dict for deterministic matching (Part 3).

    - Lower-cases all string keys.
    - Strips whitespace from string values.
    - Removes keys with ``None`` values.
    """
    normalized: dict[str, Any] = {}
    for key, value in context.items():
        norm_key = key.lower()
        if value is None:
            continue
        if isinstance(value, str):
            normalized[norm_key] = value.strip().lower()
        else:
            normalized[norm_key] = value
    return normalized


def _sort_rules(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort routing rules deterministically by (priority DESC, rule_id ASC).

    Rules without an explicit ``priority`` are treated as priority 0.
    Rules without an explicit ``rule_id`` are assigned a stable sequential id.
    """
    indexed = []
    for idx, rule in enumerate(rules):
        priority = int(rule.get("priority", 0))
        rule_id = str(rule.get("rule_id", f"rule_{idx}"))
        indexed.append((priority, rule_id, idx, rule))
    # Sort by priority DESC, then rule_id ASC.
    indexed.sort(key=lambda r: (-r[0], r[1]))
    return [item[3] for item in indexed]


def _matches(context: dict[str, Any], key: str, expected: Any) -> bool:
    """Check if a single condition matches the context value.

    Supports both exact equality and special operators for numeric comparisons:
    ``"<3"`` matches when the context value is less than 3.
    """
    if key not in context:
        return False
    actual = context[key]

    if isinstance(expected, str) and expected.startswith("<"):
        threshold = int(expected[1:])
        return bool(int(actual) < threshold)
    if isinstance(expected, str) and expected.startswith(">="):
        threshold = int(expected[2:])
        return bool(int(actual) >= threshold)

    return bool(actual == expected)


def extract_context_from_trace(trace: dict[str, Any]) -> dict[str, Any]:
    """Extract context from a full trace dict using ``extract_input``.

    This is a convenience wrapper that reconstructs the trace input from the
    trace's events, then calls :func:`extract_context`.
    """
    from swarm.simulation.replay_engine import extract_input

    events = trace.get("events") or []
    trace_input = extract_input(events)
    if not trace_input:
        return {"budget_level": "medium", "urgency": "low", "supplier_count": 3}
    return extract_context(trace_input)


def validate_param_overrides(overrides: dict[str, Any]) -> bool:
    """Validate param override deltas.

    Guardrails (Part 5):

    * Each key must be in OVERRIDABLE_THRESHOLD_KEYS or OVERRIDABLE_WEIGHT_KEYS.
    * Each delta must be within ``[-PARAM_OVERRIDE_DELTA, +PARAM_OVERRIDE_DELTA]``.
    * Zero-delta overrides are rejected (noise).
    * Total number of overrides must be <= MAX_PARAMS_PER_RULE.
    * Total absolute delta (sum of ``abs(v)``) must be <=
      ``PARAM_OVERRIDE_TOTAL_DELTA_CAP``.
    """
    if not isinstance(overrides, dict):
        return False
    if len(overrides) > MAX_PARAMS_PER_RULE:
        return False
    total_abs_delta = 0.0
    for key, delta in overrides.items():
        if key not in OVERRIDABLE_THRESHOLD_KEYS and key not in OVERRIDABLE_WEIGHT_KEYS:
            return False
        try:
            float_delta = float(delta)
        except (TypeError, ValueError):
            return False
        # Reject zero-delta overrides (noise).
        if abs(float_delta) < 1e-12:
            return False
        if abs(float_delta) > PARAM_OVERRIDE_DELTA + 1e-9:
            return False
        total_abs_delta += abs(float_delta)
    return total_abs_delta <= PARAM_OVERRIDE_TOTAL_DELTA_CAP + 1e-9


def validate_routing_strategy(strategy: dict[str, Any]) -> bool:
    """Validate a strategy dict structure.

    Accepts both single-strategy (Step 23) and routing (Step 24/25) formats.

    Single strategy:
        * type must be in STRATEGY_TYPES

    Routing strategy:
    * type == "routing"
    * rules is a list with length <= MAX_ROUTING_RULES
    * each rule has "conditions" (dict) and "strategy" (str in STRATEGY_TYPES)
    * conditions use only keys from ROUTING_CONDITION_KEYS
    * param_overrides (optional) must validate via validate_param_overrides
    * default is in STRATEGY_TYPES

    Returns:
        True if valid, False otherwise.
    """
    if not isinstance(strategy, dict):
        return False

    strat_type = strategy.get("type")

    # Single strategy (Step 23): type is the strategy name directly.
    if strat_type in STRATEGY_TYPES:
        return True

    if strat_type != "routing":
        return False

    if not isinstance(strategy.get("rules"), list):
        return False

    if len(strategy["rules"]) > MAX_ROUTING_RULES:
        return False

    default = strategy.get("default", "balanced")
    if default not in STRATEGY_TYPES:
        return False

    for rule in strategy["rules"]:
        if not isinstance(rule, dict):
            return False
        conditions = rule.get("conditions")
        if not isinstance(conditions, dict) or not conditions:
            return False
        for key in conditions:
            if key not in ROUTING_CONDITION_KEYS:
                return False
        strat = rule.get("strategy")
        if strat not in STRATEGY_TYPES:
            return False
        # Validate param overrides if present (Step 25).
        overrides = rule.get("param_overrides")
        if overrides is not None and not validate_param_overrides(overrides):
            return False

    return True


def generate_routing_candidates_with_params(base: Policy) -> list[dict[str, Any]]:
    """Generate routing candidates with param overrides (Step 25).

    For each routing condition (from ROUTING_CONDITIONS), generates candidates
    that:
    1. Use a non-default strategy with that condition
    2. Optionally combine the condition with 1 param override

    This produces a bounded set of context-aware routing policies.
    """
    base_strat_type = (
        base.strategy.get("type", "balanced")
        if base.strategy
        else "balanced"
    )

    candidates: list[dict[str, Any]] = []

    for cond_key, cond_val in ROUTING_CONDITIONS:
        for strat_type in STRATEGY_TYPES:
            if strat_type == base_strat_type:
                continue

            # Pure routing (no param override).
            rule = {
                "type": "routing",
                "rules": [
                    {
                        "conditions": {cond_key: cond_val},
                        "strategy": strat_type,
                    }
                ],
                "default": base_strat_type,
            }
            candidates.append(rule)

            # Routing + 1 param override.
            for override_key, override_delta in PARAM_OVERRIDE_SPACE:
                rule_with_param = {
                    "type": "routing",
                    "rules": [
                        {
                            "conditions": {cond_key: cond_val},
                            "strategy": strat_type,
                            "param_overrides": {override_key: override_delta},
                        }
                    ],
                    "default": base_strat_type,
                }
                candidates.append(rule_with_param)

    # Hard cap on total candidates (Part 5).
    if len(candidates) > MAX_CANDIDATES:
        # Deterministic sampling with seed=42.
        rng = random.Random(CANDIDATE_SAMPLING_SEED)
        candidates = rng.sample(candidates, MAX_CANDIDATES)

    return candidates
