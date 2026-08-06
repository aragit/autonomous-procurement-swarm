"""Adaptive threshold + weight resolution (v0.9 Step 20 / v1.1 Step 22).

Provides the runtime resolution of the LLM-gating **thresholds**
(``confidence_threshold``, ``stability_threshold``, ``trust_threshold``) and
the **strategy weights** used by the evaluation agents.

Runtime resolution priority (v1.1 Step 22 — hard locked, no deviation)::

    1. override_thresholds (ContextVar, pinned by replay / policy learner)
    2. active_policy.thresholds  (last promoted policy)
    3. config fallback           (static SWARM_*_THRESHOLD defaults)

The same three-tier priority applies to strategy weights::

    1. override_strategy_weights (ContextVar, pinned by the learner)
    2. active_policy.weights     (last promoted policy)
    3. None  ->  canonical strategy artifact (the pre-existing behaviour)

Design notes:

* **Separation of learning from execution.** Feedback is *never* read at
  runtime here. Trace feedback is consumed only by the closed-loop learner
  (:mod:`swarm.learning.learner`) to build candidate policies; the winning
  candidate is published via :func:`~swarm.storage.event_store.set_policy_active`
  and then read here as the active policy. This keeps the runtime path a pure,
  deterministic, stateless lookup.
* **Read-only.** Neither function mutates config or state.
* **Replay-safe.** The ContextVar overrides let the replay engine and learner
  pin exact values so re-runs are bit-for-bit reproducible.
* **Safe degradation.** When the DB is unavailable or no policy is active,
  config defaults are returned unconditionally.

The pure feedback-driven adaptation algorithm lives in
:func:`compute_adaptive_thresholds` (Step 20). It is retained as a documented,
side-effect-free utility — the learner may consume it as one candidate source —
but it is intentionally *not* wired into the runtime path above.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Any

from swarm.config import (
    ADAPTIVE_K,
    CONFIDENCE_THRESHOLD,
    DEFAULT_STRATEGY_TYPE,
    ENABLE_LEARNING,
    MIN_FEEDBACK_SAMPLES,
    STABILITY_THRESHOLD,
    TRUST_THRESHOLD,
)
from swarm.storage.event_store import init_db, load_active_policy

#: When set (e.g. by the replay engine / learner), ``get_adaptive_thresholds``
#: returns this value. Pin thresholds deterministically for replay/simulation.
_thresholds_override: ContextVar[dict[str, float] | None] = ContextVar(
    "swarm_thresholds_override", default=None
)

#: When set, the evaluation agents use these strategy weights instead of the
#: canonical strategy weights. Pinned by the learner during candidate eval, and
#: by promoted policies at runtime via :func:`get_strategy_weights`.
_weights_override: ContextVar[dict[str, float] | None] = ContextVar(
    "swarm_weights_override", default=None
)

#: When set (e.g. by the replay engine / learner), ``get_strategy_type``
#: returns this value. Pin strategy type deterministically for replay/simulation.
_type_override: ContextVar[str | None] = ContextVar(
    "swarm_strategy_type_override", default=None
)

#: Tracks whether param overrides have already been applied to the current
#: thresholds/weights within a given execution context. Prevents double
#: application when both the runtime path and the replay path would apply the
#: same overrides.
_overrides_applied_flag: ContextVar[bool] = ContextVar(
    "swarm_overrides_applied", default=False
)


@contextlib.contextmanager
def override_thresholds(thresholds: dict[str, float]) -> Any:
    """Pin the active thresholds for the duration of the context.

    Used by the replay/simulation engine and the policy learner to guarantee
    deterministic re-runs that do not depend on live database reads.
    """
    token = _thresholds_override.set(thresholds)
    try:
        yield
    finally:
        _thresholds_override.reset(token)


@contextlib.contextmanager
def override_strategy_weights(weights: dict[str, float] | None) -> Any:
    """Pin the active strategy weights for the duration of the context.

    Mirrors :func:`override_thresholds`: when the ContextVar is set the
    evaluation agents read these weights instead of the canonical
    strategy weights, so candidate policies are evaluated with replay parity.
    Passing ``None`` clears the override (restoring the canonical fallback).
    """
    token = _weights_override.set(weights)
    try:
        yield
    finally:
        _weights_override.reset(token)


@contextlib.contextmanager
def override_strategy_type(strategy_type: str) -> Any:
    """Pin the active strategy type for the duration of the context.

    Used by the replay engine and the policy learner to guarantee
    deterministic strategy selection during candidate evaluation.
    The strategy type must be a member of ``STRATEGY_TYPES``.
    """
    token = _type_override.set(strategy_type)
    try:
        yield
    finally:
        _type_override.reset(token)


@contextlib.contextmanager
def reset_overrides_flag() -> Any:
    """Reset the param-overrides-applied flag to ``False``.

    Used at the entry points of replay and live execution to ensure param
    overrides are applied exactly once per procurement run, preventing double
    application if the runtime path and the replay path both invoke
    ``apply_param_overrides``.
    """
    token = _overrides_applied_flag.set(False)
    try:
        yield
    finally:
        _overrides_applied_flag.reset(token)


def _active_policy() -> dict[str, Any] | None:
    """Read the active policy from the store, or ``None``.

    Returns ``None`` when learning is disabled, the DB is unavailable, or no
    policy has been promoted — signalling callers to fall back to config.
    """
    if not ENABLE_LEARNING:
        return None
    try:
        init_db()
        return load_active_policy()
    except Exception:
        return None


def get_param_overrides(context: dict[str, Any] | None = None) -> dict[str, float]:
    """Resolve param overrides from the active routing strategy (Step 25).

    If the active policy uses a routing strategy and ``context`` is provided,
    the matched rule's ``param_overrides`` are returned. Otherwise, an empty
    dict is returned.

    Args:
        context: The extracted context dict (budget_level, urgency, etc.).

    Returns:
        A dict of param_key → delta, or {} if no routing overrides apply.
    """
    policy = _active_policy()
    if policy is not None:
        strat = policy.get("strategy")
        if (
            strat
            and isinstance(strat, dict)
            and strat.get("type") == "routing"
            and context is not None
        ):
            from swarm.learning.routing import select_strategy_with_params

            decision = select_strategy_with_params(context, strat)
            return decision.param_overrides
    return {}


def get_adaptive_thresholds(
    context: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Resolve the runtime LLM-gating thresholds (strict priority).

    Resolution order — **no deviation allowed**::

        1. override_thresholds (ContextVar)  → pinned value
        2. active_policy.thresholds         → last promoted policy
        3. config fallback                  → static SWARM_* thresholds

    Param overrides from the active routing strategy are applied to the
    policy's thresholds (Step 25). When thresholds are pinned (override
    ContextVar is set — e.g. during candidate evaluation), the overrides are
    already baked in, so we do not apply them a second time.

    Feedback is never read here (kept out of the runtime path per the
    Step 22 separation of learning from execution).
    """
    override = _thresholds_override.get()
    if override is not None:
        return dict(override)

    policy = _active_policy()
    if policy is not None and policy.get("thresholds") is not None:
        thresholds = dict(policy["thresholds"])
        # Apply param overrides from routing strategy (Step 25).
        overrides = get_param_overrides(context)
        if overrides:
            from swarm.learning.policy import clamp_weights
            from swarm.learning.routing import apply_param_overrides

            policy_weights = policy.get("weights") or get_strategy_weights()
            if policy_weights is None:
                policy_weights = {"price_weight": 0.4, "score_weight": 0.4, "carbon_weight": 0.2}
            thresholds, _ = apply_param_overrides(
                thresholds, clamp_weights(policy_weights), overrides
            )
        return thresholds

    return _default_thresholds()


def get_strategy_weights() -> dict[str, float] | None:
    """Resolve the runtime strategy weights (strict priority, mirrors thresholds).

    Resolution order::

        1. override_strategy_weights (ContextVar) → pinned value
        2. active_policy.weights                 → last promoted policy
        3. None                                  → canonical strategy artifact

    Returns ``None`` when no override is pinned and no policy is active, so
    callers fall back to the canonical ``StrategyArtifact``-derived weights
    (the pre-existing behaviour — fully backward compatible).
    """
    override = _weights_override.get()
    if override is not None:
        return dict(override)

    policy = _active_policy()
    if policy is not None and policy.get("weights") is not None:
        return dict(policy["weights"])
    return None


def _default_strategy_type() -> str:
    """Return the static config default strategy type."""
    return DEFAULT_STRATEGY_TYPE


def get_strategy_type(context: dict[str, Any] | None = None) -> str:
    """Resolve the runtime strategy type (strict priority, mirrors thresholds).

    Resolution order::

        1. override_strategy_type (ContextVar) → pinned value
        2. active_policy.strategy               → last promoted policy
        3. config fallback                      → DEFAULT_STRATEGY_TYPE

    For routing strategies (Step 24), if ``context`` is provided, the routing
    rules are evaluated to select a strategy type. If ``context`` is None and
    the active strategy is routing, the default is returned.

    Strategy selection is fully deterministic and replay-safe.
    """
    override = _type_override.get()
    if override is not None:
        return override

    policy = _active_policy()
    if policy is not None:
        strat = policy.get("strategy")
        if strat and isinstance(strat, dict) and strat.get("type"):
            strat_type = strat["type"]
            # If it's a routing strategy and context is provided, select.
            if strat_type == "routing" and context is not None:
                from swarm.learning.routing import select_strategy

                return select_strategy(context, strat)
            # Single strategy: type IS the strategy name.
            return str(strat_type)

    return _default_strategy_type()


def apply_strategy_weights(
    strategy_type: str,
    base_weights: dict[str, float],
) -> dict[str, float]:
    """Apply a deterministic weight transformation for the given strategy type.

    Each strategy defines how the base weights are perturbed before clamping
    and normalization. The transformations are deterministic and bounded:

    * ``balanced`` — no change (identity).
    * ``cost_optimized`` — price_weight *= 1.2 (favor lower price).
    * ``quality_first`` — score_weight *= 1.2 (favor quality).
    * ``trust_weighted`` — carbon_weight *= 1.2 (favor trusted/low-carbon).

    After perturbation, weights are re-clamped to ``[WEIGHT_CLAMP_MIN,
    WEIGHT_CLAMP_MAX]`` and re-normalized to sum to 1.0.
    """
    from swarm.learning.policy import clamp_weights

    perturbed = dict(base_weights)
    if strategy_type == "cost_optimized":
        perturbed["price_weight"] = perturbed.get("price_weight", 0.4) * 1.2
    elif strategy_type == "quality_first":
        perturbed["score_weight"] = perturbed.get("score_weight", 0.4) * 1.2
    elif strategy_type == "trust_weighted":
        perturbed["carbon_weight"] = perturbed.get("carbon_weight", 0.2) * 1.2
    # "balanced" → no change
    return clamp_weights(perturbed)


def get_config_thresholds() -> dict[str, float]:
    """Return the static, config-defined thresholds (ignoring all feedback).

    This is the "static" threshold set used when learning is disabled or when a
    replay/simulation explicitly requests config-only thresholds.
    """
    return _default_thresholds()


def compute_adaptive_thresholds(feedback_list: list[dict[str, Any]]) -> dict[str, float]:
    """Compute threshold adjustments from a list of feedback records (Step 20).

    .. note::
        Pure utility — **not** wired into :func:`get_adaptive_thresholds`. The
        runtime path no longer reads feedback directly; candidate policies are
        produced by the closed-loop learner (:mod:`swarm.learning.learner`) and
        published as the active policy.

    Adaptation rules (deterministic, based on sorted feedback):

    1. **Confidence** — High-confidence decisions that fail indicate
       overconfidence → lower the confidence threshold.
    2. **Stability** — Failures correlating with volatility → raise the
       stability threshold.
    3. **Trust** — High trust with poor outcomes → lower the trust threshold.

    All values are clamped to ``[0.5, 0.95]``.

    Args:
        feedback_list: List of feedback dicts, each with ``outcome_score``
            (float 0-1), ``success`` (bool), and optionally ``trust`` (float).

    Returns:
        A dict with ``confidence_threshold``, ``stability_threshold``, and
        ``trust_threshold`` keys (all floats).
    """
    if not feedback_list or not _learning_enabled():
        return _default_thresholds()

    sorted_feedback = sorted(feedback_list, key=lambda f: f.get("created_at", ""))

    n = len(sorted_feedback)
    if n < MIN_FEEDBACK_SAMPLES:
        return _default_thresholds()

    failures = [f for f in sorted_feedback if not f.get("success", False)]
    failure_ratio = len(failures) / n if n > 0 else 0.0

    # --- Rule 1: Confidence calibration ---
    overconfident_failures = [f for f in failures if f.get("outcome_score", 0.0) < 0.5]
    overconfident_ratio = len(overconfident_failures) / len(failures) if failures else 0.0
    confidence_delta = ADAPTIVE_K * overconfident_ratio
    confidence_threshold = max(0.5, CONFIDENCE_THRESHOLD - confidence_delta)

    # --- Rule 2: Stability sensitivity ---
    stability_delta = ADAPTIVE_K * failure_ratio
    stability_threshold = min(0.95, STABILITY_THRESHOLD + stability_delta)

    # --- Rule 3: Trust adjustment ---
    trust_scores = [f.get("trust", 0.7) for f in sorted_feedback if "trust" in f]
    if trust_scores and failure_ratio > 0.5:
        avg_trust = sum(trust_scores) / len(trust_scores)
        trust_delta = ADAPTIVE_K * failure_ratio * (avg_trust - 0.5)
        trust_threshold = max(0.5, TRUST_THRESHOLD - trust_delta)
    else:
        trust_threshold = TRUST_THRESHOLD

    confidence_threshold = min(max(confidence_threshold, 0.5), 0.95)
    stability_threshold = min(max(stability_threshold, 0.5), 0.95)
    trust_threshold = min(max(trust_threshold, 0.5), 0.95)

    return {
        "confidence_threshold": round(confidence_threshold, 4),
        "stability_threshold": round(stability_threshold, 4),
        "trust_threshold": round(trust_threshold, 4),
    }


def _learning_enabled() -> bool:
    return ENABLE_LEARNING


def _default_thresholds() -> dict[str, float]:
    return {
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "stability_threshold": STABILITY_THRESHOLD,
        "trust_threshold": TRUST_THRESHOLD,
    }
