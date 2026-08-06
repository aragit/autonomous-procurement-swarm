"""Closed-loop policy data model, versioning, grid and safety (v1.1 Step 22).

This module owns *only* the policy **data model** and its deterministic
derivatives — signature/version computation, grid generation and the safety
guardrail. All learning (trace gathering, candidate evaluation, objective
scoring) and promotion live in :mod:`swarm.learning.learner`, keeping the
data model pure and free of I/O or replay dependencies.

Versioning (hard locked, no implicit ordering)::

    signature = sha256(JSONSorted(canonical_params(thresholds, weights)))
    version   = sha256(signature)[:12]

``canonical_params`` always emits thresholds (keys sorted lexicographically)
then weights (keys sorted lexicographically). The same params therefore always
produce the same ``version`` → idempotent ``/policies/learn``.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any

from swarm.config import (
    CANDIDATE_SAMPLING_SEED,
    DEFAULT_STRATEGY_TYPE,
    GRID_DELTAS,
    MAX_CANDIDATES,
    POLICY_MAX_PARAM_DELTA,
    STRATEGY_TYPES,
    THRESHOLD_CLAMP_MAX,
    THRESHOLD_CLAMP_MIN,
    WEIGHT_CLAMP_MAX,
    WEIGHT_CLAMP_MIN,
)

#: Canonical parameter keys — ordering is strict and documented so the
#: signature is reproducible across runs and interpreters.
THRESHOLD_KEYS: tuple[str, ...] = (
    "confidence_threshold",
    "stability_threshold",
    "trust_threshold",
)
WEIGHT_KEYS: tuple[str, ...] = (
    "price_weight",
    "score_weight",
    "carbon_weight",
)
#: Strategy key included in canonical_params for signature determinism.
STRATEGY_KEY: str = "strategy"

#: Grid deltas applied per parameter (step=0.05, range=[-0.10, +0.10]).
_DELTAS: tuple[float, ...] = tuple(GRID_DELTAS)

@dataclass
class Policy:
    """A learned or promoted procurement policy.

    Attributes:
        version: Deterministic 12-char id (``sha256(signature)[:12]``).
        signature: Full canonical parameter hash.
        thresholds: ``confidence/stability/trust`` gate thresholds.
        weights: ``price/score/carbon`` strategy weights (sum to 1.0).
        strategy: Strategy dict with ``type`` key (e.g. ``{"type": "balanced"}``).
        metric: Objective metric from the last evaluation (``None`` if unscored).
        success_rate: Legacy field; set to decision_stability during evaluation
            for backward compatibility.
        avg_score: Mean replayed score from the last evaluation.
        feedback_success_rate: Fraction of evaluated traces marked successful
            in feedback.
        feedback_outcome_score: Mean ``outcome_score`` from feedback.
        decision_stability: Fraction of replays reproducing the original
            supplier (regularizer in the hybrid objective).
        active: Whether this is the currently-promoted policy.
        created_at: ISO timestamp of when the policy was stored.
    """

    version: str
    signature: str
    thresholds: dict[str, float]
    weights: dict[str, float]
    strategy: dict[str, Any] | None = None
    metric: float | None = None
    success_rate: float | None = None
    avg_score: float | None = None
    feedback_success_rate: float | None = None
    feedback_outcome_score: float | None = None
    decision_stability: float | None = None
    active: bool = False
    created_at: str | None = None

    def __post_init__(self) -> None:
        if self.strategy is None:
            self.strategy = {"type": DEFAULT_STRATEGY_TYPE}

    def to_record(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "signature": self.signature,
            "thresholds": dict(self.thresholds),
            "weights": dict(self.weights),
            "strategy": dict(self.strategy) if self.strategy else {"type": DEFAULT_STRATEGY_TYPE},
            "metric": self.metric,
            "success_rate": self.success_rate,
            "avg_score": self.avg_score,
            "feedback_success_rate": self.feedback_success_rate,
            "feedback_outcome_score": self.feedback_outcome_score,
            "decision_stability": self.decision_stability,
            "active": self.active,
            "created_at": self.created_at,
        }


def default_thresholds() -> dict[str, float]:
    """Static config-defined thresholds (the fallback before any learning)."""
    from swarm.config import CONFIDENCE_THRESHOLD, STABILITY_THRESHOLD, TRUST_THRESHOLD

    return {
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "stability_threshold": STABILITY_THRESHOLD,
        "trust_threshold": TRUST_THRESHOLD,
    }


def default_weights() -> dict[str, float]:
    """Static config-defined strategy weights (the balanced baseline)."""
    from swarm.domain.strategy import BALANCED_STRATEGY

    return BALANCED_STRATEGY.as_weights()


def default_strategy() -> dict[str, Any]:
    """The default strategy dict: ``{"type": "balanced"}``.

    v1.1 Step 23: single strategy. v1.1 Step 24: also supports routing via
    ``single_strategy()`` helper.
    """
    return {"type": "balanced"}


def single_strategy(strategy_type: str = DEFAULT_STRATEGY_TYPE) -> dict[str, Any]:
    """Create a single-strategy dict (Step 23 style)."""
    return {"type": "balanced", "strategy": strategy_type}


def routing_strategy(
    rules: list[dict[str, Any]],
    default: str = DEFAULT_STRATEGY_TYPE,
) -> dict[str, Any]:
    """Create a routing strategy dict (Step 24 style).

    Args:
        rules: List of rule dicts, each with ``conditions`` and ``strategy``.
        default: Fallback strategy type when no rule matches.
    """
    return {"type": "routing", "rules": rules, "default": default}


def default_policy() -> Policy:
    """The baseline policy: config thresholds + balanced strategy weights.

    Constructed lazily so module import does not trigger a cascade of domain
    imports.
    """
    return build_policy(default_thresholds(), default_weights(), default_strategy())


# --------------------------------------------------------------------------- #
# Determinism: signature + version
# --------------------------------------------------------------------------- #


def canonical_params(
    thresholds: dict[str, float],
    weights: dict[str, float],
    strategy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical parameter dict in a strict, documented key order.

    Sorting rule (v1.1 Step 22 — hard locked):

      1. ``thresholds`` sub-dict, keys sorted lexicographically.
      2. ``weights`` sub-dict, keys sorted lexicographically.
      3. ``strategy`` sub-dict, keys sorted lexicographically.
      4. Top-level keys are always ``[thresholds, weights, strategy]``.

    No implicit ordering anywhere: every consumer sorts explicitly.
    """
    if strategy is None:
        strategy = default_strategy()
    sorted_thresholds = {k: thresholds[k] for k in sorted(THRESHOLD_KEYS) if k in thresholds}
    sorted_weights = {k: weights[k] for k in sorted(WEIGHT_KEYS) if k in weights}
    sorted_strategy = {k: strategy[k] for k in sorted(strategy) if k in strategy}
    return {
        "thresholds": sorted_thresholds,
        "weights": sorted_weights,
        "strategy": sorted_strategy,
    }


def compute_signature(
    thresholds: dict[str, float],
    weights: dict[str, float],
    strategy: dict[str, Any] | None = None,
) -> str:
    """SHA-256 of the JSON-serialised canonical params (sorted keys)."""
    canonical = canonical_params(thresholds, weights, strategy)
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compute_version(signature: str) -> str:
    """Deterministic 12-char version: ``sha256(signature)[:12]``."""
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]


def build_policy(
    thresholds: dict[str, float],
    weights: dict[str, float],
    strategy: dict[str, Any] | None = None,
) -> Policy:
    """Construct a :class:`Policy` with derived signature + version.

    The thresholds/weights must already be clamped/normalised (callers use
    :func:`clamp_thresholds` / :func:`clamp_weights` before constructing).
    If ``strategy`` is None, defaults to ``{"type": "balanced"}``.
    """
    if strategy is None:
        strategy = default_strategy()
    signature = compute_signature(thresholds, weights, strategy)
    version = compute_version(signature)
    return Policy(
        version=version,
        signature=signature,
        thresholds=dict(thresholds),
        weights=dict(weights),
        strategy=dict(strategy),
    )


# --------------------------------------------------------------------------- #
# Grid clamping + normalization
# --------------------------------------------------------------------------- #


def clamp_thresholds(thresholds: dict[str, float]) -> dict[str, float]:
    """Clamp every threshold to ``[THRESHOLD_CLAMP_MIN, THRESHOLD_CLAMP_MAX]``.

    Avoids overly permissive behaviour (default band is ``[0.6, 0.9]``).
    """
    return {
        k: max(THRESHOLD_CLAMP_MIN, min(THRESHOLD_CLAMP_MAX, float(v)))
        for k, v in thresholds.items()
    }


def clamp_weights(weights: dict[str, float]) -> dict[str, float]:
    """Clamp each weight to ``[WEIGHT_CLAMP_MIN, WEIGHT_CLAMP_MAX]`` then
    normalize so ``price + score + carbon == 1.0`` exactly.

    Normalization happens *after* clamping (per the Step 22 spec).
    """
    clamped = {
        k: max(WEIGHT_CLAMP_MIN, min(WEIGHT_CLAMP_MAX, float(v)))
        for k, v in weights.items()
    }
    total = sum(clamped.values())
    if total <= 0.0:
        # Degenerate fallback: equal split, then re-clamp.
        clamped = {k: 1.0 / len(clamped) for k in clamped}
        total = sum(clamped.values())
    normalized = {k: v / total for k, v in clamped.items()}
    return normalized


def _with_thresholds(base: dict[str, float], key: str, delta: float) -> dict[str, float]:
    perturbed = dict(base)
    perturbed[key] = max(
        THRESHOLD_CLAMP_MIN, min(THRESHOLD_CLAMP_MAX, perturbed[key] + delta)
    )
    return clamp_thresholds(perturbed)


def _with_weights(base: dict[str, float], key: str, delta: float) -> dict[str, float]:
    perturbed = dict(base)
    perturbed[key] = perturbed[key] + delta
    # carbon is derived to keep the sum at 1.0 before clamping/normalising
    if key in ("price_weight", "score_weight"):
        perturbed["carbon_weight"] = 1.0 - perturbed["price_weight"] - perturbed["score_weight"]
    return clamp_weights(perturbed)


def _within_param_deltas(policy: Policy, base: Policy) -> bool:
    """Check that no parameter deviates from the base by more than
    ``POLICY_MAX_PARAM_DELTA``.

    This is a hard safety boundary preventing unbounded drift across learning
    cycles. Because clamping is performed before this check, we compare the
    *clamped* policy parameters against the *clamped* base parameters.
    """
    for key in THRESHOLD_KEYS:
        if abs(policy.thresholds[key] - base.thresholds[key]) > POLICY_MAX_PARAM_DELTA:
            return False
    for key in WEIGHT_KEYS:
        if abs(policy.weights[key] - base.weights[key]) > POLICY_MAX_PARAM_DELTA:
            return False
    return True


def generate_candidates(base: Policy) -> list[Policy]:
    """Generate the grid-search neighbourhood around ``base``.

    Uses *coordinate perturbation* on parameters (3 thresholds +
    ``price_weight`` + ``score_weight``; ``carbon_weight`` is derived to keep
    the weights summing to 1.0) shifted by each grid delta
    ``[-0.10, -0.05, 0, +0.05, +0.10]``, then **cross-products** each parameter
    variant with every strategy in ``STRATEGY_TYPES`` (including the base
    strategy). The base policy itself is always included.

    Clamping + normalization may collapse distinct perturbations onto the same
    policy, so candidates are de-duplicated by signature.

    **Drift safety**: any candidate whose parameters deviate from ``base`` by
    more than ``POLICY_MAX_PARAM_DELTA`` (default 0.20) on any single parameter
    is rejected. This prevents unbounded drift across learning cycles.

    (v1.1 Step 23: strategy is a first-class policy dimension, explored as a
    finite cross-product with the parameter grid. v1.1 Step 24: also generates
    routing strategy candidates from a bounded condition space.)
    """
    seen: set[str] = set()
    candidates: list[Policy] = []

    def _add(thresholds: dict[str, float], weights: dict[str, float],
             strategy: dict[str, Any]) -> None:
        policy = build_policy(clamp_thresholds(thresholds), clamp_weights(weights), strategy)
        if policy.signature in seen:
            return
        if not _within_param_deltas(policy, base):
            return
        seen.add(policy.signature)
        candidates.append(policy)

    # Base strategy from the base policy (default: "balanced").
    base_strategy = base.strategy if base.strategy else default_strategy()

    # Base policy first (stable for idempotent selection).
    _add(base.thresholds, base.weights, base_strategy)

    # Parameter perturbations × all strategies.
    param_variants: list[tuple[dict[str, float], dict[str, float]]] = []
    seen_params: set[tuple[str, str]] = set()

    # Thresholds perturbations
    for key in THRESHOLD_KEYS:
        for delta in _DELTAS:
            if delta == 0.0:
                continue
            t = clamp_thresholds(_with_thresholds(base.thresholds, key, delta))
            w = clamp_weights(base.weights)
            sig = (json.dumps(t, sort_keys=True), json.dumps(w, sort_keys=True))
            if sig not in seen_params:
                seen_params.add(sig)
                param_variants.append((t, w))

    # Weight perturbations
    for key in ("price_weight", "score_weight"):
        for delta in _DELTAS:
            if delta == 0.0:
                continue
            t = clamp_thresholds(base.thresholds)
            w = clamp_weights(_with_weights(base.weights, key, delta))
            sig = (json.dumps(t, sort_keys=True), json.dumps(w, sort_keys=True))
            if sig not in seen_params:
                seen_params.add(sig)
                param_variants.append((t, w))

    # Add each parameter variant with each strategy type (including base).
    all_strategies = sorted(set(STRATEGY_TYPES))
    for strat_type in all_strategies:
        strat = {"type": strat_type}
        for t, w in param_variants:
            _add(t, w, strat)

    # Also add the base parameters with each non-base strategy.
    for strat_type in all_strategies:
        if strat_type == base_strategy.get("type"):
            continue
        strat = {"type": strat_type}
        _add(clamp_thresholds(base.thresholds), clamp_weights(base.weights), strat)

    # Routing strategy candidates (Step 24/25).
    from swarm.learning.routing import generate_routing_candidates_with_params

    for routing in generate_routing_candidates_with_params(base):
        _add(clamp_thresholds(base.thresholds), clamp_weights(base.weights), routing)

    candidates.sort(key=lambda p: p.signature)

    # Hard cap on total candidates (Step 25 Hardened).
    if len(candidates) > MAX_CANDIDATES:
        rng = random.Random(CANDIDATE_SAMPLING_SEED)
        candidates = rng.sample(candidates, MAX_CANDIDATES)
        candidates.sort(key=lambda p: p.signature)

    return candidates


# --------------------------------------------------------------------------- #
# Policy safety guardrail
# --------------------------------------------------------------------------- #


def is_safe_policy(policy: Policy) -> bool:
    """Reject policies that violate the Step 22 safety floor.

    A candidate is unsafe — and must never enter the DB — if any threshold is
    below ``THRESHOLD_CLAMP_MIN`` (0.6) or above ``THRESHOLD_CLAMP_MAX`` (0.9),
    or any weight leaves the ``[WEIGHT_CLAMP_MIN, WEIGHT_CLAMP_MAX]`` band, or
    the weights do not sum to 1.0, or the strategy type is not in
    ``STRATEGY_TYPES``. Because :func:`generate_candidates` already
    clamps+normalizes and restricts to known strategies, this is a defensive
    backstop for any policy constructed otherwise (e.g. by an external caller
    of :func:`build_policy`).
    """
    thr = policy.thresholds
    tmin, tmax = THRESHOLD_CLAMP_MIN, THRESHOLD_CLAMP_MAX
    if not (tmin <= thr["confidence_threshold"] <= tmax):
        return False
    if not (tmin <= thr["stability_threshold"] <= tmax):
        return False
    if not (tmin <= thr["trust_threshold"] <= tmax):
        return False

    w = policy.weights
    wmin, wmax = WEIGHT_CLAMP_MIN, WEIGHT_CLAMP_MAX
    for key in WEIGHT_KEYS:
        if not (wmin <= w[key] <= wmax):
            return False
    if abs(sum(w.values()) - 1.0) >= 1e-6:
        return False
    # Strategy must be a valid single or routing strategy.
    from swarm.learning.routing import validate_routing_strategy

    return validate_routing_strategy(policy.strategy) if policy.strategy else False
