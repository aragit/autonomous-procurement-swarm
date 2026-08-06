"""Closed-loop policy learner: evaluation, search and promotion (v1.1 Step 22).

This is the *learning* half of the closed loop. It is intentionally kept
separate from the runtime resolution in
:mod:`swarm.learning.adaptive_policy` so that:

* learning never disturbs the live (execution) path — evaluation always runs
  through the isolated replay engine with pinned thresholds/weights;
* the runtime path stays a pure, deterministic, feedback-free lookup
  (see :func:`~swarm.learning.adaptive_policy.get_adaptive_thresholds`).

Pipeline implemented by :func:`learn_candidates`:

    1. Gather raw traces + raw feedback from the event store and merge them
       locally (the event store stays dumb: it only returns raw rows).
    2. Keep traces that have BOTH a persisted decision and feedback.
    3. If fewer than ``MIN_TRACES_FOR_LEARNING`` qualify → ``insufficient_data``.
    4. Generate the deterministic grid neighbourhood of the active policy.
    5. Evaluate every candidate via isolated replay, logging each evaluation
       (``policy_version, metric, success_rate, avg_score, delta_from_active``).
    6. Reject unsafe candidates (threshold/weight floor guardrail).
    7. Persist candidates (idempotent on version) and return the best.

Promotion (:func:`promote_policy`) is atomic and applies the Step 22 rule:

    metric > active.metric + IMPROVEMENT_MARGIN
    AND success_rate >= active.success_rate
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import structlog

from swarm.config import (
    DEFAULT_STRATEGY_TYPE,
    FEEDBACK_SCORE_WEIGHT,
    FEEDBACK_SUCCESS_WEIGHT,
    IMPROVEMENT_MARGIN,
    MIN_TRACES_FOR_LEARNING,
    OBJECTIVE_SCORE_WEIGHT,
    OBJECTIVE_SUCCESS_WEIGHT,
    STABILITY_WEIGHT,
    STRATEGY_TYPES,
)
from swarm.learning.adaptive_policy import (
    override_strategy_type,
    override_strategy_weights,
    override_thresholds,
)
from swarm.learning.policy import (
    Policy,
    default_policy,
    generate_candidates,
    is_safe_policy,
)
from swarm.storage.event_store import (
    load_active_policy,
    load_full_trace,
    load_policy,
    set_policy_active,
    store_policy,
)

logger = structlog.get_logger("swarm.learning.learner")


# --------------------------------------------------------------------------- #
# Objective function + evaluation
# --------------------------------------------------------------------------- #


def compute_metric(success_rate: float, avg_score: float) -> float:
    """Legacy objective: ``0.7 * success_rate + 0.3 * avg_score``.

    Latency is deliberately excluded (Step 23+ concern).

    .. deprecated::
        Use :func:`compute_hybrid_metric` instead, which weights ground-truth
        feedback as the primary signal.
    """
    return round(OBJECTIVE_SUCCESS_WEIGHT * success_rate + OBJECTIVE_SCORE_WEIGHT * avg_score, 4)


def compute_hybrid_metric(
    feedback_success_rate: float,
    feedback_outcome_score: float,
    decision_stability: float,
) -> float:
    """Hybrid objective: feedback is primary, stability is a regularizer.

    ``metric = 0.5 * feedback_success_rate
              + 0.3 * feedback_outcome_score
              + 0.2 * decision_stability``

    Args:
        feedback_success_rate: Fraction of traces marked successful in feedback.
        feedback_outcome_score: Average outcome_score from feedback.
        decision_stability: Fraction of replays that reproduce the original
            selected supplier (regularizer against chaotic policies).
    """
    return round(
        FEEDBACK_SUCCESS_WEIGHT * feedback_success_rate
        + FEEDBACK_SCORE_WEIGHT * feedback_outcome_score
        + STABILITY_WEIGHT * decision_stability,
        4,
    )


@dataclass
class EvaluationLog:
    """A single candidate evaluation record (logged per Step 22 requirement)."""

    policy_version: str
    metric: float
    success_rate: float
    avg_score: float
    delta_from_active: float
    feedback_success_rate: float | None = None
    feedback_outcome_score: float | None = None
    decision_stability: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "metric": self.metric,
            "success_rate": self.success_rate,
            "avg_score": self.avg_score,
            "delta_from_active": self.delta_from_active,
            "feedback_success_rate": self.feedback_success_rate,
            "feedback_outcome_score": self.feedback_outcome_score,
            "decision_stability": self.decision_stability,
        }


@dataclass
class EvaluationResult:
    """Aggregated evaluation of one candidate across the trace corpus."""

    policy: Policy
    metric: float
    success_rate: float
    avg_score: float
    delta_from_active: float
    trace_count: int
    log: EvaluationLog
    feedback_success_rate: float
    feedback_outcome_score: float
    decision_stability: float

    def to_dict(self) -> dict[str, Any]:
        rec = self.policy.to_record()
        rec.update(
            {
                "metric": self.metric,
                "success_rate": self.success_rate,
                "avg_score": self.avg_score,
                "delta_from_active": self.delta_from_active,
                "trace_count": self.trace_count,
                "feedback_success_rate": self.feedback_success_rate,
                "feedback_outcome_score": self.feedback_outcome_score,
                "decision_stability": self.decision_stability,
            }
        )
        return rec


def _resolve_strategy(
    candidate: Policy, trace_input: dict[str, Any]
) -> tuple[str, dict[str, float], dict[str, float]]:
    """Determine the strategy type + param overrides for a candidate given trace input.

    For single strategies (Step 23): returns the strategy type with empty overrides.
    For routing strategies (Step 24/25): extracts context from trace input and
    selects via routing rules (priority-based selection), also returning any
    param_overrides from the matched rule.

    Returns:
        A tuple of (strategy_type, param_overrides, applied_overrides).
    """
    from swarm.learning.context import extract_context
    from swarm.learning.routing import select_strategy_with_params

    if not candidate.strategy:
        return DEFAULT_STRATEGY_TYPE, {}, {}

    strat_type = candidate.strategy.get("type")

    if strat_type == "routing":
        context = extract_context(trace_input)
        decision = select_strategy_with_params(context, candidate.strategy)
        return decision.strategy_type, decision.param_overrides, decision.applied_overrides

    if strat_type in STRATEGY_TYPES:
        return str(strat_type), {}, {}

    return DEFAULT_STRATEGY_TYPE, {}, {}


# Replay signature used by the learner for candidate evaluation.
ReplayFn = Callable[
    [dict[str, Any], dict[str, float], dict[str, float], str], dict[str, Any]
]


def _default_replay_fn(
    trace_input: dict[str, Any],
    thresholds: dict[str, float],
    weights: dict[str, float],
    strategy_type: str,
) -> dict[str, Any]:
    """Replay a single trace under pinned thresholds + weights + strategy type.

    Reuses the deterministic, isolated replay engine so candidate metrics have
    full replay parity with the live procurement path. The strategy type is
    pinned via :func:`override_strategy_type` so the evaluation agent applies
    the correct weight transformation.
    """
    from swarm.simulation.replay_engine import run_procurement

    with override_thresholds(thresholds), \
         override_strategy_weights(weights), \
         override_strategy_type(strategy_type):
        result = run_procurement(
            trace_input, adaptive=False, thresholds=thresholds, weights=weights
        )
    return {
        "selected_supplier": result.get("selected_supplier"),
        "score": result.get("score", 0.0),
    }


def _replay_input_from_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Reconstruct the requirement input needed for a replay from events."""
    from swarm.simulation.replay_engine import extract_input

    return extract_input(events)


def _extract_original_decision(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    """Pull the persisted decision (selected supplier + score) from artifacts."""
    for artifact in artifacts:
        if artifact.get("artifact_type") == "result":
            data = artifact.get("data", {})
            ranked = data.get("reasoning", {}).get("ranked", [])
            score = ranked[0].get("score", 0.0) if ranked else 0.0
            return {
                "selected_supplier": data.get("selected_supplier"),
                "score": round(score, 4),
            }
    for artifact in artifacts:
        data = artifact.get("data") or {}
        if "selected_supplier" in data:
            return {
                "selected_supplier": data.get("selected_supplier"),
                "score": data.get("score", 0.0),
            }
    return {"selected_supplier": None, "score": 0.0}


def evaluate_candidate(
    traces: list[dict[str, Any]],
    candidate: Policy,
    active_metric: float,
    replay_fn: ReplayFn | None = None,
) -> EvaluationResult:
    """Evaluate a candidate policy across a set of full traces using the hybrid objective.

    The metric combines ground-truth feedback (primary signal) with decision
    stability (regularizer):

        metric = 0.5 * feedback_success_rate
               + 0.3 * feedback_outcome_score
               + 0.2 * decision_stability

    For each trace:
      * ``feedback_success_rate`` — fraction of traces with ``feedback.success == True``.
      * ``feedback_outcome_score`` — mean of ``feedback.outcome_score``.
      * ``decision_stability`` — fraction of replays that reproduce the originally
        persisted chosen supplier.

    Only traces that have BOTH a persisted decision (selected_supplier in
    artifacts) AND feedback are counted in the denominator.

    Args:
        traces: Full trace dicts (``load_full_trace`` output) with events,
            artifacts, and feedback.
        candidate: The candidate :class:`Policy` to evaluate.
        active_metric: The active policy's metric (for delta logging).
        replay_fn: Injectable replay function for testing. Defaults to the
            real isolated replay engine (:func:`_default_replay_fn`).

    Returns:
        An :class:`EvaluationResult` with the hybrid objective metric and an
        :class:`EvaluationLog` emitted via the module logger.
    """
    if replay_fn is None:
        replay_fn = _default_replay_fn

    matched = 0          # decision_stability: replays matching original
    total = 0            # traces with decision + feedback
    scores: list[float] = []  # replayed scores (secondary, for avg_score field)
    feedback_successes = 0
    outcome_scores: list[float] = []

    for trace in traces:
        events = trace.get("events") or []
        artifacts = trace.get("artifacts") or []
        feedback = trace.get("feedback")
        trace_input = _replay_input_from_events(events)
        if not trace_input:
            continue
        original = _extract_original_decision(artifacts)
        if original.get("selected_supplier") is None:
            continue
        if feedback is None:
            continue

        total += 1
        # Collect feedback signals (ground truth).
        if feedback.get("success"):
            feedback_successes += 1
        if feedback.get("outcome_score") is not None:
            outcome_scores.append(float(feedback.get("outcome_score")))

        # Replay for decision stability + score.
        try:
            strat_type, overrides, applied = _resolve_strategy(
                candidate, trace_input
            )
            if overrides:
                from swarm.learning.routing import apply_param_overrides

                thresholds, weights = apply_param_overrides(
                    candidate.thresholds, candidate.weights, overrides
                )
            else:
                thresholds, weights = candidate.thresholds, candidate.weights
            replayed = replay_fn(trace_input, thresholds, weights, strat_type)
            # Attach applied overrides to replay result for traceability.
            if applied:
                replayed["param_overrides"] = dict(applied)
        except Exception as exc:  # noqa: BLE001 - one bad trace must not abort the batch
            logger.warning(
                "candidate_replay_failed",
                policy_version=candidate.version,
                error=str(exc),
            )
            continue
        if replayed.get("selected_supplier") == original["selected_supplier"]:
            matched += 1
        scores.append(float(replayed.get("score", 0.0)))

    if total == 0:
        feedback_success_rate = 0.0
        feedback_outcome_score = 0.0
        decision_stability = 0.0
        avg_score = 0.0
    else:
        feedback_success_rate = round(feedback_successes / total, 4)
        feedback_outcome_score = (
            round(sum(outcome_scores) / len(outcome_scores), 4)
            if outcome_scores
            else 0.0
        )
        decision_stability = round(matched / total, 4)
        avg_score = round(sum(scores) / total, 4) if scores else 0.0

    metric = compute_hybrid_metric(
        feedback_success_rate, feedback_outcome_score, decision_stability
    )
    delta = round(metric - active_metric, 4)

    # Legacy success_rate = decision_stability for backward compatibility in logs.
    success_rate = decision_stability

    log = EvaluationLog(
        policy_version=candidate.version,
        metric=metric,
        success_rate=success_rate,
        avg_score=avg_score,
        delta_from_active=delta,
        feedback_success_rate=feedback_success_rate,
        feedback_outcome_score=feedback_outcome_score,
        decision_stability=decision_stability,
    )
    logger.info(
        "policy_evaluated",
        policy_version=log.policy_version,
        metric=log.metric,
        success_rate=log.success_rate,
        avg_score=log.avg_score,
        delta_from_active=log.delta_from_active,
        feedback_success_rate=log.feedback_success_rate,
        feedback_outcome_score=log.feedback_outcome_score,
        decision_stability=log.decision_stability,
        trace_count=total,
    )

    return EvaluationResult(
        policy=candidate,
        metric=metric,
        success_rate=success_rate,
        avg_score=avg_score,
        delta_from_active=delta,
        trace_count=total,
        log=log,
        feedback_success_rate=feedback_success_rate,
        feedback_outcome_score=feedback_outcome_score,
        decision_stability=decision_stability,
    )


# --------------------------------------------------------------------------- #
# Trace corpus construction (raw traces + raw feedback merged by the learner)
# --------------------------------------------------------------------------- #


def _merge_feedback_traces(limit: int | None = None) -> list[dict[str, Any]]:
    """Build the eval corpus: full traces that carry feedback.

    The event store is kept dumb — it returns raw feedback rows and raw traces;
    this function merges them: the set of trace IDs with feedback is derived
    from :func:`~swarm.storage.event_store.load_all_feedback`, and each is
    materialised via :func:`~swarm.storage.event_store.load_full_trace`.
    """
    from swarm.storage.event_store import load_all_feedback

    feedback_rows = load_all_feedback()
    # Deterministic ordering: newest feedback first, then trace_id.
    trace_ids = sorted(
        {row["trace_id"] for row in feedback_rows if row.get("trace_id")},
        key=lambda _id: _id,
    )
    if limit is not None:
        trace_ids = trace_ids[-limit:]  # keep the most recent `limit`
    # Materialise deterministically (trace_id order, not insertion).
    traces: list[dict[str, Any]] = []
    for tid in trace_ids:
        traces.append(load_full_trace(tid))
    return traces


# --------------------------------------------------------------------------- #
# Active policy + learning
# --------------------------------------------------------------------------- #

def ensure_baseline_policy() -> Policy:
    """Persist the config-derived baseline policy if none exists in the store.

    On first run (no policies table or no rows), this stores the baseline policy
    derived from config thresholds + balanced strategy weights as a versioned row
    and marks it active. This gives us:

    * A proper policy lineage (baseline is a first-class row).
    * Regression tracking against the original config-derived behavior.
    * A clean rollback target.

    Subsequent calls are idempotent: if any policy exists, this is a no-op.
    """
    baseline = default_policy()
    try:
        record = load_active_policy()
    except Exception:
        record = None
    if record is not None:
        # Something is already active; don't overwrite.
        strat = record.get("strategy")
        if strat is None:
            strat = {"type": DEFAULT_STRATEGY_TYPE}
        return Policy(
            version=record["version"],
            signature=record["signature"],
            thresholds=record["thresholds"],
            weights=record["weights"],
            strategy=strat,
            metric=record.get("metric"),
            success_rate=record.get("success_rate"),
            avg_score=record.get("avg_score"),
            feedback_success_rate=record.get("feedback_success_rate"),
            feedback_outcome_score=record.get("feedback_outcome_score"),
            decision_stability=record.get("decision_stability"),
            active=True,
            created_at=record.get("created_at"),
        )
    # Check if any policy exists at all (baseline or learned).
    from swarm.storage.event_store import load_policy_version_ids

    existing_versions = load_policy_version_ids()
    if existing_versions:
        # A policy exists but none is active — this shouldn't happen in normal
        # operation, but if it does, just return the default policy object.
        return default_policy()
    # Persist the baseline as a versioned, active row.
    store_policy(
        version=baseline.version,
        signature=baseline.signature,
        thresholds=baseline.thresholds,
        weights=baseline.weights,
        metric=0.0,
        success_rate=0.0,
        avg_score=0.0,
        feedback_success_rate=0.0,
        feedback_outcome_score=0.0,
        decision_stability=0.0,
        active=True,
    )
    return default_policy()


def get_active_policy() -> Policy:
    """The currently-active policy (promoted), or the config default."""
    try:
        record = load_active_policy()
    except Exception:
        record = None
    if record is None:
        return default_policy()
    strat = record.get("strategy")
    if strat is None:
        strat = {"type": DEFAULT_STRATEGY_TYPE}
    return Policy(
        version=record["version"],
        signature=record["signature"],
        thresholds=record["thresholds"],
        weights=record["weights"],
        strategy=strat,
        metric=record.get("metric"),
        success_rate=record.get("success_rate"),
        avg_score=record.get("avg_score"),
        feedback_success_rate=record.get("feedback_success_rate"),
        feedback_outcome_score=record.get("feedback_outcome_score"),
        decision_stability=record.get("decision_stability"),
        active=True,
        created_at=record.get("created_at"),
    )


def learn_candidates(
    replay_fn: ReplayFn | None = None,
    trace_limit: int | None = None,
) -> dict[str, Any]:
    """Run the closed-loop policy search and return the best candidate.

    Pipeline (v1.1 Step 22):

      1. Gather traces that carry feedback (merge of raw traces + raw feedback).
      2. If fewer than ``MIN_TRACES_FOR_LEARNING`` qualify → return
         ``{"status": "insufficient_data"}`` (no candidate produced).
      3. Generate the deterministic grid neighbourhood of the *active* policy.
      4. Evaluate every candidate (replays pinned via the override ContextVars),
         logging ``policy_version, metric, success_rate, avg_score,
         delta_from_active`` for each.
      5. Reject unsafe candidates (threshold floor / weight band guardrail).
      6. Persist candidates (idempotent on version) and return the best.

    Idempotency: the grid, replay and objective are pure → the same trace set
    always yields the same best ``version``. Already-stored versions are
    preserved (``INSERT OR IGNORE``).

    Args:
        replay_fn: Injectable replay function (testing). Defaults to the real
            isolated replay engine.
        trace_limit: Optional cap on traces considered (testing).

    Returns:
        A dict with ``status`` (``"ok"`` / ``"insufficient_data"`` /
        ``"no_candidates"``), the best candidate's ``version``/``metric``/
        ``success_rate``/``avg_score`` / ``feedback_success_rate`` /
        ``feedback_outcome_score`` / ``decision_stability`` and the ranked
        ``candidates`` list.
    """
    traces = _merge_feedback_traces(limit=trace_limit)
    evalable = [
        t
        for t in traces
        if _replay_input_from_events(t.get("events") or [])
        and _extract_original_decision(t.get("artifacts") or []).get("selected_supplier")
        is not None
        and t.get("feedback") is not None
    ]

    if len(evalable) < MIN_TRACES_FOR_LEARNING:
        logger.info(
            "policy_learning_insufficient_data",
            traces_available=len(traces),
            evalable_traces=len(evalable),
            required=MIN_TRACES_FOR_LEARNING,
        )
        return {
            "status": "insufficient_data",
            "traces_available": len(traces),
            "evalable_traces": len(evalable),
            "required": MIN_TRACES_FOR_LEARNING,
            "best": None,
            "candidates": [],
        }

    active = ensure_baseline_policy()
    active_metric = active.metric if active.metric is not None else 0.0

    candidates = generate_candidates(active)
    results: list[EvaluationResult] = []
    for candidate in candidates:
        if not is_safe_policy(candidate):
            logger.warning(
                "candidate_rejected_unsafe",
                policy_version=candidate.version,
                thresholds=candidate.thresholds,
                weights=candidate.weights,
            )
            continue

        # Idempotent: skip re-evaluation of already-stored, already-scored
        # candidates (same traces -> same best version is guaranteed).
        existing = _try_load_stored(candidate.version)
        if existing is not None and existing.get("metric") is not None:
            fs = existing.get("feedback_success_rate") or 0.0
            fo = existing.get("feedback_outcome_score") or 0.0
            ds = existing.get("decision_stability") or 0.0
            result = EvaluationResult(
                policy=candidate,
                metric=existing["metric"],
                success_rate=existing.get("success_rate") or 0.0,
                avg_score=existing.get("avg_score") or 0.0,
                delta_from_active=round(existing["metric"] - active_metric, 4),
                trace_count=len(evalable),
                log=EvaluationLog(
                    policy_version=candidate.version,
                    metric=existing["metric"],
                    success_rate=existing.get("success_rate") or 0.0,
                    avg_score=existing.get("avg_score") or 0.0,
                    delta_from_active=round(existing["metric"] - active_metric, 4),
                    feedback_success_rate=fs,
                    feedback_outcome_score=fo,
                    decision_stability=ds,
                ),
                feedback_success_rate=fs,
                feedback_outcome_score=fo,
                decision_stability=ds,
            )
            logger.info(
                "policy_evaluated",
                policy_version=candidate.version,
                metric=existing["metric"],
                success_rate=existing.get("success_rate"),
                avg_score=existing.get("avg_score"),
                delta_from_active=round(existing["metric"] - active_metric, 4),
                trace_count=len(evalable),
            )
            results.append(result)
            continue

        result = evaluate_candidate(evalable, candidate, active_metric, replay_fn=replay_fn)
        store_policy(
            version=candidate.version,
            signature=candidate.signature,
            thresholds=candidate.thresholds,
            weights=candidate.weights,
            strategy=candidate.strategy,
            metric=result.metric,
            success_rate=result.success_rate,
            avg_score=result.avg_score,
            feedback_success_rate=result.feedback_success_rate,
            feedback_outcome_score=result.feedback_outcome_score,
            decision_stability=result.decision_stability,
            active=False,
        )
        results.append(result)

    if not results:
        return {
            "status": "no_candidates",
            "traces_available": len(traces),
            "evalable_traces": len(evalable),
            "best": None,
            "candidates": [],
        }

    # Best = highest metric; tie-break by version (deterministic).
    results.sort(key=lambda r: (-r.metric, r.policy.version))
    best = results[0]
    return {
        "status": "ok",
        "traces_available": len(traces),
        "evalable_traces": len(evalable),
        "active_version": active.version,
        "active_metric": active_metric,
        "best": {
            "version": best.policy.version,
            "metric": best.metric,
            "success_rate": best.success_rate,
            "avg_score": best.avg_score,
            "delta_from_active": best.delta_from_active,
            "trace_count": best.trace_count,
            "feedback_success_rate": best.feedback_success_rate,
            "feedback_outcome_score": best.feedback_outcome_score,
            "decision_stability": best.decision_stability,
        },
        "candidates": [r.to_dict() for r in results],
    }


# --------------------------------------------------------------------------- #
# Promotion (atomic, safe)
# --------------------------------------------------------------------------- #


def _try_load_stored(version: str) -> dict[str, Any] | None:
    """Look up a stored policy; swallow errors (no DB in some tests)."""
    try:
        return load_policy(version)
    except Exception:
        return None


def can_promote(candidate: Policy, active: Policy) -> tuple[bool, str]:
    """Check the Step 22 promotion rule against the active policy.

    Promotion requires BOTH:

      * ``metric > active.metric + IMPROVEMENT_MARGIN``  (strict improvement)
      * ``decision_stability >= active.decision_stability`` (no reliability loss)

    The active policy's metric/stability default to 0.0 when unscored.

    Note: ``success_rate`` on the Policy dataclass is the legacy backward-compat
    field (set to decision_stability during evaluation). The promotion rule uses
    ``decision_stability`` for the reliability guardrail.
    """
    active_metric = active.metric if active.metric is not None else 0.0
    cand_metric = candidate.metric if candidate.metric is not None else 0.0

    # Use decision_stability as the reliability guardrail. For backward
    # compatibility with policies loaded from older DB rows that may not have
    # decision_stability, fall back to success_rate.
    active_stability = getattr(active, "decision_stability", None)
    if active_stability is None:
        active_stability = active.success_rate if active.success_rate is not None else 0.0
    cand_stability = getattr(candidate, "decision_stability", None)
    if cand_stability is None:
        cand_stability = candidate.success_rate if candidate.success_rate is not None else 0.0

    if not (cand_metric > active_metric + IMPROVEMENT_MARGIN):
        return False, "fails_improvement_margin"
    if not (cand_stability >= active_stability):
        return False, "regression_in_success_rate"
    return True, "ok"


def promote_policy(version: str, *, force: bool = False) -> dict[str, Any]:
    """Atomically promote a stored candidate to the active policy.

    The promotion rule (v1.1 Step 22):

      * ``metric > active.metric + IMPROVEMENT_MARGIN``
      * ``decision_stability >= active.decision_stability``

    Safety:

      * Rejects with ``already_promoted`` if ``version`` is already active — the
        operation is atomic and never silently no-ops.
      * Rejects unsafe policies (threshold floor / weight band) that somehow
        reached the store.

    When ``force=True``, the strict-improvement rule is bypassed, allowing
    rollback to a previously-promoted (or baseline) policy regardless of its
    metric relative to the current active policy. This is the rollback path.

    Implemented as a single transaction in the store
    (:func:`~swarm.storage.event_store.set_policy_active`) so concurrent
    promote requests are serialized and only one policy can ever be active
    (partial unique constraint on ``policies.active``).

    Args:
        version: The candidate policy version to promote.
        force: If True, bypass the improvement margin + reliability rule
            (rollback mode). Safety and already_promoted checks still apply.

    Returns:
        A dict with ``status`` (``"promoted"`` / ``"rejected"``), ``version``
        and a human-readable ``reason`` when rejected.
    """
    record = _try_load_stored(version)
    if record is None:
        return {"status": "rejected", "version": version, "reason": "candidate_not_found"}

    candidate = Policy(
        version=record["version"],
        signature=record["signature"],
        thresholds=record["thresholds"],
        weights=record["weights"],
        strategy=record.get("strategy") or {"type": DEFAULT_STRATEGY_TYPE},
        metric=record["metric"],
        success_rate=record["success_rate"],
        avg_score=record["avg_score"],
        feedback_success_rate=record.get("feedback_success_rate"),
        feedback_outcome_score=record.get("feedback_outcome_score"),
        decision_stability=record.get("decision_stability"),
        active=bool(record["active"]),
        created_at=record.get("created_at"),
    )

    if not is_safe_policy(candidate):
        return {"status": "rejected", "version": version, "reason": "unsafe_policy"}

    if candidate.active and not force:
        # Already the active policy — reject atomically (no silent no-op).
        return {"status": "rejected", "version": version, "reason": "already_promoted"}

    if not force:
        active = get_active_policy()
        ok, reason = can_promote(candidate, active)
        if not ok:
            return {"status": "rejected", "version": version, "reason": reason}

    activated = set_policy_active(version)
    if not activated:
        # Lost the race / vanished between load and commit.
        return {"status": "rejected", "version": version, "reason": "not_found"}
    logger.info("policy_promoted", version=version, metric=candidate.metric, force=force)
    return {
        "status": "promoted",
        "version": version,
        "metric": candidate.metric,
        "success_rate": candidate.success_rate,
        "avg_score": candidate.avg_score,
    }
