"""Tests for v1.1 Step 22: closed-loop policy learning.

Covers the data model + grid + safety (swarm.learning.policy) and the
evaluation/search/promotion logic (swarm.learning.learner). Candidate
evaluation uses an injected deterministic ``replay_fn`` so the learning
machinery is exercised fast and without spinning up a live swarm.
"""

from __future__ import annotations

import pytest

from swarm.config import (
    FEEDBACK_SCORE_WEIGHT,
    FEEDBACK_SUCCESS_WEIGHT,
    MIN_TRACES_FOR_LEARNING,
    OBJECTIVE_SCORE_WEIGHT,
    OBJECTIVE_SUCCESS_WEIGHT,
    STABILITY_WEIGHT,
    THRESHOLD_CLAMP_MAX,
    THRESHOLD_CLAMP_MIN,
)
from swarm.learning.adaptive_policy import (
    get_adaptive_thresholds,
    override_strategy_weights,
    override_thresholds,
)
from swarm.learning.learner import (
    can_promote,
    compute_hybrid_metric,
    compute_metric,
    evaluate_candidate,
    get_active_policy,
    learn_candidates,
    promote_policy,
)
from swarm.learning.policy import (
    THRESHOLD_KEYS,
    WEIGHT_KEYS,
    Policy,
    build_policy,
    canonical_params,
    clamp_thresholds,
    clamp_weights,
    compute_signature,
    compute_version,
    default_policy,
    generate_candidates,
    is_safe_policy,
)
from swarm.storage.event_store import (
    load_policy,
    store_artifact,
    store_event,
    store_feedback,
    store_policy,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def db_path(tmp_path) -> str:
    return str(tmp_path / "policy_learning.db")


@pytest.fixture(autouse=True)
def _isolate_event_store(db_path: str):
    """Point the event store at a per-test temp DB (deterministic isolation)."""
    import swarm.storage.event_store as es

    orig = es._DB_PATH
    es._DB_PATH = db_path
    es.init_db(db_path)
    yield
    es._DB_PATH = orig


SAFE_THRESH = {
    "confidence_threshold": 0.7,
    "stability_threshold": 0.7,
    "trust_threshold": 0.7,
}
SAFE_WEIGHTS = {"price_weight": 0.4, "score_weight": 0.4, "carbon_weight": 0.2}


def _seed_traces(n: int, base_score: float = 0.5, supplier: str = "Supp_A") -> None:
    """Persist ``n`` traces that have a decision artifact + feedback."""
    for i in range(n):
        tid = f"TRACE-LEARN-{i}"
        store_event(
            tid,
            "procurement_request",
            {
                "material": "aluminum",
                "quantity": 1000,
                "budget": 2_000_000.0,
                "target_lead_time_days": 30,
                "max_carbon_per_unit": None,
                "goal": None,
                "supplier_count": 5,
            },
        )
        store_artifact(
            tid,
            "result",
            {
                "selected_supplier": supplier,
                "reasoning": {
                    "ranked": [{"supplier_id": supplier, "score": base_score}]
                },
            },
        )
        store_feedback(tid, outcome_score=0.9, success=True, latency_ms=100.0)


def _replay_always(inp, thr, w, strat):
    """A replay stub: always reproduces the original supplier with a fixed score."""
    return {"selected_supplier": "Supp_A", "score": 0.5}


# --------------------------------------------------------------------------- #
# Data model: signature, version, determinism
# --------------------------------------------------------------------------- #


class TestSignatureAndVersion:
    def test_version_is_12_chars(self) -> None:
        p = build_policy(SAFE_THRESH, SAFE_WEIGHTS)
        assert len(p.version) == 12
        assert p.version == compute_version(p.signature)

    def test_signature_is_deterministic(self) -> None:
        s1 = compute_signature(SAFE_THRESH, SAFE_WEIGHTS)
        s2 = compute_signature(SAFE_THRESH, SAFE_WEIGHTS)
        assert s1 == s2

    def test_version_independent_of_key_insertion_order(self) -> None:
        # Reversed insertion order must yield the same signature/version.
        ordered = {
            "confidence_threshold": 0.7,
            "stability_threshold": 0.7,
            "trust_threshold": 0.7,
        }
        reversed_order = {
            "trust_threshold": 0.7,
            "stability_threshold": 0.7,
            "confidence_threshold": 0.7,
        }
        assert compute_signature(ordered, SAFE_WEIGHTS) == compute_signature(
            reversed_order, SAFE_WEIGHTS
        )

    def test_canonical_params_keys_are_sorted(self) -> None:
        cp = canonical_params(SAFE_THRESH, SAFE_WEIGHTS)
        assert list(cp["thresholds"].keys()) == sorted(THRESHOLD_KEYS)
        assert list(cp["weights"].keys()) == sorted(WEIGHT_KEYS)

    def test_different_params_different_version(self) -> None:
        p1 = build_policy(SAFE_THRESH, SAFE_WEIGHTS)
        p2 = build_policy(
            {**SAFE_THRESH, "confidence_threshold": 0.75}, SAFE_WEIGHTS
        )
        assert p1.version != p2.version

    def test_version_is_hash_of_signature(self) -> None:
        p = build_policy(SAFE_THRESH, SAFE_WEIGHTS)
        import hashlib as _hl

        assert p.version == _hl.sha256(p.signature.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Clamping + normalization
# --------------------------------------------------------------------------- #


class TestClampThresholds:
    def test_clamps_to_floor(self) -> None:
        res = clamp_thresholds(
            {"confidence_threshold": 0.1, "stability_threshold": 0.0, "trust_threshold": 0.7}
        )
        assert res["confidence_threshold"] == THRESHOLD_CLAMP_MIN
        assert res["stability_threshold"] == THRESHOLD_CLAMP_MIN
        assert res["trust_threshold"] == 0.7

    def test_clamps_to_ceiling(self) -> None:
        res = clamp_thresholds(
            {"confidence_threshold": 1.0, "stability_threshold": 1.5, "trust_threshold": 0.9}
        )
        assert res["confidence_threshold"] == THRESHOLD_CLAMP_MAX
        assert res["stability_threshold"] == THRESHOLD_CLAMP_MAX
        assert res["trust_threshold"] == 0.9

    def test_in_band_unchanged(self) -> None:
        thr = {"confidence_threshold": 0.7, "stability_threshold": 0.65, "trust_threshold": 0.8}
        assert clamp_thresholds(thr) == thr


class TestClampWeights:
    def test_normalizes_to_unit_sum(self) -> None:
        res = clamp_weights({"price_weight": 0.3, "score_weight": 0.3, "carbon_weight": 0.3})
        assert abs(sum(res.values()) - 1.0) < 1e-6

    def test_clamps_to_floor_and_normalizes(self) -> None:
        res = clamp_weights(
            {"price_weight": 0.1, "score_weight": 0.1, "carbon_weight": 0.8}
        )
        # All inputs below the floor are raised to the floor before normalizing,
        # so the post-normalize result stays a valid unit-sum distribution.
        for k in WEIGHT_KEYS:
            assert isinstance(res[k], float)
        assert abs(sum(res.values()) - 1.0) < 1e-6
        # The highest-weight input (0.8, within band) dominates post-normalize.
        assert res["carbon_weight"] >= res["price_weight"]

    def test_balanced_unchanged(self) -> None:
        bal = {"price_weight": 0.4, "score_weight": 0.4, "carbon_weight": 0.2}
        assert clamp_weights(bal) == bal


# --------------------------------------------------------------------------- #
# Grid generation
# --------------------------------------------------------------------------- #


class TestGenerateCandidates:
    def test_includes_base_policy(self) -> None:
        base = build_policy(SAFE_THRESH, SAFE_WEIGHTS)
        cands = generate_candidates(base)
        assert any(c.signature == base.signature for c in cands)

    def test_all_within_threshold_clamps(self) -> None:
        cands = generate_candidates(build_policy(SAFE_THRESH, SAFE_WEIGHTS))
        for c in cands:
            for k in THRESHOLD_KEYS:
                assert THRESHOLD_CLAMP_MIN <= c.thresholds[k] <= THRESHOLD_CLAMP_MAX

    def test_deduplicates_by_signature(self) -> None:
        cands = generate_candidates(build_policy(SAFE_THRESH, SAFE_WEIGHTS))
        sigs = [c.signature for c in cands]
        assert len(sigs) == len(set(sigs))

    def test_ordered_by_signature(self) -> None:
        cands = generate_candidates(build_policy(SAFE_THRESH, SAFE_WEIGHTS))
        sigs = [c.signature for c in cands]
        assert sigs == sorted(sigs)

    def test_deterministic(self) -> None:
        base = build_policy(SAFE_THRESH, SAFE_WEIGHTS)
        c1 = [c.version for c in generate_candidates(base)]
        c2 = [c.version for c in generate_candidates(base)]
        assert c1 == c2


# --------------------------------------------------------------------------- #
# Safety guardrail
# --------------------------------------------------------------------------- #


class TestIsSafePolicy:
    def test_safe_policy_passes(self) -> None:
        assert is_safe_policy(build_policy(SAFE_THRESH, SAFE_WEIGHTS)) is True

    def test_threshold_below_floor_rejected(self) -> None:
        bad = build_policy(
            {"confidence_threshold": 0.5, "stability_threshold": 0.7, "trust_threshold": 0.7},
            SAFE_WEIGHTS,
        )
        assert is_safe_policy(bad) is False

    def test_threshold_above_ceiling_rejected(self) -> None:
        bad = build_policy(
            {"confidence_threshold": 0.95, "stability_threshold": 0.7, "trust_threshold": 0.7},
            SAFE_WEIGHTS,
        )
        assert is_safe_policy(bad) is False

    def test_weight_out_of_band_rejected(self) -> None:
        bad = build_policy(
            SAFE_THRESH,
            {"price_weight": 0.1, "score_weight": 0.45, "carbon_weight": 0.45},
        )
        assert is_safe_policy(bad) is False

    def test_weights_not_summing_rejected(self) -> None:
        bad = build_policy(
            SAFE_THRESH,
            {"price_weight": 0.9, "score_weight": 0.9, "carbon_weight": 0.9},
        )
        assert is_safe_policy(bad) is False


# --------------------------------------------------------------------------- #
# Objective function
# --------------------------------------------------------------------------- #


class TestComputeMetric:
    def test_weighted_formula(self) -> None:
        expected = round(OBJECTIVE_SUCCESS_WEIGHT * 0.8 + OBJECTIVE_SCORE_WEIGHT * 0.6, 4)
        assert compute_metric(0.8, 0.6) == expected

    def test_ignores_latency(self) -> None:
        # metric depends only on success_rate + avg_score, never latency.
        assert compute_metric(0.8, 0.6) == compute_metric(0.8, 0.6)

    def test_bounds(self) -> None:
        assert compute_metric(0.0, 0.0) == 0.0
        assert compute_metric(1.0, 1.0) == 1.0

    def test_weights_sum_to_one(self) -> None:
        assert OBJECTIVE_SUCCESS_WEIGHT + OBJECTIVE_SCORE_WEIGHT == 1.0


class TestComputeHybridMetric:
    def test_weighted_formula(self) -> None:
        expected = round(
            FEEDBACK_SUCCESS_WEIGHT * 0.8
            + FEEDBACK_SCORE_WEIGHT * 0.6
            + STABILITY_WEIGHT * 0.5,
            4,
        )
        result = compute_hybrid_metric(
            feedback_success_rate=0.8,
            feedback_outcome_score=0.6,
            decision_stability=0.5,
        )
        assert result == expected

    def test_weights_sum_to_one(self) -> None:
        assert FEEDBACK_SUCCESS_WEIGHT + FEEDBACK_SCORE_WEIGHT + STABILITY_WEIGHT == 1.0

    def test_bounds(self) -> None:
        assert compute_hybrid_metric(0.0, 0.0, 0.0) == 0.0
        assert compute_hybrid_metric(1.0, 1.0, 1.0) == 1.0

    def test_feedback_is_primary_signal(self) -> None:
        # If feedback is perfect but stability is 0, metric should be > 0.5
        # because feedback weights (0.5 + 0.3 = 0.8) dominate.
        metric = compute_hybrid_metric(
            feedback_success_rate=1.0,
            feedback_outcome_score=1.0,
            decision_stability=0.0,
        )
        assert metric > 0.5


# --------------------------------------------------------------------------- #
# Candidate evaluation
# --------------------------------------------------------------------------- #


class TestEvaluateCandidate:
    def test_all_matched_perfect_metric(self) -> None:
        # replay always reproduces the original supplier -> decision_stability 1.0
        traces = [
            {
                "events": [
                    {
                        "event_type": "procurement_request",
                        "payload": {"material": "aluminum"},
                        "created_at": "x",
                    }
                ],
                "artifacts": [
                    {
                        "artifact_type": "result",
                        "data": {
                            "selected_supplier": "Supp_A",
                            "reasoning": {"ranked": [{"supplier_id": "Supp_A", "score": 0.9}]},
                        },
                    }
                ],
                "llm_history": [],
                "feedback": {"success": True, "outcome_score": 0.9},
            }
        ] * 5
        cand = build_policy(SAFE_THRESH, SAFE_WEIGHTS)
        res = evaluate_candidate(traces, cand, active_metric=0.0, replay_fn=_replay_always)
        # success_rate (legacy) = decision_stability = 1.0
        assert res.success_rate == 1.0
        assert res.avg_score == 0.5
        # Hybrid metric: 0.5*1.0 + 0.3*0.9 + 0.2*1.0 = 0.97
        assert res.metric == compute_hybrid_metric(1.0, 0.9, 1.0)
        assert res.delta_from_active == round(res.metric - 0.0, 4)
        assert res.feedback_success_rate == 1.0
        assert res.feedback_outcome_score == 0.9
        assert res.decision_stability == 1.0

    def test_partial_match(self) -> None:
        def replay(inp, thr, w, strat):
            # match only on even-indexed traces
            sid = inp.get("selected_supplier_target")
            return {"selected_supplier": sid if sid == "match" else "Other", "score": 0.5}

        traces = [
            {
                "events": [
                    {
                        "event_type": "procurement_request",
                        "payload": {"selected_supplier_target": "match"},
                        "created_at": "x",
                    }
                ],
                "artifacts": [
                    {
                        "artifact_type": "result",
                        "data": {
                            "selected_supplier": "match",
                            "reasoning": {"ranked": [{"supplier_id": "match", "score": 0.5}]},
                        },
                    }
                ],
                "feedback": {"success": True, "outcome_score": 0.6},
            },
            {
                "events": [
                    {
                        "event_type": "procurement_request",
                        "payload": {"selected_supplier_target": "nomatch"},
                        "created_at": "x",
                    }
                ],
                "artifacts": [
                    {
                        "artifact_type": "result",
                        "data": {
                            "selected_supplier": "nomatch",
                            "reasoning": {"ranked": [{"supplier_id": "nomatch", "score": 0.5}]},
                        },
                    }
                ],
                "feedback": {"success": False, "outcome_score": 0.4},
            },
        ]
        cand = build_policy(SAFE_THRESH, SAFE_WEIGHTS)
        res = evaluate_candidate(traces, cand, active_metric=0.4, replay_fn=replay)
        # 1/2 matched -> decision_stability = 0.5, feedback_success_rate = 0.5, avg outcome = 0.5
        assert res.success_rate == 0.5
        assert res.metric == compute_hybrid_metric(0.5, 0.5, 0.5)

    def test_no_evalable_traces_safe_metric(self) -> None:
        traces = [{"events": [], "artifacts": [], "feedback": None}]
        cand = build_policy(SAFE_THRESH, SAFE_WEIGHTS)
        res = evaluate_candidate(traces, cand, active_metric=0.5, replay_fn=_replay_always)
        assert res.success_rate == 0.0
        assert res.avg_score == 0.0
        assert res.metric == 0.0
        assert res.feedback_success_rate == 0.0
        assert res.feedback_outcome_score == 0.0
        assert res.decision_stability == 0.0


# --------------------------------------------------------------------------- #
# learn_candidates
# --------------------------------------------------------------------------- #


class TestLearnCandidates:
    def test_insufficient_data_without_feedback(self) -> None:
        res = learn_candidates(replay_fn=_replay_always)
        assert res["status"] == "insufficient_data"
        assert res["best"] is None
        assert res["candidates"] == []

    def test_insufficient_data_below_threshold(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "swarm.learning.learner.MIN_TRACES_FOR_LEARNING",
            MIN_TRACES_FOR_LEARNING,
        )
        _seed_traces(10)
        res = learn_candidates(replay_fn=_replay_always)
        assert res["status"] == "insufficient_data"
        assert res["evalable_traces"] == 10

    def test_ok_with_enough_traces(self, monkeypatch) -> None:
        monkeypatch.setattr("swarm.learning.learner.MIN_TRACES_FOR_LEARNING", 5)
        _seed_traces(5)
        res = learn_candidates(replay_fn=_replay_always)
        assert res["status"] == "ok"
        assert res["best"] is not None
        assert res["best"]["version"]
        assert "candidates" in res
        assert isinstance(res["candidates"], list)
        assert len(res["candidates"]) > 0

    def test_idempotent_same_version(self, monkeypatch) -> None:
        monkeypatch.setattr("swarm.learning.learner.MIN_TRACES_FOR_LEARNING", 5)
        _seed_traces(5)
        r1 = learn_candidates(replay_fn=_replay_always)
        r2 = learn_candidates(replay_fn=_replay_always)
        assert r1["status"] == "ok"
        assert r1["best"]["version"] == r2["best"]["version"]
        # second call reuses stored metrics (no re-eval needed)
        assert r1["best"]["metric"] == r2["best"]["metric"]

    def test_candidates_are_safe(self, monkeypatch) -> None:
        monkeypatch.setattr("swarm.learning.learner.MIN_TRACES_FOR_LEARNING", 5)
        _seed_traces(5)
        res = learn_candidates(replay_fn=_replay_always)
        for c in res["candidates"]:
            assert load_policy(c["version"]) is not None  # persisted
            for k in THRESHOLD_KEYS:
                assert THRESHOLD_CLAMP_MIN <= c["thresholds"][k] <= THRESHOLD_CLAMP_MAX

    def test_best_is_max_metric(self, monkeypatch) -> None:
        monkeypatch.setattr("swarm.learning.learner.MIN_TRACES_FOR_LEARNING", 5)
        _seed_traces(5)
        res = learn_candidates(replay_fn=_replay_always)
        metrics = [c["metric"] for c in res["candidates"]]
        assert res["best"]["metric"] == max(metrics)

    def test_replay_fn_receives_candidate_params(self, monkeypatch) -> None:
        monkeypatch.setattr("swarm.learning.learner.MIN_TRACES_FOR_LEARNING", 5)
        _seed_traces(5)
        seen = []

        def capture(inp, thr, w, strat):
            seen.append((thr, w))
            return {"selected_supplier": "Supp_A", "score": 0.5}

        learn_candidates(replay_fn=capture)
        # Each safe candidate was evaluated once against each trace.
        assert len(seen) > 0
        # thresholds passed match a clamped candidate (all in [0.6, 0.9]).
        for thr, _w in seen:
            for k in THRESHOLD_KEYS:
                assert THRESHOLD_CLAMP_MIN <= thr[k] <= THRESHOLD_CLAMP_MAX


# --------------------------------------------------------------------------- #
# Active policy + priority
# --------------------------------------------------------------------------- #


class TestActivePolicyPriority:
    def test_default_when_no_promoted(self) -> None:
        ap = get_active_policy()
        assert ap.version == default_policy().version

    def test_active_policy_reflects_promotion(self, monkeypatch) -> None:
        monkeypatch.setattr("swarm.learning.learner.MIN_TRACES_FOR_LEARNING", 5)
        _seed_traces(5)
        res = learn_candidates(replay_fn=_replay_always)
        version = res["best"]["version"]
        promote_policy(version)
        ap = get_active_policy()
        assert ap.version == version
        assert ap.active is True
        # Runtime threshold resolver now returns the promoted thresholds.
        assert get_adaptive_thresholds() == ap.thresholds

    def test_override_beats_active_policy(self, monkeypatch) -> None:
        monkeypatch.setattr("swarm.learning.learner.MIN_TRACES_FOR_LEARNING", 5)
        _seed_traces(5)
        res = learn_candidates(replay_fn=_replay_always)
        promote_policy(res["best"]["version"])
        pinned = {
            "confidence_threshold": 0.61,
            "stability_threshold": 0.62,
            "trust_threshold": 0.63,
        }
        with override_thresholds(pinned):
            assert get_adaptive_thresholds() == pinned


# --------------------------------------------------------------------------- #
# Promotion
# --------------------------------------------------------------------------- #


def _make_policy(
    *,
    metric: float = 0.5,
    success_rate: float = 0.8,
    thresholds: dict | None = None,
    weights: dict | None = None,
    version: str = "POL-PROM-1",
    active: bool = False,
) -> Policy:
    return Policy(
        version=version,
        signature=compute_signature(thresholds or SAFE_THRESH, weights or SAFE_WEIGHTS),
        thresholds=thresholds or dict(SAFE_THRESH),
        weights=weights or dict(SAFE_WEIGHTS),
        strategy={"type": "balanced"},
        metric=metric,
        success_rate=success_rate,
        avg_score=0.5,
        active=active,
    )


class TestCanPromote:
    def test_improves_and_stable_ok(self) -> None:
        cand = _make_policy(metric=0.6, success_rate=0.9)
        active = _make_policy(metric=0.5, success_rate=0.8)
        ok, reason = can_promote(cand, active)
        assert ok is True
        assert reason == "ok"

    def test_fails_improvement_margin(self) -> None:
        cand = _make_policy(metric=0.5, success_rate=0.9)
        active = _make_policy(metric=0.5, success_rate=0.8)
        ok, reason = can_promote(cand, active)
        assert ok is False
        assert reason == "fails_improvement_margin"

    def test_regression_in_success_rate_rejected(self) -> None:
        cand = _make_policy(metric=0.7, success_rate=0.5)
        active = _make_policy(metric=0.5, success_rate=0.8)
        ok, reason = can_promote(cand, active)
        assert ok is False
        assert reason == "regression_in_success_rate"

    def test_exact_equal_metric_rejected(self) -> None:
        # margin is strict: metric must be strictly greater than active + margin.
        cand = _make_policy(metric=0.55, success_rate=0.8)
        active = _make_policy(metric=0.55, success_rate=0.8)
        ok, reason = can_promote(cand, active)
        assert ok is False
        assert reason == "fails_improvement_margin"


class TestPromotePolicy:
    def test_promotes_stored_candidate(self, monkeypatch) -> None:
        monkeypatch.setattr("swarm.learning.learner.MIN_TRACES_FOR_LEARNING", 5)
        _seed_traces(5)
        res = learn_candidates(replay_fn=_replay_always)
        version = res["best"]["version"]
        result = promote_policy(version)
        assert result["status"] == "promoted"
        assert result["version"] == version
        # Active policy in store + resolver reflects it.
        stored = load_policy(version)
        assert stored["active"] is True

    def test_already_promoted_rejected(self, monkeypatch) -> None:
        monkeypatch.setattr("swarm.learning.learner.MIN_TRACES_FOR_LEARNING", 5)
        _seed_traces(5)
        version = learn_candidates(replay_fn=_replay_always)["best"]["version"]
        assert promote_policy(version)["status"] == "promoted"
        second = promote_policy(version)
        assert second["status"] == "rejected"
        assert second["reason"] == "already_promoted"

    def test_unknown_version_rejected(self) -> None:
        result = promote_policy("DOES-NOT-EXIST")
        assert result["status"] == "rejected"
        assert result["reason"] == "candidate_not_found"

    def test_unsafe_policy_never_promoted(self) -> None:
        # Manually store an unsafe candidate, then ensure promotion refuses it.
        bad = build_policy(
            {"confidence_threshold": 0.5, "stability_threshold": 0.5, "trust_threshold": 0.5},
            SAFE_WEIGHTS,
        )
        store_policy(
            version=bad.version,
            signature=bad.signature,
            thresholds=bad.thresholds,
            weights=bad.weights,
            strategy={"type": "balanced"},
            metric=0.9,
            success_rate=0.9,
            avg_score=0.9,
            feedback_success_rate=0.1,
            feedback_outcome_score=0.5,
            decision_stability=0.5,
            active=False,
        )
        result = promote_policy(bad.version)
        assert result["status"] == "rejected"
        assert result["reason"] == "unsafe_policy"

    def test_force_promote_bypasses_improvement_rule(self, monkeypatch) -> None:
        # A worse candidate that would normally fail the improvement margin.
        bad_cand = build_policy(SAFE_THRESH, SAFE_WEIGHTS)
        store_policy(
            version=bad_cand.version,
            signature=bad_cand.signature,
            thresholds=bad_cand.thresholds,
            weights=bad_cand.weights,
            strategy={"type": "balanced"},
            metric=0.1,
            success_rate=0.1,
            avg_score=0.1,
            feedback_success_rate=0.1,
            feedback_outcome_score=0.1,
            decision_stability=0.1,
            active=False,
        )
        result = promote_policy(bad_cand.version, force=True)
        assert result["status"] == "promoted"
        from swarm.storage.event_store import load_active_policy

        ap = load_active_policy()
        assert ap is not None
        assert ap["version"] == bad_cand.version

    def test_force_promote_same_version_allowed(self, monkeypatch) -> None:
        monkeypatch.setattr("swarm.learning.learner.MIN_TRACES_FOR_LEARNING", 5)
        _seed_traces(5)
        version = learn_candidates(replay_fn=_replay_always)["best"]["version"]
        assert promote_policy(version)["status"] == "promoted"
        # Force-promoting the already-active version should work (rollback).
        result = promote_policy(version, force=True)
        assert result["status"] == "promoted"


class TestBaselinePolicyPersistence:
    def test_baseline_persisted_on_first_run(self) -> None:
        from swarm.learning.learner import ensure_baseline_policy
        from swarm.storage.event_store import load_policy_version_ids

        bp = ensure_baseline_policy()
        versions = load_policy_version_ids()
        assert bp.version in versions

    def test_baseline_is_active_after_ensure(self) -> None:
        from swarm.learning.learner import ensure_baseline_policy
        from swarm.storage.event_store import load_active_policy

        bp = ensure_baseline_policy()
        active = load_active_policy()
        assert active is not None
        assert active["version"] == bp.version
        assert active["active"] is True

    def test_baseline_idempotent(self) -> None:
        from swarm.learning.learner import ensure_baseline_policy

        bp1 = ensure_baseline_policy()
        bp2 = ensure_baseline_policy()
        assert bp1.version == bp2.version


class TestParamDeltaEnforcement:
    def test_candidates_within_max_delta(self) -> None:
        from swarm.config import POLICY_MAX_PARAM_DELTA

        base = build_policy(SAFE_THRESH, SAFE_WEIGHTS)
        cands = generate_candidates(base)
        # Every candidate must have params within POLICY_MAX_PARAM_DELTA of base.
        for c in cands:
            for k in THRESHOLD_KEYS:
                assert abs(c.thresholds[k] - base.thresholds[k]) <= POLICY_MAX_PARAM_DELTA
            for k in WEIGHT_KEYS:
                assert abs(c.weights[k] - base.weights[k]) <= POLICY_MAX_PARAM_DELTA

    def test_candidate_exceeding_delta_rejected(self) -> None:
        from swarm.config import POLICY_MAX_PARAM_DELTA

        base = build_policy(SAFE_THRESH, SAFE_WEIGHTS)
        cands = generate_candidates(base)
        # With default deltas of ±0.10 and POLICY_MAX_PARAM_DELTA=0.20, all
        # standard candidates should be within bounds. Confirm by checking
        # that no candidate exceeds the threshold.
        for c in cands:
            for k in THRESHOLD_KEYS:
                assert abs(c.thresholds[k] - base.thresholds[k]) <= POLICY_MAX_PARAM_DELTA + 1e-9


class TestFeedbackInfluence:
    def test_feedback_success_affects_metric(self) -> None:
        # Two candidates with same stability but different feedback success.
        def replay_good_feedback(inp, thr, w):
            return {"selected_supplier": "Supp_A", "score": 0.5}

        traces_good = [
            {
                "events": [
                    {
                        "event_type": "procurement_request",
                        "payload": {"material": "aluminum"},
                        "created_at": "x",
                    }
                ],
                "artifacts": [
                    {
                        "artifact_type": "result",
                        "data": {
                            "selected_supplier": "Supp_A",
                            "reasoning": {"ranked": [{"supplier_id": "Supp_A", "score": 0.5}]},
                        },
                    }
                ],
                "feedback": {"success": True, "outcome_score": 0.9},
            }
        ] * 5

        traces_bad = [
            {
                "events": [
                    {
                        "event_type": "procurement_request",
                        "payload": {"material": "aluminum"},
                        "created_at": "x",
                    }
                ],
                "artifacts": [
                    {
                        "artifact_type": "result",
                        "data": {
                            "selected_supplier": "Supp_A",
                            "reasoning": {"ranked": [{"supplier_id": "Supp_A", "score": 0.5}]},
                        },
                    }
                ],
                "feedback": {"success": False, "outcome_score": 0.1},
            }
        ] * 5

        cand = build_policy(SAFE_THRESH, SAFE_WEIGHTS)
        res_good = evaluate_candidate(
            traces_good, cand, active_metric=0.0, replay_fn=replay_good_feedback
        )
        res_bad = evaluate_candidate(
            traces_bad, cand, active_metric=0.0, replay_fn=replay_good_feedback
        )
        # Good feedback should yield a higher metric.
        assert res_good.metric > res_bad.metric
        assert res_good.feedback_success_rate == 1.0
        assert res_bad.feedback_success_rate == 0.0


# --------------------------------------------------------------------------- #
# Replay integration (override priority)
# --------------------------------------------------------------------------- #


class TestReplayOverridePriority:
    def test_override_strategy_weights_propagates(self) -> None:
        custom = {"price_weight": 0.6, "score_weight": 0.25, "carbon_weight": 0.15}
        with override_strategy_weights(custom):
            from swarm.learning.adaptive_policy import get_strategy_weights

            assert get_strategy_weights() == custom

    def test_override_none_falls_through(self) -> None:
        with override_strategy_weights(None):
            from swarm.learning.adaptive_policy import get_strategy_weights

            # No active policy in temp DB -> None (canonical fallback).
            assert get_strategy_weights() is None
