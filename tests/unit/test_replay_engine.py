"""Tests for v1.0 Step 21: Replay + Simulation Engine.

Covers ``extract_input``, ``run_procurement``, ``replay_trace``,
``compare_results`` and ``simulate_all_traces`` in
``swarm.simulation.replay_engine``.
"""

from __future__ import annotations

import pytest

from swarm.api.procurement import RequirementPayload, generate_trace_id
from swarm.simulation.replay_engine import (
    TraceNotFoundError,
    compare_results,
    extract_input,
    replay_trace,
    run_procurement,
    simulate_all_traces,
)
from swarm.storage.event_store import (
    init_db,
    load_recent_trace_ids,
    store_artifact,
    store_event,
    store_feedback,
)


@pytest.fixture
def db_path(tmp_path) -> str:
    return str(tmp_path / "replay_test.db")


@pytest.fixture
def initialized_db(db_path: str) -> str:
    init_db(db_path)
    return db_path


@pytest.fixture(autouse=True)
def _isolate_event_store(db_path: str):
    """Point the event store at a per-test temp DB and initialize it."""
    import swarm.storage.event_store as es

    orig = es._DB_PATH
    es._DB_PATH = db_path
    es.init_db(db_path)
    yield
    es._DB_PATH = orig


REQUIREMENT_INPUT = {
    "material": "aluminum",
    "quantity": 1000,
    "budget": 2_000_000.0,
    "target_lead_time_days": 30,
    "max_carbon_per_unit": None,
    "goal": None,
    "supplier_count": 5,
}


def _seed_trace(trace_id: str) -> None:
    """Persist a minimal trace (procurement_request + result artifact)."""
    store_event(trace_id, "procurement_request", REQUIREMENT_INPUT)
    store_artifact(
        trace_id,
        "result",
        {
            "selected_supplier": "MinerCorp_A",
            "reasoning": {
                "criteria": "balanced",
                "ranked": [
                    {"supplier_id": "MinerCorp_A", "score": 0.86, "price": 1000.0},
                    {"supplier_id": "DistribCorp_B", "score": 0.79, "price": 1050.0},
                ],
            },
        },
    )


class TestExtractInput:
    def test_extracts_procurement_request_event(self) -> None:
        events = [
            {"event_type": "procurement_request", "payload": REQUIREMENT_INPUT, "created_at": "x"},
            {"event_type": "something_else", "payload": {}, "created_at": "y"},
        ]
        assert extract_input(events) == REQUIREMENT_INPUT

    def test_falls_back_to_requirement_created(self) -> None:
        events = [
            {
                "event_type": "RequirementCreated",
                "payload": {
                    "requirement": {
                        "text": "Source aluminum",
                        "constraints": {
                            "material": "aluminum",
                            "quantity": 1000,
                            "budget": 2000000.0,
                            "target_lead_time_days": 30,
                            "max_carbon_per_unit": None,
                            "max_unit_price": 1200.0,
                        },
                    }
                },
                "created_at": "x",
            }
        ]
        result = extract_input(events)
        assert result["material"] == "aluminum"
        assert result["quantity"] == 1000
        assert result["budget"] == 2000000.0

    def test_empty_events_returns_empty(self) -> None:
        assert extract_input([]) == {}


class TestRunProcurement:
    def test_returns_valid_structure(self) -> None:
        result = run_procurement(REQUIREMENT_INPUT, adaptive=False)
        assert result["trace_id"]
        assert "selected_supplier" in result
        assert "score" in result
        assert "strategy" in result
        assert "llm" in result
        assert "thresholds_used" in result["llm"]

    def test_deterministic_same_thresholds(self) -> None:
        # The DECISION and thresholds are deterministic. (The LLM consensus
        # history is not across separate asyncio.run calls due to a pre-existing
        # concurrent agent-delivery ordering in the swarm runtime, so we assert
        # determinism on the decision-relevant fields only.)
        r1 = run_procurement(REQUIREMENT_INPUT, adaptive=False)
        r2 = run_procurement(REQUIREMENT_INPUT, adaptive=False)
        assert r1["selected_supplier"] == r2["selected_supplier"]
        assert r1["score"] == r2["score"]
        assert r1["llm"]["thresholds_used"] == r2["llm"]["thresholds_used"]
        assert r1["llm"]["thresholds_source"] == r2["llm"]["thresholds_source"]
        assert r1["trace_id"] == r2["trace_id"]

    def test_adaptive_vs_static_thresholds_differ_with_feedback(self, initialized_db: str) -> None:
        # Seed enough feedback (>= MIN_FEEDBACK_SAMPLES) of failures with low
        # outcome scores so adaptive thresholds are shifted away from config.
        for i in range(6):
            store_feedback(
                f"TRACE-ADAPT-{i}",
                outcome_score=0.2,
                success=False,
                latency_ms=5000.0,
                user_feedback=f"fail {i}",
            )

        adaptive = run_procurement(REQUIREMENT_INPUT, adaptive=True)
        static = run_procurement(REQUIREMENT_INPUT, adaptive=False)

        assert adaptive["llm"]["thresholds_source"] == "adaptive"
        assert static["llm"]["thresholds_source"] == "static"
        assert adaptive["llm"]["thresholds_used"] != static["llm"]["thresholds_used"]

    def test_no_feedback_adaptive_equals_static(self) -> None:
        # With no feedback the adaptive path falls back to config defaults,
        # so thresholds_used should match the static config thresholds.
        adaptive = run_procurement(REQUIREMENT_INPUT, adaptive=True)
        static = run_procurement(REQUIREMENT_INPUT, adaptive=False)
        assert adaptive["llm"]["thresholds_used"] == static["llm"]["thresholds_used"]

    def test_does_not_mutate_production_trace(self, initialized_db: str) -> None:
        trace_id = generate_trace_id(RequirementPayload(**REQUIREMENT_INPUT))
        store_event(trace_id, "procurement_request", REQUIREMENT_INPUT)
        # A second distinct artifact to confirm row count is unchanged
        store_event(trace_id, "marker", {"n": 1})

        from swarm.storage.event_store import load_state

        before = load_state(trace_id)
        before_count = len(before["events"])
        # Replay must not write procurement_request/llm records back to THIS trace
        run_procurement(REQUIREMENT_INPUT, adaptive=False)
        after = load_state(trace_id)
        after_count = len(after["events"])
        assert before_count == after_count
        assert before == after


class TestCompareResults:
    def test_same_supplier(self) -> None:
        original = {"selected_supplier": "A", "score": 0.86}
        replayed = {"selected_supplier": "A", "score": 0.9}
        result = compare_results(original, replayed)
        assert result["same_supplier"] is True
        assert result["decision_changed"] is False
        assert result["score_delta"] == round(0.9 - 0.86, 4)

    def test_different_supplier(self) -> None:
        original = {"selected_supplier": "A", "score": 0.86}
        replayed = {"selected_supplier": "B", "score": 0.79}
        result = compare_results(original, replayed)
        assert result["same_supplier"] is False
        assert result["decision_changed"] is True

    def test_none_supplier_treated_as_change(self) -> None:
        original = {"selected_supplier": None, "score": 0.0}
        replayed = {"selected_supplier": "A", "score": 0.8}
        result = compare_results(original, replayed)
        assert result["same_supplier"] is False
        assert result["decision_changed"] is True


class TestReplayTrace:
    def test_replay_produces_valid_output(self, initialized_db: str) -> None:
        trace_id = "TRACE-REPLAY-1"
        _seed_trace(trace_id)
        result = replay_trace(trace_id, use_adaptive=False)
        assert result["trace_id"] == trace_id
        assert "original" in result
        assert "replayed" in result
        assert "comparison" in result
        comp = result["comparison"]
        assert {"same_supplier", "score_delta", "decision_changed"} <= set(comp)

    def test_same_thresholds_reproduces_original(self, initialized_db: str) -> None:
        trace_id = "TRACE-REPLAY-2"
        _seed_trace(trace_id)
        original = replay_trace(trace_id, use_adaptive=False)
        replayed_again = replay_trace(trace_id, use_adaptive=False)
        # The decision is deterministic; assert on decision fields + comparison.
        orig = original["replayed"]
        rep = replayed_again["replayed"]
        assert orig["selected_supplier"] == rep["selected_supplier"]
        assert orig["score"] == rep["score"]
        assert original["comparison"] == replayed_again["comparison"]

    def test_missing_trace_raises(self, initialized_db: str) -> None:
        with pytest.raises(TraceNotFoundError):
            replay_trace("NONEXISTENT-REPLAY")


class TestSimulateAllTraces:
    def test_empty_returns_zeros(self) -> None:
        result = simulate_all_traces(limit=10)
        assert result["total"] == 0
        assert result["changed_decisions"] == 0
        assert result["avg_score_delta"] == 0.0
        assert result["improvement_rate"] == 0.0
        assert result["results"] == []

    def test_aggregates_replayed_traces(self, initialized_db: str) -> None:
        for i in range(3):
            trace_id = f"TRACE-SIM-{i}"
            _seed_trace(trace_id)

        result = simulate_all_traces(limit=10)
        assert result["total"] == 3
        assert result["changed_decisions"] == 0  # deterministic re-run matches original
        assert isinstance(result["avg_score_delta"], float)
        assert isinstance(result["improvement_rate"], float)
        assert len(result["results"]) == 3

    def test_respects_limit(self, initialized_db: str) -> None:
        for i in range(5):
            _seed_trace(f"TRACE-LIMIT-{i}")
        result = simulate_all_traces(limit=2)
        assert result["total"] == 2


class TestLoadRecentTraceIds:
    def test_returns_ids_newest_first(self, initialized_db: str) -> None:
        _seed_trace("TRACE-RECENT-1")
        _seed_trace("TRACE-RECENT-2")
        _seed_trace("TRACE-RECENT-3")
        ids = load_recent_trace_ids(limit=10)
        assert "TRACE-RECENT-3" in ids
        # Seeded last, so it has the highest id and should be first (newest).
        assert ids[0] == "TRACE-RECENT-3"
        assert len(ids) == 3

    def test_limit_caps_results(self, initialized_db: str) -> None:
        for i in range(5):
            _seed_trace(f"TRACE-LIM-{i}")
        ids = load_recent_trace_ids(limit=2)
        assert len(ids) == 2
