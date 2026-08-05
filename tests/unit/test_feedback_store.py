"""Tests for v1.0 Step 19: Feedback Store & Full Trace Loading.

Tests ``store_feedback``, ``load_feedback``, and ``load_full_trace`` in
``swarm.storage.event_store``.
"""

import sqlite3

import pytest

from swarm.storage.event_store import (
    init_db,
    load_all_feedback,
    load_feedback,
    load_full_trace,
    store_event,
    store_feedback,
    store_llm_record,
)


@pytest.fixture
def db_path(tmp_path) -> str:
    return str(tmp_path / "test_feedback_store.db")


@pytest.fixture
def initialized_db(db_path: str) -> str:
    init_db(db_path)
    return db_path


def _set_db_path(db_path: str) -> None:
    import swarm.storage.event_store as es
    es._DB_PATH = db_path


class TestFeedbackTable:
    def test_feedback_table_created_on_init(self, db_path: str) -> None:
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        tables = {
            row[0]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        assert "feedback" in tables
        conn.close()


class TestStoreFeedback:
    def test_store_feedback_persists_correctly(self, initialized_db: str) -> None:
        _set_db_path(initialized_db)
        store_feedback("TRACE-FB-1", outcome_score=0.85, success=True, latency_ms=1500.5)

        conn = sqlite3.connect(initialized_db)
        cursor = conn.cursor()
        row = cursor.execute(
            "SELECT trace_id, outcome_score, success, latency_ms, user_feedback FROM feedback"
        ).fetchone()
        assert row is not None
        assert row[0] == "TRACE-FB-1"
        assert row[1] == 0.85
        assert row[2] == 1  # stored as 0/1
        assert row[3] == 1500.5
        assert row[4] is None  # no user feedback
        conn.close()

    def test_store_feedback_with_user_feedback(self, initialized_db: str) -> None:
        _set_db_path(initialized_db)
        store_feedback(
            "TRACE-FB-2",
            outcome_score=0.6,
            success=False,
            latency_ms=3000.0,
            user_feedback="Too slow",
        )

        conn = sqlite3.connect(initialized_db)
        cursor = conn.cursor()
        row = cursor.execute(
            "SELECT user_feedback FROM feedback WHERE trace_id='TRACE-FB-2'"
        ).fetchone()
        assert row is not None
        assert row[0] == "Too slow"
        conn.close()

    def test_store_feedback_appends_multiple_rows(self, initialized_db: str) -> None:
        _set_db_path(initialized_db)
        store_feedback("TRACE-FB-3", 0.7, True, 1000.0)
        store_feedback("TRACE-FB-3", 0.9, True, 900.0, "great")

        conn = sqlite3.connect(initialized_db)
        cursor = conn.cursor()
        count = cursor.execute(
            "SELECT COUNT(*) FROM feedback WHERE trace_id='TRACE-FB-3'"
        ).fetchone()[0]
        assert count == 2
        conn.close()


class TestLoadFeedback:
    def test_load_feedback_returns_none_no_data(self, initialized_db: str) -> None:
        _set_db_path(initialized_db)
        result = load_feedback("NONEXISTENT-FEEDBACK")
        assert result is None

    def test_load_feedback_returns_latest(self, initialized_db: str) -> None:
        _set_db_path(initialized_db)
        store_feedback("TRACE-LOAD-1", 0.5, True, 1000.0, "first")
        store_feedback("TRACE-LOAD-1", 0.9, True, 900.0, "second")

        fb = load_feedback("TRACE-LOAD-1")
        assert fb is not None
        assert fb["outcome_score"] == 0.9
        assert fb["success"] is True
        assert fb["latency_ms"] == 900.0
        assert fb["user_feedback"] == "second"

    def test_load_feedback_success_false(self, initialized_db: str) -> None:
        _set_db_path(initialized_db)
        store_feedback("TRACE-LOAD-2", 0.1, False, 5000.0, "failed")

        fb = load_feedback("TRACE-LOAD-2")
        assert fb is not None
        assert fb["success"] is False


class TestLoadFullTrace:
    def test_load_full_trace_includes_feedback_none(self, initialized_db: str) -> None:
        _set_db_path(initialized_db)
        store_event("TRACE-FT-1", "started", {"goal": "test"})
        store_llm_record(
            "TRACE-FT-1",
            {
                "round": 1,
                "confidence": 0.8,
                "stability": 0.9,
                "trust": 0.72,
                "decision_reason": "accepted",
                "payload": {},
            },
        )

        trace = load_full_trace("TRACE-FT-1")
        assert len(trace["events"]) == 1
        assert len(trace["artifacts"]) == 0
        assert len(trace["llm_history"]) == 1
        assert trace["feedback"] is None

    def test_load_full_trace_with_feedback(self, initialized_db: str) -> None:
        _set_db_path(initialized_db)
        store_event("TRACE-FT-2", "started", {"goal": "test"})
        store_llm_record(
            "TRACE-FT-2",
            {
                "round": 1,
                "confidence": 0.8,
                "stability": 0.9,
                "trust": 0.72,
                "decision_reason": "accepted",
                "payload": {},
            },
        )
        store_feedback("TRACE-FT-2", 0.85, True, 1200.0, "good")

        trace = load_full_trace("TRACE-FT-2")
        assert len(trace["events"]) == 1
        assert len(trace["llm_history"]) == 1
        assert trace["feedback"] is not None
        assert trace["feedback"]["outcome_score"] == 0.85
        assert trace["feedback"]["success"] is True

    def test_load_full_trace_empty_trace(self, initialized_db: str) -> None:
        _set_db_path(initialized_db)
        trace = load_full_trace("EMPTY-TRACE-FT")
        assert trace["events"] == []
        assert trace["artifacts"] == []
        assert trace["llm_history"] == []
        assert trace["feedback"] is None


class TestLoadAllFeedback:
    def test_returns_empty_list_no_feedback(self, initialized_db: str) -> None:
        _set_db_path(initialized_db)
        result = load_all_feedback()
        assert result == []

    def test_returns_all_feedback_sorted_by_created_at(self, initialized_db: str) -> None:
        _set_db_path(initialized_db)
        store_feedback("TRACE-ALL-1", 0.8, True, 1000.0, "first")
        store_feedback("TRACE-ALL-2", 0.6, False, 2000.0, "second")
        store_feedback("TRACE-ALL-3", 0.9, True, 1500.0, "third")

        result = load_all_feedback()
        assert len(result) == 3
        assert result[0]["trace_id"] == "TRACE-ALL-1"
        assert result[1]["trace_id"] == "TRACE-ALL-2"
        assert result[2]["trace_id"] == "TRACE-ALL-3"

    def test_feedback_has_all_fields(self, initialized_db: str) -> None:
        _set_db_path(initialized_db)
        store_feedback("TRACE-ALL-4", 0.85, True, 1500.5, "good")

        result = load_all_feedback()
        assert len(result) == 1
        fb = result[0]
        assert fb["trace_id"] == "TRACE-ALL-4"
        assert fb["outcome_score"] == 0.85
        assert fb["success"] is True
        assert fb["latency_ms"] == 1500.5
        assert fb["user_feedback"] == "good"
        assert "created_at" in fb
