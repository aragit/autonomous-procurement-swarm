"""Tests for v1.0 Step 18: Persistent Event Store.

Tests ``swarm.storage.event_store`` — the SQLite-backed append-only store
for events, artifacts, and LLM consensus history.
"""

import sqlite3

import pytest

from swarm.storage.event_store import (
    init_db,
    load_state,
    store_artifact,
    store_event,
    store_llm_record,
)


@pytest.fixture
def db_path(tmp_path) -> str:
    """Provide a fresh temp database path for each test."""
    return str(tmp_path / "test_event_store.db")


@pytest.fixture
def initialized_db(db_path: str) -> str:
    init_db(db_path)
    return db_path


def _set_db_path(db_path: str) -> None:
    import swarm.storage.event_store as es
    es._DB_PATH = db_path


class TestInitDb:
    def test_creates_all_tables(self, db_path: str) -> None:
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        tables = {
            row[0]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        assert "events" in tables
        assert "artifacts" in tables
        assert "llm_history" in tables
        assert "feedback" in tables
        conn.close()

    def test_init_db_idempotent(self, db_path: str) -> None:
        init_db(db_path)
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        tables = {
            row[0]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        assert tables == {"events", "artifacts", "llm_history", "feedback", "policies"}
        conn.close()


class TestStoreEvent:
    def test_store_event_persists_correctly(self, initialized_db: str) -> None:
        _set_db_path(initialized_db)
        store_event("TRACE-1", "requirement_created", {"material": "steel"})

        conn = sqlite3.connect(initialized_db)
        cursor = conn.cursor()
        row = cursor.execute(
            "SELECT trace_id, event_type, payload FROM events"
        ).fetchone()
        assert row is not None
        assert row[0] == "TRACE-1"
        assert row[1] == "requirement_created"
        assert "steel" in row[2]
        conn.close()

    def test_multiple_events_preserve_order(self, initialized_db: str) -> None:
        _set_db_path(initialized_db)
        store_event("TRACE-2", "event_a", {"seq": 1})
        store_event("TRACE-2", "event_b", {"seq": 2})
        store_event("TRACE-2", "event_c", {"seq": 3})

        conn = sqlite3.connect(initialized_db)
        cursor = conn.cursor()
        rows = cursor.execute(
            "SELECT event_type FROM events WHERE trace_id='TRACE-2' ORDER BY id ASC"
        ).fetchall()
        assert [r[0] for r in rows] == ["event_a", "event_b", "event_c"]
        conn.close()

    def test_deterministic_serialization(self, initialized_db: str) -> None:
        """Same payload always serializes to the same JSON string."""
        _set_db_path(initialized_db)
        payload = {"z": 1, "a": 2, "m": 3}
        store_event("TRACE-3", "test", payload)

        conn = sqlite3.connect(initialized_db)
        cursor = conn.cursor()
        stored = cursor.execute(
            "SELECT payload FROM events WHERE trace_id='TRACE-3'"
        ).fetchone()[0]
        assert stored == '{"a": 2, "m": 3, "z": 1}'  # sorted keys
        conn.close()


class TestStoreArtifact:
    def test_store_artifact_persists_correctly(self, initialized_db: str) -> None:
        _set_db_path(initialized_db)
        store_artifact("TRACE-4", "strategy", {"name": "balanced", "weights": {"price": 0.5}})

        conn = sqlite3.connect(initialized_db)
        cursor = conn.cursor()
        row = cursor.execute(
            "SELECT trace_id, artifact_type, data FROM artifacts"
        ).fetchone()
        assert row is not None
        assert row[0] == "TRACE-4"
        assert row[1] == "strategy"
        assert "balanced" in row[2]
        conn.close()


class TestStoreLlmRecord:
    def test_store_llm_record_persists_correctly(self, initialized_db: str) -> None:
        _set_db_path(initialized_db)
        record = {
            "round": 1,
            "confidence": 0.85,
            "stability": 0.9,
            "trust": 0.765,
            "decision_reason": "accepted",
            "payload": {"confidence": 0.85, "agreement_score": 0.8},
        }
        store_llm_record("TRACE-5", record)

        conn = sqlite3.connect(initialized_db)
        cursor = conn.cursor()
        row = cursor.execute(
            "SELECT trace_id, round, confidence, stability, trust, decision_reason FROM llm_history"
        ).fetchone()
        assert row is not None
        assert row[0] == "TRACE-5"
        assert row[1] == 1
        assert row[2] == 0.85
        assert row[3] == 0.9
        assert row[4] == 0.765
        assert row[5] == "accepted"
        conn.close()

    def test_missing_fields_use_defaults(self, initialized_db: str) -> None:
        _set_db_path(initialized_db)
        record = {"decision_reason": "low_confidence"}
        store_llm_record("TRACE-6", record)

        state = load_state("TRACE-6")
        assert len(state["llm_history"]) == 1
        assert state["llm_history"][0]["round"] == 0
        assert state["llm_history"][0]["confidence"] is None
        assert state["llm_history"][0]["decision_reason"] == "low_confidence"


class TestLoadState:
    def test_load_state_reconstructs_full_state(self, initialized_db: str) -> None:
        _set_db_path(initialized_db)
        store_event("TRACE-7", "started", {"goal": "source steel"})
        store_artifact("TRACE-7", "strategy", {"name": "balanced"})
        store_llm_record(
            "TRACE-7",
            {"round": 1, "confidence": 0.8, "stability": 0.9, "trust": 0.72,
             "decision_reason": "accepted", "payload": {}},
        )
        store_event("TRACE-8", "started", {"goal": "source aluminum"})
        store_artifact("TRACE-8", "strategy", {"name": "cost_optimized"})

        state = load_state("TRACE-7")
        assert len(state["events"]) == 1
        assert len(state["artifacts"]) == 1
        assert len(state["llm_history"]) == 1
        assert state["events"][0]["event_type"] == "started"
        assert state["artifacts"][0]["artifact_type"] == "strategy"
        assert state["llm_history"][0]["round"] == 1

    def test_load_state_empty_trace_returns_empty(self, initialized_db: str) -> None:
        _set_db_path(initialized_db)
        state = load_state("NONEXISTENT-TRACE")
        assert state["events"] == []
        assert state["artifacts"] == []
        assert state["llm_history"] == []

    def test_multiple_writes_preserve_order(self, initialized_db: str) -> None:
        _set_db_path(initialized_db)
        for i in range(1, 4):
            store_llm_record(
                "TRACE-ORDER",
                {"round": i, "confidence": 0.8, "stability": 0.9, "trust": 0.72,
                 "decision_reason": "accepted", "payload": {}},
            )

        state = load_state("TRACE-ORDER")
        rounds = [r["round"] for r in state["llm_history"]]
        assert rounds == [1, 2, 3]


class TestDeterminism:
    def test_same_state_produces_same_output(self, initialized_db: str) -> None:
        _set_db_path(initialized_db)
        store_event("TRACE-DET", "started", {"data": "test"})
        store_artifact("TRACE-DET", "strategy", {"name": "balanced"})
        store_llm_record(
            "TRACE-DET",
            {"round": 1, "confidence": 0.8, "stability": 0.9, "trust": 0.72,
             "decision_reason": "accepted", "payload": {"k": "v"}},
        )

        s1 = load_state("TRACE-DET")
        s2 = load_state("TRACE-DET")
        assert s1 == s2

    def test_serialization_uses_sorted_keys(self, initialized_db: str) -> None:
        """Payloads are serialized with sorted keys for determinism."""
        _set_db_path(initialized_db)
        store_event("TRACE-SORTED", "test", {"c": 3, "a": 1, "b": 2})

        conn = sqlite3.connect(initialized_db)
        cursor = conn.cursor()
        stored = cursor.execute(
            "SELECT payload FROM events WHERE trace_id='TRACE-SORTED'"
        ).fetchone()[0]
        conn.close()

        assert stored == '{"a": 1, "b": 2, "c": 3}'
