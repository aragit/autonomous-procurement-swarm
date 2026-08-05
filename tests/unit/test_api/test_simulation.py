"""Tests for v1.0 Step 21: Simulation & Replay API endpoints."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from swarm.api.procurement import router as procurement_router
from swarm.api.simulation import router as simulation_router


@pytest.fixture
def client(tmp_path) -> TestClient:
    app = FastAPI()
    app.include_router(procurement_router)
    app.include_router(simulation_router)
    return TestClient(app)


@pytest.fixture
def temp_db(tmp_path):
    import swarm.storage.event_store as es

    db_path = str(tmp_path / "simulation_api.db")
    es._DB_PATH = db_path
    es.init_db(db_path)
    yield db_path
    import swarm.api.strategy as strat

    strat.set_state(None)
    strat.set_state_provider(None)


REQUIREMENT = {
    "material": "aluminum",
    "quantity": 1000,
    "budget": 2_000_000.0,
    "target_lead_time_days": 30,
}


class TestReplayEndpoint:
    def test_replay_existing_trace(self, client: TestClient, temp_db) -> None:
        # Run a procurement so a trace exists in the DB
        run = client.post("/procurement/run", json={"requirement": REQUIREMENT})
        assert run.status_code == 200
        trace_id = run.json()["trace_id"]

        resp = client.post(f"/simulation/replay/{trace_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["trace_id"] == trace_id
        assert "original" in data
        assert "replayed" in data
        assert "comparison" in data
        assert {"same_supplier", "score_delta", "decision_changed"} <= set(data["comparison"])

    def test_replay_reproduces_supplier(self, client: TestClient, temp_db) -> None:
        run = client.post("/procurement/run", json={"requirement": REQUIREMENT})
        trace_id = run.json()["trace_id"]
        original_supplier = run.json()["result"]["selected_supplier"]

        resp = client.post(f"/simulation/replay/{trace_id}")
        replayed = resp.json()["replayed"]["selected_supplier"]
        assert replayed == original_supplier

    def test_replay_missing_trace(self, client: TestClient, temp_db) -> None:
        resp = client.post("/simulation/replay/TRACE-DOES-NOT-EXIST")
        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] == "trace_not_found"
        assert data["trace_id"] == "TRACE-DOES-NOT-EXIST"

    def test_replay_deterministic(self, client: TestClient, temp_db) -> None:
        run = client.post("/procurement/run", json={"requirement": REQUIREMENT})
        trace_id = run.json()["trace_id"]
        r1 = client.post(f"/simulation/replay/{trace_id}").json()
        r2 = client.post(f"/simulation/replay/{trace_id}").json()
        # The decision + comparison are deterministic (LLM observability history
        # is not across separate process runs — see replay_engine docs).
        assert r1["comparison"] == r2["comparison"]
        assert r1["replayed"]["selected_supplier"] == r2["replayed"]["selected_supplier"]
        assert r1["replayed"]["score"] == r2["replayed"]["score"]
        assert r1["replayed"]["llm"]["thresholds_used"] == r2["replayed"]["llm"]["thresholds_used"]


class TestSimulationRunEndpoint:
    def test_batch_summary_structure(self, client: TestClient, temp_db) -> None:
        # seed two traces
        client.post("/procurement/run", json={"requirement": REQUIREMENT})
        client.post(
            "/procurement/run",
            json={"requirement": {**REQUIREMENT, "material": "steel"}},
        )

        resp = client.get("/simulation/run")
        assert resp.status_code == 200
        data = resp.json()
        assert "summary" in data
        summary = data["summary"]
        assert {"total", "changed_decisions", "avg_score_delta", "improvement_rate"} <= set(summary)
        assert summary["total"] == 2

    def test_empty_summary(self, client: TestClient, temp_db) -> None:
        resp = client.get("/simulation/run")
        assert resp.status_code == 200
        summary = resp.json()["summary"]
        assert summary["total"] == 0
        assert summary["changed_decisions"] == 0

    def test_limit_param(self, client: TestClient, temp_db) -> None:
        # Use distinct requirements so each produces a distinct trace_id.
        materials = ["aluminum", "steel", "copper", "plastic"]
        for material in materials:
            client.post(
                "/procurement/run",
                json={"requirement": {**REQUIREMENT, "material": material}},
            )
        resp = client.get("/simulation/run?limit=2")
        assert resp.status_code == 200
        assert resp.json()["summary"]["total"] == 2


class TestSimulationNoSideEffects:
    def test_replay_does_not_add_events(self, client: TestClient, temp_db) -> None:
        from swarm.storage.event_store import load_state

        run = client.post("/procurement/run", json={"requirement": REQUIREMENT})
        trace_id = run.json()["trace_id"]

        events_before = load_state(trace_id)["events"]
        count_before = len(events_before)
        assert any(e["event_type"] == "procurement_request" for e in events_before)

        # Two replays via the API
        client.post(f"/simulation/replay/{trace_id}")
        client.post(f"/simulation/replay/{trace_id}")

        events_after = load_state(trace_id)["events"]
        assert len(events_after) == count_before
        assert events_before == events_after

    def test_replay_does_not_add_artifacts(self, client: TestClient, temp_db) -> None:
        from swarm.storage.event_store import load_full_trace

        run = client.post("/procurement/run", json={"requirement": REQUIREMENT})
        trace_id = run.json()["trace_id"]

        artifacts_before = load_full_trace(trace_id)["artifacts"]
        n_before = len(artifacts_before)

        client.post(f"/simulation/replay/{trace_id}")

        artifacts_after = load_full_trace(trace_id)["artifacts"]
        assert len(artifacts_after) == n_before
