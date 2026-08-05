"""Tests for v0.9 Step 15: single-entry procurement execution endpoint."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from swarm.api.procurement import (
    RequirementPayload,
    generate_trace_id,
    router,
)
from swarm.api.strategy import app as strategy_app


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def strategy_client() -> TestClient:
    return TestClient(strategy_app)


class TestTraceIdGeneration:
    def test_deterministic_same_input(self) -> None:
        p1 = RequirementPayload(
            material="steel",
            quantity=1000,
            budget=2000000.0,
            target_lead_time_days=30,
        )
        p2 = RequirementPayload(
            material="steel",
            quantity=1000,
            budget=2000000.0,
            target_lead_time_days=30,
        )
        assert generate_trace_id(p1) == generate_trace_id(p2)

    def test_different_input(self) -> None:
        p1 = RequirementPayload(
            material="steel",
            quantity=1000,
            budget=2000000.0,
            target_lead_time_days=30,
        )
        p2 = RequirementPayload(
            material="aluminum",
            quantity=1000,
            budget=2000000.0,
            target_lead_time_days=30,
        )
        assert generate_trace_id(p1) != generate_trace_id(p2)


class TestProcurementRunEndpoint:
    def test_happy_path(self, client: TestClient) -> None:
        response = client.post(
            "/procurement/run",
            json={
                "requirement": {
                    "material": "aluminum",
                    "quantity": 1000,
                    "budget": 2000000.0,
                    "target_lead_time_days": 30,
                }
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert "strategy" in data
        assert "llm" in data
        assert "trace_id" in data

        # Result contains selected supplier
        assert "selected_supplier" in data["result"]
        assert data["result"]["selected_supplier"] is not None

        # LLM observability structure
        assert "used" in data["llm"]
        assert "reason" in data["llm"]
        assert "metrics" in data["llm"]
        assert "drift" in data["llm"]
        assert "explain" in data["llm"]

    def test_deterministic_same_input(self, client: TestClient) -> None:
        payload = {
            "requirement": {
                "material": "aluminum",
                "quantity": 1000,
                "budget": 2000000.0,
                "target_lead_time_days": 30,
            }
        }
        r1 = client.post("/procurement/run", json=payload)
        r2 = client.post("/procurement/run", json=payload)
        assert r1.status_code == 200
        assert r2.status_code == 200
        d1 = r1.json()
        d2 = r2.json()
        assert d1["trace_id"] == d2["trace_id"]
        assert d1["result"] == d2["result"]

    def test_no_llm_usage_fallback_reflected(self, client: TestClient) -> None:
        """LLM context is recorded but first-round stability=0 so fallback fires."""
        response = client.post(
            "/procurement/run",
            json={
                "requirement": {
                    "material": "aluminum",
                    "quantity": 1000,
                    "budget": 2000000.0,
                    "target_lead_time_days": 30,
                }
            },
        )
        data = response.json()
        # The SupplierAnalysisLLMAgent runs on QuotesCompleted and creates
        # completions, but first-round stability is 0 (no prior history),
        # so trust = 0 and the fallback fires with 'low_stability'.
        assert data["llm"]["used"] is False
        assert data["llm"]["reason"] in ("no_llm_data", "low_stability")

    def test_metrics_present(self, client: TestClient) -> None:
        response = client.post(
            "/procurement/run",
            json={
                "requirement": {
                    "material": "aluminum",
                    "quantity": 1000,
                    "budget": 2000000.0,
                    "target_lead_time_days": 30,
                }
            },
        )
        data = response.json()
        metrics = data["llm"]["metrics"]
        assert "acceptance_rate" in metrics
        assert "avg_confidence" in metrics
        assert "avg_stability" in metrics
        assert "avg_trust" in metrics
        assert "history_depth" in metrics

    def test_trace_id_uniqueness(self) -> None:
        p1 = RequirementPayload(
            material="steel",
            quantity=1000,
            budget=1000.0,
            target_lead_time_days=30,
        )
        p2 = RequirementPayload(
            material="steel",
            quantity=2000,
            budget=1000.0,
            target_lead_time_days=30,
        )
        assert generate_trace_id(p1) != generate_trace_id(p2)

    def test_error_missing_requirement(self, client: TestClient) -> None:
        response = client.post("/procurement/run", json={})
        data = response.json()
        assert data["error"] == "invalid_requirement"

    def test_error_invalid_schema(self, client: TestClient) -> None:
        response = client.post(
            "/procurement/run",
            json={
                "requirement": {
                    "material": "",
                    "quantity": -1,
                }
            },
        )
        data = response.json()
        assert data["error"] == "invalid_requirement"


class TestDashboardDbBacked:
    """Integration: verify dashboard reads from DB after procurement run."""

    @pytest.fixture(autouse=True)
    def _clean_strategy_state(self) -> None:
        """Clear in-memory state so dashboard must read from DB."""
        from swarm.api.strategy import set_state, set_state_provider
        set_state(None)
        set_state_provider(None)

    @pytest.fixture(autouse=True)
    def _temp_db(self, tmp_path) -> None:
        """Use a temp DB for each test and restore original after."""
        import swarm.storage.event_store as es
        self._orig_db_path = es._DB_PATH
        db_path = str(tmp_path / "integration_test.db")
        es._DB_PATH = db_path
        es.init_db(db_path)
        yield
        es._DB_PATH = self._orig_db_path

    def test_dashboard_reads_data_from_db(
        self,
        client: TestClient,
        strategy_client: TestClient,
    ) -> None:
        """Run procurement, then fetch dashboard — data must come from DB."""
        # Run procurement which stores to DB
        run_response = client.post(
            "/procurement/run",
            json={
                "requirement": {
                    "material": "aluminum",
                    "quantity": 1000,
                    "budget": 2000000.0,
                    "target_lead_time_days": 30,
                }
            },
        )
        assert run_response.status_code == 200
        run_data = run_response.json()
        trace_id = run_data["trace_id"]

        # Fetch dashboard from strategy API (no in-memory state set)
        dash_response = strategy_client.get(f"/llm/dashboard/{trace_id}")
        assert dash_response.status_code == 200
        dash_data = dash_response.json()

        # Dashboard should return the same trace_id and full structure
        assert dash_data["trace_id"] == trace_id
        assert "summary" in dash_data
        assert "decision" in dash_data
        assert "metrics" in dash_data
        assert "drift" in dash_data
        assert "history" in dash_data
        assert "feedback" in dash_data
        assert "learning_signals" in dash_data

        # Metrics should match what was returned by the procurement endpoint
        assert dash_data["metrics"] == run_data["llm"]["metrics"]

    def test_dashboard_missing_trace_after_run(self, strategy_client: TestClient) -> None:
        """Dashboard for unknown trace_id returns trace_not_found."""
        response = strategy_client.get("/llm/dashboard/TRACE-UNKNOWN")
        assert response.status_code == 200
        assert response.json() == {"error": "trace_not_found"}

    def test_dashboard_empty_history_safe_defaults(
        self,
        strategy_client: TestClient,
    ) -> None:
        """Dashboard for trace with no LLM history returns safe defaults."""
        from swarm.api.strategy import set_state
        from swarm.core.state import SwarmState
        # Set in-memory state with no LLM history
        set_state(SwarmState(request_id="TRACE-EMPTY-DASH", goal="test"))

        response = strategy_client.get("/llm/dashboard/TRACE-EMPTY-DASH")
        assert response.status_code == 200
        data = response.json()
        assert data["trace_id"] == "TRACE-EMPTY-DASH"
        assert data["decision"]["used_llm"] is False
        assert data["decision"]["reason"] == "no_data"
        assert data["history"] == []

    def test_dashboard_deterministic(
        self,
        client: TestClient,
        strategy_client: TestClient,
    ) -> None:
        """Dashboard returns the same data for repeated calls."""
        run_response = client.post(
            "/procurement/run",
            json={
                "requirement": {
                    "material": "aluminum",
                    "quantity": 1000,
                    "budget": 2000000.0,
                    "target_lead_time_days": 30,
                }
            },
        )
        trace_id = run_response.json()["trace_id"]

        r1 = strategy_client.get(f"/llm/dashboard/{trace_id}").json()
        r2 = strategy_client.get(f"/llm/dashboard/{trace_id}").json()
        assert r1 == r2
