"""Tests for v0.9 Step 15: single-entry procurement execution endpoint."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from swarm.api.procurement import (
    RequirementPayload,
    generate_trace_id,
    router,
)


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


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
