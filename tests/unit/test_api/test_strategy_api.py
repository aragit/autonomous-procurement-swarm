"""Tests for v0.9 Step 13: FastAPI observability endpoints."""

import pytest
from fastapi.testclient import TestClient

from swarm.api.strategy import app, set_state
from swarm.core.state import SwarmState
from swarm.utils.llm_memory import record_llm_consensus


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    set_state(None)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def state() -> SwarmState:
    return SwarmState(request_id="REQ-API-TEST", goal="strategy")


@pytest.fixture
def populated_state(state: SwarmState) -> SwarmState:
    record_llm_consensus(
        state,
        correlation_id="REQ-API-TEST",
        consensus={"confidence": 0.8, "agreement_score": 0.75, "completeness": 1.0},
        round_number=1,
        stability=0.9,
        trust=0.72,
        decision_reason="accepted",
    )
    return state


class TestHealthEndpoint:
    def test_health(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestMetricsEndpoint:
    def test_metrics_found(self, client: TestClient) -> None:
        s = SwarmState(request_id="REQ-API-TEST", goal="strategy")
        record_llm_consensus(
            s,
            correlation_id="REQ-API-METRICS",
            consensus={"confidence": 0.8, "agreement_score": 0.75, "completeness": 1.0},
            round_number=1,
            stability=0.9,
            trust=0.72,
            decision_reason="accepted",
        )
        set_state(s)
        response = client.get("/llm/metrics/REQ-API-METRICS")
        assert response.status_code == 200
        data = response.json()
        assert data["history_depth"] == 1
        assert data["acceptance_rate"] == 1.0

    def test_metrics_empty_history(self, client: TestClient) -> None:
        s = SwarmState(request_id="REQ-API-EMPTY", goal="strategy")
        set_state(s)
        response = client.get("/llm/metrics/REQ-API-EMPTY")
        assert response.status_code == 200
        data = response.json()
        assert data["history_depth"] == 0
        assert data["acceptance_rate"] == 0.0


class TestDriftEndpoint:
    def test_drift_detected(self, client: TestClient) -> None:
        s = SwarmState(request_id="REQ-API-DRIFT", goal="strategy")
        record_llm_consensus(
            s,
            correlation_id="REQ-API-DRIFT",
            consensus={"confidence": 0.95},
            round_number=1,
            stability=0.9,
            trust=0.855,
            decision_reason="accepted",
        )
        record_llm_consensus(
            s,
            correlation_id="REQ-API-DRIFT",
            consensus={"confidence": 0.70},
            round_number=2,
            stability=0.9,
            trust=0.63,
            decision_reason="accepted",
        )
        set_state(s)
        response = client.get("/llm/drift/REQ-API-DRIFT")
        assert response.status_code == 200
        data = response.json()
        assert data["drift_detected"] is True


class TestExplanationEndpoint:
    def test_explanation(self, client: TestClient) -> None:
        s = SwarmState(request_id="REQ-API-EXP", goal="strategy")
        record_llm_consensus(
            s,
            correlation_id="REQ-API-EXP",
            consensus={"confidence": 0.8, "agreement_score": 0.75, "completeness": 1.0},
            round_number=1,
            stability=0.9,
            trust=0.72,
            decision_reason="accepted",
        )
        set_state(s)
        response = client.get("/llm/explanation/REQ-API-EXP")
        assert response.status_code == 200
        data = response.json()
        assert "aggregate" in data
        assert data["aggregate"]["total_rounds"] == 1


class TestHistoryEndpoint:
    def test_history_returns_records(self, client: TestClient) -> None:
        s = SwarmState(request_id="REQ-API-HIST", goal="strategy")
        record_llm_consensus(
            s,
            correlation_id="REQ-API-HIST",
            consensus={"confidence": 0.8},
            round_number=1,
            stability=0.9,
            trust=0.72,
        )
        set_state(s)
        response = client.get("/llm/history/REQ-API-HIST")
        assert response.status_code == 200
        data = response.json()
        assert len(data["records"]) == 1
