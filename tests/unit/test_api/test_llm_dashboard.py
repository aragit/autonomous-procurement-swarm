"""Tests for v1.0 Step 17: Unified LLM Dashboard Endpoint.

Tests ``GET /llm/dashboard/{trace_id}`` — a single endpoint that
consolidates metrics, drift, explanation, and history into one
deterministic response.
"""

import pytest
from fastapi.testclient import TestClient

from swarm.api.strategy import app, build_dashboard, set_state, set_state_provider
from swarm.config import CONFIDENCE_THRESHOLD
from swarm.core.state import SwarmState
from swarm.utils.llm_memory import record_llm_consensus


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    set_state(None)
    set_state_provider(None)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _seed_history(
    state: SwarmState,
    correlation_id: str,
    rounds: int = 2,
    confidence: float = 0.8,
    stability: float = 0.9,
    trust: float = 0.72,
    decision_reason: str = "accepted",
) -> None:
    for i in range(1, rounds + 1):
        record_llm_consensus(
            state,
            correlation_id=correlation_id,
            consensus={"confidence": confidence, "agreement_score": 0.75, "completeness": 1.0},
            round_number=i,
            stability=stability,
            trust=trust,
            decision_reason=decision_reason,
        )


class TestDashboardFullResponse:
    def test_valid_trace_returns_full_structure(self, client: TestClient) -> None:
        s = SwarmState(request_id="DASH-VALID", goal="strategy")
        _seed_history(s, "DASH-VALID", rounds=2)
        set_state(s)

        response = client.get("/llm/dashboard/DASH-VALID")
        assert response.status_code == 200
        data = response.json()

        assert data["trace_id"] == "DASH-VALID"
        assert "summary" in data
        assert "decision" in data
        assert "metrics" in data
        assert "drift" in data
        assert "history" in data
        assert "adaptive" in data
        assert "thresholds" in data["adaptive"]
        assert data["adaptive"]["thresholds"]["confidence"] == CONFIDENCE_THRESHOLD

    def test_decision_structure(self, client: TestClient) -> None:
        s = SwarmState(request_id="DASH-DECISION", goal="strategy")
        _seed_history(s, "DASH-DECISION", rounds=1, decision_reason="accepted")
        set_state(s)

        response = client.get("/llm/dashboard/DASH-DECISION")
        data = response.json()

        assert data["decision"]["used_llm"] is True
        assert data["decision"]["reason"] == "accepted"

    def test_metrics_structure(self, client: TestClient) -> None:
        s = SwarmState(request_id="DASH-METRICS", goal="strategy")
        _seed_history(s, "DASH-METRICS", rounds=2)
        set_state(s)

        response = client.get("/llm/dashboard/DASH-METRICS")
        data = response.json()

        metrics = data["metrics"]
        assert "acceptance_rate" in metrics
        assert "avg_confidence" in metrics
        assert "avg_stability" in metrics
        assert "avg_trust" in metrics
        assert "history_depth" in metrics
        assert metrics["history_depth"] == 2

    def test_drift_structure(self, client: TestClient) -> None:
        s = SwarmState(request_id="DASH-DRIFT", goal="strategy")
        _seed_history(s, "DASH-DRIFT", rounds=2)
        set_state(s)

        response = client.get("/llm/dashboard/DASH-DRIFT")
        data = response.json()

        drift = data["drift"]
        assert "drifting" in drift
        assert isinstance(drift["drifting"], bool)
        assert "reasons" in drift
        assert isinstance(drift["reasons"], list)

    def test_history_formatting(self, client: TestClient) -> None:
        s = SwarmState(request_id="DASH-HIST", goal="strategy")
        _seed_history(s, "DASH-HIST", rounds=3)
        set_state(s)

        response = client.get("/llm/dashboard/DASH-HIST")
        data = response.json()

        history = data["history"]
        assert len(history) == 3
        for i, entry in enumerate(history, start=1):
            assert entry["round"] == i
            assert "confidence" in entry
            assert "stability" in entry
            assert "trust" in entry
            assert "decision_reason" in entry


class TestDashboardMissingTrace:
    def test_missing_trace_returns_error(self, client: TestClient) -> None:
        s = SwarmState(request_id="DASH-EXISTING", goal="strategy")

        def provider(trace_id: str) -> SwarmState | None:
            if trace_id == "DASH-EXISTING":
                return s
            return None

        set_state(None)
        set_state_provider(provider)
        response = client.get("/llm/dashboard/DASH-NONEXISTENT")
        assert response.status_code == 200
        assert response.json() == {"error": "trace_not_found"}

    def test_no_state_set_returns_error(self, client: TestClient) -> None:
        response = client.get("/llm/dashboard/ANY-TRACE")
        assert response.status_code == 200
        assert response.json() == {"error": "trace_not_found"}


class TestDashboardEmptyHistory:
    def test_empty_history_safe_defaults(self, client: TestClient) -> None:
        s = SwarmState(request_id="DASH-EMPTY", goal="strategy")
        set_state(s)

        response = client.get("/llm/dashboard/DASH-EMPTY")
        data = response.json()

        assert data["trace_id"] == "DASH-EMPTY"
        assert data["decision"]["used_llm"] is False
        assert data["decision"]["reason"] == "no_data"
        assert data["history"] == []
        assert data["drift"]["drifting"] is False
        assert data["drift"]["reasons"] == []
        assert "adaptive" in data
        assert data["adaptive"]["enabled"] is True
        assert "thresholds" in data["adaptive"]

    def test_build_dashboard_empty_history(self) -> None:
        s = SwarmState(request_id="DASH-FUNC-EMPTY", goal="strategy")
        set_state(s)

        result = build_dashboard("DASH-FUNC-EMPTY")
        assert result["decision"]["used_llm"] is False
        assert result["history"] == []
        assert result["metrics"]["history_depth"] == 0


class TestDashboardConsistency:
    def test_consistent_with_metrics_endpoint(self, client: TestClient) -> None:
        s = SwarmState(request_id="DASH-CONSISTENT", goal="strategy")
        _seed_history(s, "DASH-CONSISTENT", rounds=2)
        set_state(s)

        dashboard_resp = client.get("/llm/dashboard/DASH-CONSISTENT")
        metrics_resp = client.get("/llm/metrics/DASH-CONSISTENT")

        assert dashboard_resp.status_code == 200
        assert metrics_resp.status_code == 200

        dashboard_metrics = dashboard_resp.json()["metrics"]
        endpoint_metrics = metrics_resp.json()

        assert dashboard_metrics == endpoint_metrics

    def test_consistent_with_drift_endpoint(self, client: TestClient) -> None:
        s = SwarmState(request_id="DASH-CONSISTENT-DRIFT", goal="strategy")
        _seed_history(s, "DASH-CONSISTENT-DRIFT", rounds=2)
        set_state(s)

        dashboard_resp = client.get("/llm/dashboard/DASH-CONSISTENT-DRIFT")
        drift_resp = client.get("/llm/drift/DASH-CONSISTENT-DRIFT")

        assert dashboard_resp.status_code == 200
        assert drift_resp.status_code == 200

        dashboard_drift = dashboard_resp.json()["drift"]
        endpoint_drift = drift_resp.json()

        assert dashboard_drift["drifting"] == endpoint_drift["drift_detected"]
        assert dashboard_drift["reasons"] == endpoint_drift["reasons"]

    def test_consistent_with_explanation_endpoint(self, client: TestClient) -> None:
        s = SwarmState(request_id="DASH-CONSISTENT-EXP", goal="strategy")
        _seed_history(s, "DASH-CONSISTENT-EXP", rounds=2)
        set_state(s)

        dashboard_resp = client.get("/llm/dashboard/DASH-CONSISTENT-EXP")
        explain_resp = client.get("/llm/explanation/DASH-CONSISTENT-EXP")

        assert dashboard_resp.status_code == 200
        assert explain_resp.status_code == 200

        dashboard_summary = dashboard_resp.json()["summary"]
        endpoint_summary = explain_resp.json()["summary"]

        assert dashboard_summary == endpoint_summary


class TestDashboardDeterminism:
    def test_same_state_produces_same_results(self, client: TestClient) -> None:
        s = SwarmState(request_id="DASH-DETERMINISTIC", goal="strategy")
        _seed_history(s, "DASH-DETERMINISTIC", rounds=2)
        set_state(s)

        r1 = client.get("/llm/dashboard/DASH-DETERMINISTIC").json()
        r2 = client.get("/llm/dashboard/DASH-DETERMINISTIC").json()
        assert r1 == r2

    def test_no_mutation_of_state(self, client: TestClient) -> None:
        s = SwarmState(request_id="DASH-NO-MUTATE", goal="strategy")
        _seed_history(s, "DASH-NO-MUTATE", rounds=2)
        set_state(s)

        response = client.get("/llm/dashboard/DASH-NO-MUTATE")
        assert response.status_code == 200

        history_before = len(s.artifacts)
        _ = client.get("/llm/dashboard/DASH-NO-MUTATE").json()
        history_after = len(s.artifacts)
        assert history_before == history_after


class TestDashboardDecisionReasons:
    def test_rejected_decision(self, client: TestClient) -> None:
        s = SwarmState(request_id="DASH-REJECTED", goal="strategy")
        _seed_history(s, "DASH-REJECTED", rounds=1, decision_reason="low_trust")
        set_state(s)

        response = client.get("/llm/dashboard/DASH-REJECTED")
        data = response.json()

        assert data["decision"]["used_llm"] is False
        assert data["decision"]["reason"] == "low_trust"
