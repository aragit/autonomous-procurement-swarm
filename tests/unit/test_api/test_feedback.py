"""Tests for v1.0 Step 19: Feedback submission endpoint."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from swarm.api.procurement import FeedbackPayload, router


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def temp_db(monkeypatch, tmp_path):
    """Use a temp DB for each test."""
    import swarm.storage.event_store as es
    db_path = str(tmp_path / "feedback_endpoint.db")
    es._DB_PATH = db_path
    es.init_db(db_path)
    yield db_path
    # cleanup not strictly needed for tmp_path


class TestFeedbackPayload:
    def test_valid_payload(self) -> None:
        payload = FeedbackPayload(
            trace_id="TRACE-1",
            outcome_score=0.85,
            success=True,
            latency_ms=1500.5,
            user_feedback="Good",
        )
        assert payload.trace_id == "TRACE-1"
        assert payload.outcome_score == 0.85
        assert payload.success is True

    def test_score_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError):
            FeedbackPayload(
                trace_id="TRACE-1",
                outcome_score=1.5,
                success=True,
                latency_ms=100.0,
            )

    def test_negative_latency_rejected(self) -> None:
        with pytest.raises(ValueError):
            FeedbackPayload(
                trace_id="TRACE-1",
                outcome_score=0.85,
                success=True,
                latency_ms=-1.0,
            )

    def test_empty_trace_id_rejected(self) -> None:
        with pytest.raises(ValueError):
            FeedbackPayload(
                trace_id="",
                outcome_score=0.85,
                success=True,
                latency_ms=100.0,
            )


class TestFeedbackEndpoint:
    def test_submit_feedback_success(self, client: TestClient, temp_db: str) -> None:
        response = client.post(
            "/procurement/feedback",
            json={
                "trace_id": "TRACE-FB-1",
                "outcome_score": 0.85,
                "success": True,
                "latency_ms": 1500.5,
                "user_feedback": "Good result",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "stored"
        assert data["trace_id"] == "TRACE-FB-1"

    def test_submit_feedback_minimal(self, client: TestClient, temp_db: str) -> None:
        response = client.post(
            "/procurement/feedback",
            json={
                "trace_id": "TRACE-FB-2",
                "outcome_score": 0.5,
                "success": False,
                "latency_ms": 2000.0,
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "stored"

    def test_submit_feedback_invalid_score(self, client: TestClient, temp_db: str) -> None:
        response = client.post(
            "/procurement/feedback",
            json={
                "trace_id": "TRACE-FB-3",
                "outcome_score": 1.5,  # out of range
                "success": True,
                "latency_ms": 100.0,
            },
        )
        assert response.status_code == 200
        assert response.json()["error"] == "invalid_feedback"

    def test_submit_feedback_missing_trace_id(self, client: TestClient, temp_db: str) -> None:
        response = client.post(
            "/procurement/feedback",
            json={
                "outcome_score": 0.85,
                "success": True,
                "latency_ms": 100.0,
            },
        )
        assert response.status_code == 200
        assert response.json()["error"] == "invalid_feedback"
