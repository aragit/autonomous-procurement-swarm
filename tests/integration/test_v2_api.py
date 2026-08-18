"""Integration tests for the v2 (Mesh Runtime) API endpoints.

These do **not** require Ray: a :class:`FakeRuntime` implementing the
:class:`api.v2.runtime.MeshRuntime` protocol is injected via ``set_runtime`` so
the routes can be exercised end-to-end through ``httpx.ASGITransport``.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from api.v2 import app
from api.v2.models import (
    ProcurementRunResponse,
    ProcurementStatusResponse,
)
from api.v2.runtime import set_runtime


class FakeRuntime:
    """In-memory ``MeshRuntime`` for testing the v2 routes."""

    def __init__(self) -> None:
        self.runs: dict[str, dict] = {}
        self.shutdown_calls = 0

    async def run_procurement(
        self, trace_id: str, requirement: dict
    ) -> ProcurementRunResponse:
        correlation_id = requirement.get("correlation_id", f"{trace_id}-CORR")
        # Simulate a blackboard snapshot / stats view.
        channels = {
            "requirement": 1,
            "discovery": 6,
            "score": 5,
            "risk": 5,
            "deal": 5,
            "decision": 1,
        }
        decision = {
            "selected_supplier": "DistribCorp_B",
            "composite_score": 0.812,
            "method": "deterministic_mcda",
            "ranked": [
                {"supplier_id": "DistribCorp_B", "composite_score": 0.812},
                {"supplier_id": "MinerCorp_A", "composite_score": 0.643},
            ],
        }
        self.runs[trace_id] = {
            "correlation_id": correlation_id,
            "channels": channels,
            "stats": {"total_writes": 23, "total_reads": 31},
            "decision": decision,
            "status": "completed",
        }
        return ProcurementRunResponse(
            trace_id=trace_id,
            correlation_id=correlation_id,
            status="completed",
            decision=decision,
            buyer_result={"status": "success"},
        )

    async def get_status(self, trace_id: str) -> ProcurementStatusResponse | None:
        run = self.runs.get(trace_id)
        if run is None:
            return None
        return ProcurementStatusResponse(
            trace_id=trace_id,
            correlation_id=run["correlation_id"],
            status=run["status"],
            decision=run["decision"],
            channels=run["channels"],
            blackboard_stats=run["stats"],
        )

    async def shutdown(self) -> None:
        self.shutdown_calls += 1


@pytest_asyncio.fixture
async def client():
    """ASGI test client with a FakeRuntime injected."""
    set_runtime(FakeRuntime())
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    set_runtime(None)


# ─── POST /v2/procurement/run ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_procurement_returns_decision(client):
    response = await client.post(
        "/v2/procurement/run",
        json={
            "material": "steel",
            "quantity": 1000,
            "budget": 2_000_000.0,
            "target_lead_time_days": 30,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["trace_id"].startswith("RUN-")
    assert data["correlation_id"] == f"{data['trace_id']}-CORR"
    assert data["status"] == "completed"
    assert data["decision"]["selected_supplier"] == "DistribCorp_B"
    assert data["decision"]["method"] == "deterministic_mcda"


@pytest.mark.asyncio
async def test_run_procurement_rejects_invalid_material(client):
    response = await client.post(
        "/v2/procurement/run",
        json={"material": "unobtanium", "quantity": 10, "budget": 100.0},
    )
    assert response.status_code == 422
    assert "material" in response.json()["detail"][0]["loc"]


@pytest.mark.asyncio
async def test_run_procurement_defaults_applied(client):
    response = await client.post("/v2/procurement/run", json={})
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] is not None


# ─── GET /v2/procurement/{trace_id}/status ──────────────────────────────────


@pytest.mark.asyncio
async def test_status_after_run(client):
    run = await client.post("/v2/procurement/run", json={"material": "steel"})
    trace_id = run.json()["trace_id"]

    status = await client.get(f"/v2/procurement/{trace_id}/status")
    assert status.status_code == 200
    body = status.json()
    assert body["trace_id"] == trace_id
    assert body["status"] == "completed"
    assert body["decision"]["selected_supplier"] == "DistribCorp_B"
    assert body["channels"]["decision"] == 1
    assert body["channels"]["discovery"] == 6
    assert body["blackboard_stats"]["total_writes"] == 23


@pytest.mark.asyncio
async def test_status_unknown_trace_returns_404(client):
    response = await client.get("/v2/procurement/DOES-NOT-EXIST/status")
    assert response.status_code == 404
    assert "trace_id" in response.json()["detail"] or (
        "No procurement run" in response.json()["detail"]
    )


# ─── liveness ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mesh_health(client):
    response = await client.get("/v2/procurement/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


# ─── isolation: no runtime injected -> 500 on run ───────────────────────────


@pytest.mark.asyncio
async def test_run_without_runtime_returns_500():
    """Without an injected runtime the lazy default (Ray) is unavailable."""
    set_runtime(None)
    # Patch RayMeshRuntime to raise so we simulate an environment where Ray
    # is not usable (e.g. not installed or cluster unreachable).
    import api.v2.runtime as rt
    import_original = rt._build_default_runtime
    def _fail():
        raise RuntimeError(
            "Mesh runtime unavailable: 'ray' is not installed. Call "
            "api.v2.runtime.set_runtime(<MeshRuntime>) to inject a runtime."
        )
    rt._build_default_runtime = _fail
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post(
                "/v2/procurement/run", json={"material": "steel", "quantity": 1, "budget": 1.0}
            )
        assert response.status_code == 500
    finally:
        rt._build_default_runtime = import_original
        set_runtime(FakeRuntime())  # restore for any later test
