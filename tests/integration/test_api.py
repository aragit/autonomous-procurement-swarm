"""Integration tests for FastAPI endpoints."""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from api.main import app


@pytest_asyncio.fixture
async def client(test_ledger):
    """Async HTTP client for FastAPI testing.

    ASGITransport does not run the app lifespan, so the module-global ledger
    would stay None (all ledger endpoints return 503). Wire the test ledger in.
    """
    import api.main as main

    main.ledger = test_ledger
    from core.memory.heuristics import HeuristicReservationEstimator

    main.shared_memory = HeuristicReservationEstimator()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_check(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


@pytest.mark.asyncio
async def test_create_auction(client):
    response = await client.post(
        "/auctions",
        json={
            "material": "steel",
            "quantity": 100,
            "supplier_count": 3,
            "enable_bartering": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["status"] == "AWARDED"
    assert len(data["scored_bids"]) == 3


@pytest.mark.asyncio
async def test_create_auction_with_bartering(client):
    response = await client.post(
        "/auctions",
        json={
            "material": "aluminum",
            "quantity": 500,
            "supplier_count": 5,
            "enable_bartering": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["AWARDED", "TERMINATED"]
    assert "bartering_result" in data


@pytest.mark.asyncio
async def test_get_auction_not_found(client):
    response = await client.get("/auctions/nonexistent")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_ledger_stats(client):
    # First create an auction to populate ledger
    await client.post(
        "/auctions",
        json={
            "material": "copper",
            "quantity": 100,
            "supplier_count": 2,
            "enable_bartering": False,
        },
    )

    response = await client.get("/ledger/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total_events"] >= 2  # RFQ + at least one bid


@pytest.mark.asyncio
async def test_supplier_profile_endpoint(client):
    # Run an auction with bartering so memory gets populated
    await client.post(
        "/auctions",
        json={
            "material": "aluminum",
            "quantity": 300,
            "supplier_count": 3,
            "enable_bartering": True,
        },
    )
    response = await client.get("/suppliers")
    assert response.status_code == 200
    suppliers = response.json()["suppliers"]
    assert len(suppliers) >= 1

    # Profile endpoint for an arbitrary supplier
    response = await client.get("/suppliers/does_not_exist/profile")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_supplier_similar_endpoint(client):
    response = await client.get("/suppliers/similar")
    assert response.status_code == 503  # vector store not wired in tests


# ─── Deterministic swarm trace endpoints ────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_swarm_requirement_returns_traceable_ids(client):
    response = await client.post(
        "/swarm/requirements",
        json={"material": "aluminum", "quantity": 1000, "budget": 2_000_000.0},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["request_id"].startswith("REQ-")
    assert data["correlation_id"].endswith("-CONV")
    assert data["decision"]["selected_supplier"] == "MinerCorp_A"
    assert data["completions"] == {
        data["correlation_id"]: ["evaluation", "quote"],
    }
    assert data["event_count"] > 0
    assert data["artifact_count"] >= 5


@pytest.mark.asyncio
async def test_get_swarm_execution_trace(client):
    dispatch = await client.post(
        "/swarm/requirements",
        json={"material": "aluminum", "quantity": 1000},
    )
    request_id = dispatch.json()["request_id"]

    response = await client.get(f"/swarm/trace/{request_id}")
    assert response.status_code == 200
    trace = response.json()
    assert trace["correlation_id"].endswith("-CONV")
    assert any(event["type"] == "QuoteGenerated" for event in trace["events"])
    assert any(action["action"] == "artifact_created" for action in trace["agent_actions"])
    assert any(artifact["kind"] == "quote" for artifact in trace["artifacts"])


@pytest.mark.asyncio
async def test_get_swarm_completions_and_state(client):
    dispatch = await client.post(
        "/swarm/requirements",
        json={"material": "steel", "quantity": 500, "budget": 500_000.0},
    )
    request_id = dispatch.json()["request_id"]
    correlation_id = dispatch.json()["correlation_id"]

    completions = await client.get(f"/swarm/trace/{request_id}/completions")
    assert completions.status_code == 200
    assert completions.json()["completions"][correlation_id] == ["evaluation", "quote"]

    state = await client.get(f"/swarm/state/{request_id}")
    assert state.status_code == 200
    assert state.json()["request_id"] == request_id
    assert state.json()["completions"][correlation_id] == ["evaluation", "quote"]


@pytest.mark.asyncio
async def test_swarm_trace_unknown_request_id_returns_404(client):
    response = await client.get("/swarm/trace/DOES-NOT-EXIST")
    assert response.status_code == 404
