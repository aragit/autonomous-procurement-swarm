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
