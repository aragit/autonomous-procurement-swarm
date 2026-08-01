"""Load test: verify system handles concurrent auctions."""

import asyncio

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

import api.main as main
from api.main import app

TEST_DB_URL = "postgresql+asyncpg://procurement:procurement@localhost:5433/procurement"


@pytest_asyncio.fixture
async def load_app():
    """Wire the in-process app to a real PostgreSQL ledger + vector store.

    ASGITransport does not run the app lifespan, so the module-global
    ledger / memory / vector_store start out None. Initialize them here
    using the same database the local compose stack exposes on 5433.
    """
    from core.ledger.repository import PostgresLedgerRepository
    from core.memory.heuristics import HeuristicReservationEstimator
    from core.memory.semantic import PgVectorMemoryStore

    main.ledger = PostgresLedgerRepository(TEST_DB_URL)
    await main.ledger.init_schema()
    main.shared_memory = HeuristicReservationEstimator()
    main.shared_vector_store = PgVectorMemoryStore(TEST_DB_URL)
    await main.shared_vector_store.init_schema()

    yield app

    await main.ledger.close()
    await main.shared_vector_store.engine.dispose()
    main.ledger = None
    main.shared_memory = None
    main.shared_vector_store = None


@pytest.mark.asyncio
async def test_concurrent_auctions(load_app):
    """Run 10 auctions concurrently via API."""
    transport = ASGITransport(app=load_app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Warmup health check
        response = await client.get("/health")
        assert response.status_code == 200

        # Fire 10 auctions concurrently
        tasks = [
            client.post(
                "/auctions",
                json={
                    "material": "steel",
                    "quantity": 100,
                    "supplier_count": 3,
                    "enable_bartering": False,
                },
            )
            for _ in range(10)
        ]

        results = await asyncio.gather(*tasks)

        assert all(r.status_code == 200 for r in results)
        data = [r.json() for r in results]

        # All should be awarded (mock suppliers always have capacity)
        assert all(d["status"] == "AWARDED" for d in data)

        # All session IDs unique
        session_ids = [d["session_id"] for d in data]
        assert len(set(session_ids)) == 10

        # Ledger should have 10+ sessions
        stats = await client.get("/ledger/stats")
        assert stats.status_code == 200
        assert stats.json()["total_sessions"] >= 10


@pytest.mark.asyncio
async def test_auction_with_bartering_under_load(load_app):
    """5 auctions with bartering, sequentially to avoid FSM state conflicts."""
    transport = ASGITransport(app=load_app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(5):
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

            # Verify chain integrity
            session_id = data["session_id"]
            chain = await client.get(f"/auctions/{session_id}")
            assert chain.json()["chain_valid"] is True
