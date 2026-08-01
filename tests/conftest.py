"""Shared pytest fixtures."""

import pytest
import pytest_asyncio

from core.ledger.repository import PostgresLedgerRepository

TEST_DB_URL = "postgresql+asyncpg://procurement:procurement@localhost:5433/procurement"


@pytest_asyncio.fixture
async def test_ledger():
    """Provide a fresh PostgreSQL ledger repository for tests."""
    repo = PostgresLedgerRepository(TEST_DB_URL)
    await repo.init_schema()
    yield repo
    await repo.close()


@pytest.fixture
def mock_llm():
    from core.llm_engine import LLMEngineFactory

    return LLMEngineFactory.create(use_mock=True)
