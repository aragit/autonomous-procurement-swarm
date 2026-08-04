"""API integration tests for the Phase 8 external-call audit + sync endpoints.

These exercise /swarm/requirements -> /execute -> /external/{id} -> /sync
end-to-end. They do NOT depend on Postgres: dispatch and execution use the
in-memory supplier store and the deterministic MockConnector.

Note: /swarm/requirements runs the *full* deterministic flow (decision ->
contract -> risk -> governance -> approval -> order -> execution), so after a
dispatch the state already carries the purchase order, execution status and the
external-call audit trail. /execute and /sync are therefore idempotent no-ops on
a freshly dispatched run.
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from api.main import app


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture(autouse=True)
def _clean_swarm_states():
    from api.main import swarm_states

    swarm_states.clear()
    yield
    swarm_states.clear()


def _payload():
    return {"material": "aluminum", "quantity": 100, "budget": 500_000.0}


@pytest.mark.asyncio
async def test_external_audit_trail_and_idempotent_execute_sync(client) -> None:
    dispatched = await client.post("/swarm/requirements", json=_payload())
    assert dispatched.status_code == 200, dispatched.text
    request_id = dispatched.json()["request_id"]

    # Dispatch already executed: one submit + one status check are audited.
    resp = await client.get(f"/swarm/external/{request_id}")
    assert resp.status_code == 200
    calls = resp.json()["external_calls"]
    assert {c["action"] for c in calls} == {"submit_order", "get_order_status"}
    assert all(c["system"] == "mock" for c in calls)

    # /execute is idempotent: no new external calls, PO status unchanged.
    executed = await client.post(
        f"/swarm/{request_id}/execute", json={"approver": "governance_sim"}
    )
    assert executed.status_code == 200, executed.text
    assert executed.json()["purchase_order"]["status"] == "SUBMITTED"
    assert len(executed.json()["external_calls"]) == len(calls)

    # /sync re-reconciles but stays idempotent: same count, status DELIVERED.
    synced = await client.post(f"/swarm/{request_id}/sync", json={"force": True})
    assert synced.status_code == 200, synced.text
    assert synced.json()["execution_status"]["status"] == "DELIVERED"
    assert len(synced.json()["external_calls"]) == len(calls)


@pytest.mark.asyncio
async def test_external_endpoint_404_for_unknown_run(client) -> None:
    resp = await client.get("/swarm/external/NOPE")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_sync_requires_known_run(client) -> None:
    resp = await client.post("/swarm/NOPE/sync", json={"force": True})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_sync_rejects_unsupported_connector_override(client) -> None:
    r = await client.post("/swarm/requirements", json=_payload())
    request_id = r.json()["request_id"]
    resp = await client.post(
        f"/swarm/{request_id}/sync", json={"connector": "unknown_erp"}
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_sync_accepts_runtime_connector_override(client) -> None:
    """The /sync connector override is now runtime-resolved via build_connector."""
    r = await client.post("/swarm/requirements", json=_payload())
    request_id = r.json()["request_id"]
    resp = await client.post(
        f"/swarm/{request_id}/sync", json={"connector": "supplier_api"}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["execution_status"] is not None
