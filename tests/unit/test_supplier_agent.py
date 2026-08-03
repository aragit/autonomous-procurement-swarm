"""Unit tests for the Phase 2 SupplierDiscoveryAgent."""

import pytest

from swarm import Event, EventBus, SwarmState
from swarm.domain import ProcurementEventType, SupplierDiscoveryAgent
from swarm.domain.artifacts import RequirementArtifact
from tests.unit.procurement_helpers import drive

REQUIREMENT_EVENT = Event(
    type=ProcurementEventType.REQUIREMENT_CREATED,
    source="requirement_agent",
    payload={"artifact": "requirement"},
    correlation_id="REQ-CONV",
)


def seed_requirement(state: SwarmState, material: str = "aluminum") -> None:
    state.put_artifact(
        RequirementArtifact(
            data={
                "text": "buy aluminum",
                "constraints": {
                    "material": material,
                    "quantity": 1000,
                    "budget": 2_000_000.0,
                    "max_unit_price": 2640.0,
                    "target_lead_time_days": 30,
                },
                "metadata": {},
            },
            created_by="requirement_agent",
            correlation_id="REQ-CONV",
        )
    )


@pytest.mark.asyncio
async def test_supplier_discovery_agent_builds_deterministic_pool():
    agent = SupplierDiscoveryAgent()
    state = SwarmState()
    seed_requirement(state)
    await drive(agent, state, REQUIREMENT_EVENT)

    pool = state.get_artifact("suppliers")
    assert pool is not None
    assert pool.kind == "supplier_list"
    assert pool.correlation_id == "REQ-CONV"
    assert pool.data["material"] == "aluminum"
    assert pool.data["quantity"] == 1000
    assert pool.data["spot_price"] == 2200.0

    suppliers = pool.data["suppliers"]
    assert [s["supplier_id"] for s in suppliers] == [
        "MinerCorp_A",
        "DistribCorp_B",
        "RecycleCorp_C",
        "TraderCorp_D",
        "PremiumSteel_E",
    ]
    miner, _, _, _, premium = suppliers
    assert miner["base_cost_per_unit"] == 770.0  # 2200 * 0.35
    assert miner["logistics_premium_per_unit"] == 50
    assert miner["min_margin_pct"] == 0.20
    assert premium["base_cost_per_unit"] == 1100.0  # 2200 * 0.50


@pytest.mark.asyncio
async def test_supplier_discovery_agent_falls_back_to_steel_for_unknown_material():
    agent = SupplierDiscoveryAgent()
    state = SwarmState()
    seed_requirement(state, material="unobtainium")
    await drive(agent, state, REQUIREMENT_EVENT)

    pool = state.get_artifact("suppliers")
    assert pool.data["material"] == "steel"
    assert pool.data["spot_price"] == 450.0


@pytest.mark.asyncio
async def test_supplier_discovery_agent_ignores_replayed_events():
    agent = SupplierDiscoveryAgent()
    state = SwarmState()
    seed_requirement(state)
    event = REQUIREMENT_EVENT.model_copy(update={"replayed": True})
    agent.state = state
    await agent.step(event)

    assert state.get_artifact("suppliers") is None


@pytest.mark.asyncio
async def test_supplier_discovery_agent_publishes_per_supplier_events():
    agent = SupplierDiscoveryAgent()
    bus = EventBus()
    agent.bus = bus
    seen: list[Event] = []

    async def record(event: Event) -> None:
        seen.append(event)

    bus.subscribe(ProcurementEventType.SUPPLIER_DISCOVERED, record)
    state = SwarmState()
    seed_requirement(state)
    await drive(agent, state, REQUIREMENT_EVENT)

    assert len(seen) == 5
    assert [event.correlation_id for event in seen] == ["REQ-CONV"] * 5
    assert [event.payload["supplier_id"] for event in seen] == [
        "MinerCorp_A",
        "DistribCorp_B",
        "RecycleCorp_C",
        "TraderCorp_D",
        "PremiumSteel_E",
    ]
    assert seen[0].payload["artifact"] == "suppliers"


@pytest.mark.asyncio
async def test_supplier_discovery_agent_declares_completion_expectations():
    agent = SupplierDiscoveryAgent()
    state = SwarmState()
    seed_requirement(state)
    await drive(agent, state, REQUIREMENT_EVENT)

    assert state.expectations["REQ-CONV"]["evaluation"] == 5
    assert state.expectations["REQ-CONV"]["quote"] == 5


@pytest.mark.asyncio
async def test_supplier_pool_artifact_records_parent_lineage():
    agent = SupplierDiscoveryAgent()
    state = SwarmState()
    seed_requirement(state)
    await drive(agent, state, REQUIREMENT_EVENT)

    pool = state.get_artifact("suppliers")
    assert pool.parent_ids == ["requirement"]
