"""Unit tests for the Phase 3 EvaluationAgent (per-supplier evaluation)."""

import pytest

from swarm import Event, EventBus, SwarmState
from swarm.domain import ProcurementEventType, SupplierDiscoveryAgent
from swarm.domain.agents import EvaluationAgent
from swarm.domain.artifacts import RequirementArtifact
from tests.unit.procurement_helpers import drive

SUPPLIER_IDS = [
    "MinerCorp_A",
    "DistribCorp_B",
    "RecycleCorp_C",
    "TraderCorp_D",
    "PremiumSteel_E",
]


def seed_requirement(state: SwarmState) -> None:
    state.put_artifact(
        RequirementArtifact(
            data={
                "text": "buy aluminum",
                "constraints": {
                    "material": "aluminum",
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


async def seed_pool(state: SwarmState) -> None:
    seed_requirement(state)
    discovery = SupplierDiscoveryAgent()
    await drive(
        discovery,
        state,
        Event(
            type=ProcurementEventType.REQUIREMENT_CREATED,
            source="requirement_agent",
            payload={"artifact": "requirement"},
            correlation_id="REQ-CONV",
        ),
    )


def discovery_event(supplier_id: str) -> Event:
    """One ``SupplierDiscovered`` event, as the discovery agent would publish."""
    return Event(
        type=ProcurementEventType.SUPPLIER_DISCOVERED,
        source="supplier_discovery_agent",
        payload={"supplier_id": supplier_id, "material": "aluminum", "artifact": "suppliers"},
        correlation_id="REQ-CONV",
    )


def discovery_events() -> list[Event]:
    return [discovery_event(supplier_id) for supplier_id in SUPPLIER_IDS]


@pytest.mark.asyncio
async def test_evaluation_agent_scores_each_supplier_deterministically():
    agent = EvaluationAgent()
    state = SwarmState()
    await seed_pool(state)
    for event in discovery_events():
        await drive(agent, state, event)

    evaluations = state.find_artifacts(kind="evaluation")
    assert len(evaluations) == 5

    scores = {e.data["supplier_id"]: e.data["score"] for e in evaluations}
    assert scores == {
        "MinerCorp_A": 0.858,
        "PremiumSteel_E": 0.803,
        "DistribCorp_B": 0.795,
        "RecycleCorp_C": 0.7687,
        "TraderCorp_D": 0.7263,
    }


@pytest.mark.asyncio
async def test_evaluation_agent_artifacts_are_taggable_named_and_linaged():
    agent = EvaluationAgent()
    state = SwarmState()
    await seed_pool(state)
    await drive(agent, state, discovery_event("MinerCorp_A"))

    miner = state.get_artifact("evaluation_MinerCorp_A")
    assert miner is not None
    assert miner.kind == "evaluation"
    assert miner.tags == {"supplier": "MinerCorp_A"}
    assert miner.parent_ids == ["suppliers"]
    assert miner.correlation_id == "REQ-CONV"
    assert set(miner.data["breakdown"]) == {"price", "lead_time", "esg", "reliability"}
    assert miner.data["bid"]["unit_price"] == 984.0


@pytest.mark.asyncio
async def test_evaluation_agent_ignores_replayed_events():
    agent = EvaluationAgent()
    state = SwarmState()
    await seed_pool(state)
    event = discovery_event("MinerCorp_A").model_copy(update={"replayed": True})
    agent.state = state
    await agent.step(event)

    assert state.find_artifacts(kind="evaluation") == []


@pytest.mark.asyncio
async def test_evaluation_agent_publishes_supplier_evaluated():
    agent = EvaluationAgent()
    bus = EventBus()
    agent.bus = bus
    seen: list[Event] = []

    async def record(event: Event) -> None:
        seen.append(event)

    bus.subscribe(ProcurementEventType.SUPPLIER_EVALUATED, record)
    state = SwarmState()
    await seed_pool(state)
    await drive(agent, state, discovery_event("MinerCorp_A"))

    assert len(seen) == 1
    assert seen[0].correlation_id == "REQ-CONV"
    assert seen[0].payload["supplier_id"] == "MinerCorp_A"
    assert seen[0].payload["artifact"] == "evaluation_MinerCorp_A"
