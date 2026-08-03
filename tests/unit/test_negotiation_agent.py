"""Unit tests for the Phase 3 NegotiationAgent (per-supplier quotes)."""

import pytest

from swarm import Event, EventBus, SwarmState
from swarm.domain import ProcurementEventType, SupplierDiscoveryAgent
from swarm.domain.agents import EvaluationAgent, NegotiationAgent
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


async def seed_evaluations(state: SwarmState) -> None:
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
    evaluation = EvaluationAgent()
    for supplier_id in SUPPLIER_IDS:
        await drive(
            evaluation,
            state,
            Event(
                type=ProcurementEventType.SUPPLIER_DISCOVERED,
                source="supplier_discovery_agent",
                payload={
                    "supplier_id": supplier_id,
                    "material": "aluminum",
                    "artifact": "suppliers",
                },
                correlation_id="REQ-CONV",
            ),
        )


def evaluation_event(supplier_id: str) -> Event:
    """One ``SupplierEvaluated`` event, as the evaluation agent would publish."""
    return Event(
        type=ProcurementEventType.SUPPLIER_EVALUATED,
        source="evaluation_agent",
        payload={
            "supplier_id": supplier_id,
            "artifact": f"evaluation_{supplier_id}",
        },
        correlation_id="REQ-CONV",
    )


@pytest.mark.asyncio
async def test_negotiation_agent_creates_one_quote_per_supplier():
    agent = NegotiationAgent()
    state = SwarmState()
    await seed_evaluations(state)
    for supplier_id in SUPPLIER_IDS:
        await drive(agent, state, evaluation_event(supplier_id))

    quotes = state.find_artifacts(kind="quote")
    assert len(quotes) == 5

    prices = {q.data["supplier_id"]: q.data["price"] for q in quotes}
    assert prices == {
        "MinerCorp_A": 984.0,
        "PremiumSteel_E": 1534.0,
        "DistribCorp_B": 1870.4,
        "RecycleCorp_C": 2058.5,
        "TraderCorp_D": 2149.2,
    }


@pytest.mark.asyncio
async def test_negotiation_agent_quote_contract():
    agent = NegotiationAgent()
    state = SwarmState()
    await seed_evaluations(state)
    await drive(agent, state, evaluation_event("MinerCorp_A"))

    miner = state.get_artifact("quote_MinerCorp_A")
    assert miner is not None
    assert miner.kind == "quote"
    assert miner.tags == {"supplier": "MinerCorp_A"}
    assert miner.parent_ids == ["evaluation_MinerCorp_A"]
    assert miner.correlation_id == "REQ-CONV"
    assert miner.data["terms"] == "net_30"
    assert miner.data["metadata"]["quantity"] == 1000
    assert miner.data["metadata"]["lead_time_days"] == 28
    assert miner.data["metadata"]["reliability_score"] == 0.85


@pytest.mark.asyncio
async def test_negotiation_agent_ignores_replayed_events():
    agent = NegotiationAgent()
    state = SwarmState()
    await seed_evaluations(state)
    event = evaluation_event("MinerCorp_A").model_copy(update={"replayed": True})
    agent.state = state
    await agent.step(event)

    assert state.find_artifacts(kind="quote") == []


@pytest.mark.asyncio
async def test_negotiation_agent_publishes_quote_generated():
    agent = NegotiationAgent()
    bus = EventBus()
    agent.bus = bus
    seen: list[Event] = []

    async def record(event: Event) -> None:
        seen.append(event)

    bus.subscribe(ProcurementEventType.QUOTE_GENERATED, record)
    state = SwarmState()
    await seed_evaluations(state)
    await drive(agent, state, evaluation_event("MinerCorp_A"))

    assert len(seen) == 1
    assert seen[0].correlation_id == "REQ-CONV"
    assert seen[0].payload["supplier_id"] == "MinerCorp_A"
    assert seen[0].payload["artifact"] == "quote_MinerCorp_A"
