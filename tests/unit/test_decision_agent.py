"""Unit tests for the Phase 3 DecisionAgent (reacts to QuotesCompleted)."""

import pytest

from swarm import Event, EventBus, SwarmState
from swarm.domain import (
    ProcurementEventType,
    SupplierDiscoveryAgent,
)
from swarm.domain.agents import DecisionAgent, EvaluationAgent, NegotiationAgent
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


async def seed_quotes(state: SwarmState) -> list[str]:
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
    negotiation = NegotiationAgent()
    for supplier_id in SUPPLIER_IDS:
        await drive(
            negotiation,
            state,
            Event(
                type=ProcurementEventType.SUPPLIER_EVALUATED,
                source="evaluation_agent",
                payload={
                    "supplier_id": supplier_id,
                    "artifact": f"evaluation_{supplier_id}",
                },
                correlation_id="REQ-CONV",
            ),
        )
    return [f"quote_{supplier_id}" for supplier_id in SUPPLIER_IDS]


def decision_event() -> Event:
    return Event(
        type=ProcurementEventType.QUOTES_COMPLETED,
        source="completion_tracker",
        payload={"group": "quote", "count": 5},
        correlation_id="REQ-CONV",
    )


@pytest.mark.asyncio
async def test_decision_agent_selects_highest_score_within_policy():
    agent = DecisionAgent()
    state = SwarmState()
    await seed_quotes(state)
    await drive(agent, state, decision_event())

    decision = state.get_artifact("decision")
    assert decision is not None
    assert decision.kind == "decision"
    assert decision.correlation_id == "REQ-CONV"
    assert decision.data["selected_supplier"] == "MinerCorp_A"


@pytest.mark.asyncio
async def test_decision_agent_ranks_and_applies_policy():
    agent = DecisionAgent()
    state = SwarmState()
    await seed_quotes(state)
    await drive(agent, state, decision_event())

    ranked = state.get_artifact("decision").data["reasoning"]["ranked"]
    assert [entry["supplier_id"] for entry in ranked] == [
        "MinerCorp_A",
        "PremiumSteel_E",
        "DistribCorp_B",
        "RecycleCorp_C",
        "TraderCorp_D",
    ]
    by_id = {entry["supplier_id"]: entry for entry in ranked}
    assert by_id["MinerCorp_A"]["policy_reason"] == "POLICY_PASSED"
    assert by_id["RecycleCorp_C"]["policy_reason"] == "REJECT_EXCEEDS_BUDGET"
    assert by_id["TraderCorp_D"]["policy_reason"] == "REJECT_EXCEEDS_BUDGET"
    assert by_id["PremiumSteel_E"]["policy_passed"] is True


@pytest.mark.asyncio
async def test_decision_agent_records_quote_lineage():
    agent = DecisionAgent()
    state = SwarmState()
    quote_names = await seed_quotes(state)
    await drive(agent, state, decision_event())

    decision = state.get_artifact("decision")
    assert decision.parent_ids == quote_names


@pytest.mark.asyncio
async def test_decision_agent_ignores_replayed_events():
    agent = DecisionAgent()
    state = SwarmState()
    await seed_quotes(state)
    event = decision_event().model_copy(update={"replayed": True})
    agent.state = state
    await agent.step(event)

    assert state.get_artifact("decision") is None


@pytest.mark.asyncio
async def test_decision_agent_publishes_decision_made():
    agent = DecisionAgent()
    bus = EventBus()
    agent.bus = bus
    seen: list[Event] = []

    async def record(event: Event) -> None:
        seen.append(event)

    bus.subscribe(ProcurementEventType.DECISION_MADE, record)
    state = SwarmState()
    await seed_quotes(state)
    await drive(agent, state, decision_event())

    assert len(seen) == 1
    assert seen[0].correlation_id == "REQ-CONV"
    assert seen[0].payload["artifact"] == "decision"
    assert seen[0].payload["selected_supplier"] == "MinerCorp_A"
