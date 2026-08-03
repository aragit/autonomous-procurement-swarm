"""Unit tests for Phase 4 DecisionExplanationArtifact generation."""

import pytest

from swarm import Event, EventBus, SwarmState
from swarm.domain import (
    ProcurementEventType,
    StrategyAgent,
    SupplierDiscoveryAgent,
)
from swarm.domain.agents import DecisionAgent, EvaluationAgent, NegotiationAgent
from swarm.domain.artifacts import (
    DECISION_EXPLANATION_ARTIFACT_NAME,
    RequirementArtifact,
)
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
                    "max_carbon_per_unit": 800.0,
                },
                "metadata": {},
            },
            created_by="requirement_agent",
            correlation_id="REQ-CONV",
        )
    )


async def seed_strategy(state: SwarmState) -> str:
    """Run the StrategyAgent so the swarm state carries a real strategy."""
    agent = StrategyAgent()
    await drive(
        agent,
        state,
        Event(
            type=ProcurementEventType.REQUIREMENT_CREATED,
            source="requirement_agent",
            payload={"artifact": "requirement"},
            correlation_id="REQ-CONV",
        ),
    )
    strategy = state.get_artifact("strategy")
    assert strategy is not None
    return str(strategy.data["strategy_name"])


async def seed_quotes(state: SwarmState) -> list[str]:
    seed_requirement(state)
    await seed_strategy(state)
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
async def test_decision_agent_publishes_explanation_artifact():
    agent = DecisionAgent()
    state = SwarmState()
    await seed_quotes(state)
    await drive(agent, state, decision_event())

    explanation = state.get_artifact(DECISION_EXPLANATION_ARTIFACT_NAME)
    assert explanation is not None
    assert explanation.kind == "decision_explanation"
    assert explanation.correlation_id == "REQ-CONV"
    assert explanation.parent_ids == ["decision"]


@pytest.mark.asyncio
async def test_explanation_describes_selection_and_strategy():
    agent = DecisionAgent()
    state = SwarmState()
    await seed_quotes(state)
    await drive(agent, state, decision_event())

    explanation = state.get_artifact(DECISION_EXPLANATION_ARTIFACT_NAME)
    assert explanation is not None
    data = explanation.data
    assert data["selected_supplier"] == "MinerCorp_A"
    assert data["strategy_used"] == "low_carbon"
    assert any("low_carbon" in factor for factor in data["top_factors"])


@pytest.mark.asyncio
async def test_explanation_lists_every_rejected_supplier_with_reason():
    agent = DecisionAgent()
    state = SwarmState()
    await seed_quotes(state)
    await drive(agent, state, decision_event())

    explanation = state.get_artifact(DECISION_EXPLANATION_ARTIFACT_NAME)
    assert explanation is not None
    rejected = explanation.data["rejected_suppliers"]
    assert len(rejected) == 4
    assert {entry["supplier_id"] for entry in rejected} == {
        "DistribCorp_B",
        "RecycleCorp_C",
        "TraderCorp_D",
        "PremiumSteel_E",
    }
    assert all(entry["reason"] for entry in rejected)
    assert all(
        set(entry) >= {"supplier_id", "score", "price", "policy_passed", "policy_reason", "reason"}
        for entry in rejected
    )


@pytest.mark.asyncio
async def test_explanation_reasoning_is_deterministic():
    agent = DecisionAgent()
    state = SwarmState()
    await seed_quotes(state)
    await drive(agent, state, decision_event())

    state2 = SwarmState()
    await seed_quotes(state2)
    await drive(DecisionAgent(), state2, decision_event())

    first = state.get_artifact(DECISION_EXPLANATION_ARTIFACT_NAME).data
    second = state2.get_artifact(DECISION_EXPLANATION_ARTIFACT_NAME).data
    assert first == second


@pytest.mark.asyncio
async def test_explanation_falls_back_to_balanced_without_strategy():
    state = SwarmState()
    seed_requirement(state)
    await seed_quotes_without_strategy(state)
    agent = DecisionAgent()
    await drive(agent, state, decision_event())

    explanation = state.get_artifact(DECISION_EXPLANATION_ARTIFACT_NAME)
    assert explanation is not None
    assert explanation.data["strategy_used"] == "balanced"


async def seed_quotes_without_strategy(state: SwarmState) -> None:
    """Full quote pipeline without running the StrategyAgent."""
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


@pytest.mark.asyncio
async def test_explanation_publication_is_idempotent_via_bus():
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
    assert seen[0].payload["selected_supplier"] == "MinerCorp_A"
