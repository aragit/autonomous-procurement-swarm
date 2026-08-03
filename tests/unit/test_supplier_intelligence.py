"""Unit tests for the Phase 5 SupplierIntelligenceAgent."""

import pytest

from swarm import Event, EventBus, SwarmState
from swarm.domain.agents import SupplierIntelligenceAgent
from swarm.domain.artifacts import (
    DecisionArtifact,
    OutcomeArtifact,
    SupplierPerformanceArtifact,
    outcome_artifact_name,
)
from swarm.domain.events import ProcurementEventType
from swarm.memory import SupplierMemoryStore
from tests.unit.procurement_helpers import drive

DECISION_ID = "decision-uuid-1234"
DECISION_NAME = "decision"
SUPPLIER = "MinerCorp_A"


def seed_decision_and_outcome(
    state: SwarmState,
    *,
    quality_score: float = 0.92,
    delivered_on_time: bool = True,
    actual_price: float = 984.0,
    carbon_score: float = 1800.0,
) -> str:
    decision = DecisionArtifact(
        id=DECISION_ID,
        data={
            "selected_supplier": SUPPLIER,
            "reasoning": {
                "criteria": "score",
                "ranked": [
                    {
                        "supplier_id": SUPPLIER,
                        "score": 0.858,
                        "price": 984.0,
                        "policy_passed": True,
                        "policy_reason": "POLICY_PASSED",
                    }
                ],
            },
        },
        parent_ids=[],
        created_by="decision_agent",
        correlation_id="REQ-INT-01",
    )
    state.put_artifact(decision)

    outcome_name = outcome_artifact_name(DECISION_ID)
    outcome = OutcomeArtifact(
        name=outcome_name,
        data={
            "supplier_id": SUPPLIER,
            "decision_id": DECISION_ID,
            "delivered_on_time": delivered_on_time,
            "quality_score": quality_score,
            "actual_price": actual_price,
            "carbon_score": carbon_score,
        },
        parent_ids=[DECISION_ID],
        tags={"supplier": SUPPLIER},
        created_by="outcome_agent",
        correlation_id="REQ-INT-02",
    )
    state.put_artifact(outcome)

    return outcome_name


def outcome_recorded_event(outcome_name: str, cid: str = "REQ-INT-02") -> Event:
    return Event(
        type=ProcurementEventType.OUTCOME_RECORDED,
        source="outcome_agent",
        payload={"supplier_id": SUPPLIER, "decision_id": DECISION_ID, "artifact": outcome_name},
        correlation_id=cid,
    )


def test_performance_artifact_kind_and_name() -> None:
    perf = SupplierPerformanceArtifact(
        name="performance_MinerCorp_A",
        data={
            "supplier_id": SUPPLIER,
            "performance_metrics": {"total_orders": 1},
            "order_count": 1,
            "updated_at": "2026-01-01T00:00:00+00:00",
        },
        created_by="supplier_intelligence_agent",
    )
    assert perf.kind == "supplier_performance"
    assert perf.name == "performance_MinerCorp_A"


@pytest.mark.asyncio
async def test_intelligence_agent_creates_performance_artifact_and_updates_memory() -> None:
    store = SupplierMemoryStore()
    agent = SupplierIntelligenceAgent(memory=store)
    state = SwarmState()
    outcome_name = seed_decision_and_outcome(state)
    await drive(agent, state, outcome_recorded_event(outcome_name))

    perf_artifact = state.get_artifact("performance_MinerCorp_A")
    assert perf_artifact is not None
    assert perf_artifact.kind == "supplier_performance"
    assert perf_artifact.parent_ids == [state.get_artifact(outcome_name).id]
    data = perf_artifact.data
    assert data["supplier_id"] == SUPPLIER
    assert data["order_count"] == 1
    assert data["performance_metrics"]["total_orders"] == 1
    assert data["performance_metrics"]["average_delivery_score"] == 1.0
    assert data["performance_metrics"]["average_quality_score"] == 0.92

    in_memory = store.get_supplier_performance(SUPPLIER)
    assert in_memory is not None
    assert in_memory.total_orders == 1
    assert in_memory.delivery_reliability == 1.0


@pytest.mark.asyncio
async def test_intelligence_agent_publishes_performance_updated() -> None:
    store = SupplierMemoryStore()
    agent = SupplierIntelligenceAgent(memory=store)
    bus = EventBus()
    agent.bus = bus
    seen: list[Event] = []

    async def record(event: Event) -> None:
        seen.append(event)

    bus.subscribe(ProcurementEventType.SUPPLIER_PERFORMANCE_UPDATED, record)
    state = SwarmState()
    outcome_name = seed_decision_and_outcome(state)
    await drive(agent, state, outcome_recorded_event(outcome_name))

    assert len(seen) == 1
    assert seen[0].type == ProcurementEventType.SUPPLIER_PERFORMANCE_UPDATED
    assert seen[0].payload["supplier_id"] == SUPPLIER
    assert seen[0].payload["total_orders"] == 1


@pytest.mark.asyncio
async def test_intelligence_agent_without_memory_is_noop() -> None:
    agent = SupplierIntelligenceAgent(memory=None)
    state = SwarmState()
    outcome_name = seed_decision_and_outcome(state)
    await drive(agent, state, outcome_recorded_event(outcome_name))

    assert state.get_artifact("performance_MinerCorp_A") is None


@pytest.mark.asyncio
async def test_intelligence_agent_ignores_replayed_events() -> None:
    store = SupplierMemoryStore()
    agent = SupplierIntelligenceAgent(memory=store)
    state = SwarmState()
    outcome_name = seed_decision_and_outcome(state)
    event = outcome_recorded_event(outcome_name).model_copy(update={"replayed": True})
    agent.state = state
    await agent.step(event)

    assert state.get_artifact("performance_MinerCorp_A") is None
    assert store.get_supplier_performance(SUPPLIER) is None


@pytest.mark.asyncio
async def test_intelligence_agent_accumulates_multiple_outcomes_deterministically() -> None:
    store = SupplierMemoryStore()
    agent = SupplierIntelligenceAgent(memory=store)
    state = SwarmState()

    # two outcomes for the same supplier via two different decisions
    out1 = seed_decision_and_outcome(
        state, quality_score=0.8, delivered_on_time=False, actual_price=1000.0
    )
    await drive(agent, state, outcome_recorded_event(out1, cid="REQ-INT-A"))

    state.put_artifact(
        DecisionArtifact(
            id="decision-uuid-2",
            data={"selected_supplier": SUPPLIER, "reasoning": {"criteria": "score", "ranked": []}},
            parent_ids=[],
            created_by="decision_agent",
            correlation_id="REQ-INT-B",
        )
    )
    out2_name = outcome_artifact_name("decision-uuid-2")
    state.put_artifact(
        OutcomeArtifact(
            name=out2_name,
            data={
                "supplier_id": SUPPLIER,
                "decision_id": "decision-uuid-2",
                "delivered_on_time": True,
                "quality_score": 0.92,
                "actual_price": 984.0,
                "carbon_score": 1700.0,
            },
            parent_ids=["decision-uuid-2"],
            tags={"supplier": SUPPLIER},
            created_by="outcome_agent",
            correlation_id="REQ-INT-B",
        )
    )
    await drive(agent, state, outcome_recorded_event(out2_name, cid="REQ-INT-B"))

    perf = store.get_supplier_performance(SUPPLIER)
    assert perf.total_orders == 2
    assert perf.successful_orders == 1
    assert perf.delivery_reliability == 0.5
    assert perf.average_quality_score == pytest.approx((0.8 + 0.92) / 2)
    assert perf.average_carbon_score == 1750.0
