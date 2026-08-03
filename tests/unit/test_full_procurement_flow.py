"""End-to-end Phase 3 flow: parallel, per-supplier events with completion gates."""

import pytest

from swarm import Swarm, SwarmEventType
from swarm.domain import (
    CREATE_REQUIREMENT_INTENT,
    ProcurementEventType,
)
from swarm.domain.agents import (
    DecisionAgent,
    EvaluationAgent,
    NegotiationAgent,
    RequirementAgent,
    SupplierDiscoveryAgent,
)
from swarm.domain.wiring import build_procurement_swarm

DOMAIN_EVENTS = {
    ProcurementEventType.REQUIREMENT_CREATED,
    ProcurementEventType.SUPPLIER_DISCOVERED,
    ProcurementEventType.SUPPLIER_EVALUATED,
    ProcurementEventType.QUOTE_GENERATED,
    ProcurementEventType.EVALUATION_COMPLETED,
    ProcurementEventType.QUOTES_COMPLETED,
    ProcurementEventType.DECISION_MADE,
}

PER_SUPPLIER_EVENTS = {
    ProcurementEventType.SUPPLIER_DISCOVERED,
    ProcurementEventType.SUPPLIER_EVALUATED,
    ProcurementEventType.QUOTE_GENERATED,
}

REQUEST = {
    "text": "Source 1000 units of aluminum at market price",
    "material": "aluminum",
    "quantity": 1000,
    "budget": 2_000_000.0,
    "target_lead_time_days": 30,
}


def build_swarm() -> Swarm:
    return build_procurement_swarm(request_id="REQ-002", goal="Source 1000 units of aluminum")


@pytest.mark.asyncio
async def test_full_procurement_flow_reaches_a_decision():
    swarm = build_swarm()
    await swarm.start()
    errors = await swarm.send_message(
        CREATE_REQUIREMENT_INTENT,
        REQUEST,
        sender="user",
        correlation_id="REQ-002-CONV",
    )
    await swarm.shutdown()

    assert errors == []
    state = swarm.state

    requirement = state.get_artifact("requirement")
    pool = state.get_artifact("suppliers")
    decision = state.get_artifact("decision")
    assert requirement is not None
    assert pool is not None
    assert decision is not None

    assert len(pool.data["suppliers"]) == 5
    assert len(state.find_artifacts(kind="evaluation")) == 5
    assert len(state.find_artifacts(kind="quote")) == 5
    assert decision.data["selected_supplier"] == "MinerCorp_A"


@pytest.mark.asyncio
async def test_full_procurement_flow_emits_per_supplier_and_completion_events():
    swarm = build_swarm()
    await swarm.start()
    await swarm.send_message(
        CREATE_REQUIREMENT_INTENT,
        {"material": "aluminum", "quantity": 1000, "budget": 2_000_000.0},
        sender="user",
        correlation_id="REQ-002-CONV",
    )
    await swarm.shutdown()

    types = [event.type for event in swarm.state.events]
    for domain_event in DOMAIN_EVENTS:
        assert domain_event in types

    per_supplier = {
        event_type: types.count(event_type)
        for event_type in PER_SUPPLIER_EVENTS
    }
    assert per_supplier == {
        ProcurementEventType.SUPPLIER_DISCOVERED: 5,
        ProcurementEventType.SUPPLIER_EVALUATED: 5,
        ProcurementEventType.QUOTE_GENERATED: 5,
    }

    assert types.index(ProcurementEventType.REQUIREMENT_CREATED) < types.index(
        ProcurementEventType.DECISION_MADE
    )
    assert types.index(ProcurementEventType.EVALUATION_COMPLETED) < types.index(
        ProcurementEventType.QUOTES_COMPLETED
    )


@pytest.mark.asyncio
async def test_full_procurement_flow_propagates_correlation_id():
    swarm = build_swarm()
    await swarm.start()
    await swarm.send_message(
        CREATE_REQUIREMENT_INTENT,
        {"material": "aluminum", "budget": 2_000_000.0},
        sender="user",
        correlation_id="REQ-002-CONV",
    )
    await swarm.shutdown()

    for event in swarm.state.events:
        if event.type in DOMAIN_EVENTS:
            assert event.correlation_id == "REQ-002-CONV"
    assert swarm.state.get_artifact("decision").correlation_id == "REQ-002-CONV"


@pytest.mark.asyncio
async def test_full_procurement_flow_tracks_completions():
    swarm = build_swarm()
    await swarm.start()
    await swarm.send_message(
        CREATE_REQUIREMENT_INTENT,
        {"material": "aluminum", "budget": 2_000_000.0},
        sender="user",
        correlation_id="REQ-002-CONV",
    )
    await swarm.shutdown()

    state = swarm.state
    assert state.expectations["REQ-002-CONV"] == {"evaluation": 5, "quote": 5}
    assert state.completions["REQ-002-CONV"] == ["evaluation", "quote"]


@pytest.mark.asyncio
async def test_full_procurement_flow_is_replay_safe():
    swarm = build_swarm()
    await swarm.start()
    await swarm.send_message(
        CREATE_REQUIREMENT_INTENT,
        {"material": "aluminum", "budget": 2_000_000.0},
        sender="user",
        correlation_id="REQ-002-CONV",
    )
    before = swarm.state.to_dict()
    await swarm.bus.replay(mode="read_only")
    after = swarm.state.to_dict()
    await swarm.shutdown()

    assert before == after


@pytest.mark.asyncio
async def test_send_message_accepts_raw_create_requirement():
    """The runtime-level entry point used by the demo works without Message."""
    swarm = build_swarm()
    await swarm.start()
    await swarm.send_message(CREATE_REQUIREMENT_INTENT, {"material": "steel", "quantity": 100})
    await swarm.shutdown()

    decision = swarm.state.get_artifact("decision")
    assert decision is not None
    # For steel the ESG baseline favors the low-carbon supplier.
    assert decision.data["selected_supplier"] == "RecycleCorp_C"


@pytest.mark.asyncio
async def test_manual_registration_matches_wiring_subscriptions():
    """The wiring module's subscriptions mirror the manual Phase 2 style."""
    swarm = Swarm(request_id="REQ-002", goal="Source 1000 units of aluminum")
    swarm.register(RequirementAgent(), event_types=[SwarmEventType.MESSAGE])
    swarm.register(
        SupplierDiscoveryAgent(), event_types=[ProcurementEventType.REQUIREMENT_CREATED]
    )
    swarm.register(EvaluationAgent(), event_types=[ProcurementEventType.SUPPLIER_DISCOVERED])
    swarm.register(NegotiationAgent(), event_types=[ProcurementEventType.SUPPLIER_EVALUATED])
    swarm.register(DecisionAgent(), event_types=[ProcurementEventType.QUOTES_COMPLETED])
    assert set(swarm.registry.names()) == {
        "requirement_agent",
        "supplier_discovery_agent",
        "evaluation_agent",
        "negotiation_agent",
        "decision_agent",
    }
