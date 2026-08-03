"""Unit tests for the Phase 2 RequirementAgent."""

import pytest

from swarm import Event, EventBus, Message, SwarmState
from swarm.domain import CREATE_REQUIREMENT_INTENT, ProcurementEventType, RequirementAgent
from tests.unit.procurement_helpers import drive


def message_event(payload: dict, correlation_id: str = "REQ-CONV") -> Event:
    message = Message(
        sender="user",
        intent=CREATE_REQUIREMENT_INTENT,
        payload=payload,
        correlation_id=correlation_id,
    )
    return Event.from_message(message)


@pytest.mark.asyncio
async def test_requirement_agent_builds_requirement_artifact():
    agent = RequirementAgent()
    state = SwarmState()
    payload = {"text": "buy aluminum", "material": "aluminum", "quantity": 1000}
    await drive(agent, state, message_event(payload))

    requirement = state.get_artifact("requirement")
    assert requirement is not None
    assert requirement.kind == "requirement"
    assert requirement.created_by == "requirement_agent"
    assert requirement.correlation_id == "REQ-CONV"

    constraints = requirement.data["constraints"]
    assert constraints["material"] == "aluminum"
    assert constraints["quantity"] == 1000
    assert constraints["budget"] == 500_000.0
    assert constraints["max_unit_price"] == 2640.0  # aluminum spot 2200 * 1.2

    rfq = requirement.data["metadata"]["rfq"]
    assert rfq["material"] == "aluminum"
    assert rfq["quantity"] == 1000
    assert rfq["payment_terms"] == "net_30"


@pytest.mark.asyncio
async def test_requirement_agent_defaults_to_steel_and_spot_cap():
    agent = RequirementAgent()
    state = SwarmState()
    await drive(agent, state, message_event({}))

    constraints = state.get_artifact("requirement").data["constraints"]
    assert constraints["material"] == "steel"
    assert constraints["quantity"] == 1000
    assert constraints["budget"] == 500_000.0
    assert constraints["max_unit_price"] == 540.0  # steel spot 450 * 1.2


@pytest.mark.asyncio
async def test_requirement_agent_unknown_material_falls_back_to_steel():
    agent = RequirementAgent()
    state = SwarmState()
    await drive(agent, state, message_event({"material": "unobtainium"}))

    constraints = state.get_artifact("requirement").data["constraints"]
    assert constraints["material"] == "steel"
    assert constraints["max_unit_price"] == 540.0


@pytest.mark.asyncio
async def test_requirement_agent_respects_explicit_max_unit_price():
    agent = RequirementAgent()
    state = SwarmState()
    await drive(agent, state, message_event({"material": "copper", "max_unit_price": 11_000.0}))

    constraints = state.get_artifact("requirement").data["constraints"]
    assert constraints["material"] == "copper"
    assert constraints["max_unit_price"] == 11_000.0


@pytest.mark.asyncio
async def test_requirement_agent_ignores_other_message_intents():
    agent = RequirementAgent()
    state = SwarmState()
    other = Message(sender="user", intent="unrelated", payload={})
    await drive(agent, state, Event.from_message(other))

    assert state.get_artifact("requirement") is None


@pytest.mark.asyncio
async def test_requirement_agent_ignores_replayed_events():
    agent = RequirementAgent()
    state = SwarmState()
    event = message_event({"material": "aluminum"})
    event.replayed = True
    await drive(agent, state, event)

    assert state.get_artifact("requirement") is None


@pytest.mark.asyncio
async def test_requirement_agent_publishes_requirement_created():
    agent = RequirementAgent()
    bus = EventBus()
    agent.bus = bus
    seen: list[Event] = []

    async def record(event: Event) -> None:
        seen.append(event)

    bus.subscribe(ProcurementEventType.REQUIREMENT_CREATED, record)
    state = SwarmState()
    await drive(agent, state, message_event({"material": "aluminum"}), bus=bus)

    assert len(seen) == 1
    assert seen[0].source == "requirement_agent"
    assert seen[0].correlation_id == "REQ-CONV"
    assert seen[0].payload["artifact"] == "requirement"


@pytest.mark.asyncio
async def test_requirement_agent_requires_a_bus_to_publish():
    agent = RequirementAgent()
    state = SwarmState()

    with pytest.raises(RuntimeError, match="no event bus"):
        agent.state = state
        await agent.step(message_event({}))
