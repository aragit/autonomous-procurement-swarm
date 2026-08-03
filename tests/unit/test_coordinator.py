"""Unit tests for the swarm coordinator (Phase 1)."""

import pytest

from swarm import AgentStatus, BaseAgent, Event, SwarmState
from swarm.orchestration.coordinator import SwarmCoordinator


class ReactAgent(BaseAgent):
    """Test agent that records perceived events and acts on shared state."""

    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self.perceived: list[Event] = []
        self.acted = False

    async def perceive(self, event: Event) -> None:
        self.perceived.append(event)

    async def reason(self, state: SwarmState) -> None:
        pass

    async def act(self, state: SwarmState) -> None:
        self.acted = True
        state.results[f"{self.name}_acted"] = True


@pytest.mark.asyncio
async def test_coordinator_registers_agents_and_lifecycle():
    coordinator = SwarmCoordinator(request_id="REQ-1", goal="g")
    agent = ReactAgent("a")
    coordinator.register_agent(agent)
    assert coordinator.agents == {"a": agent}
    assert agent.status == AgentStatus.CREATED

    await coordinator.start()
    assert agent.status == AgentStatus.INITIALIZED

    await coordinator.shutdown()
    assert agent.status == AgentStatus.TERMINATED


@pytest.mark.asyncio
async def test_coordinator_routes_event_and_keeps_state():
    coordinator = SwarmCoordinator()
    agent = ReactAgent("a")
    coordinator.register_agent(agent)

    errors = await coordinator.receive_event(
        Event(type="RequirementCreated", source="requirement_agent", payload={"item": "laptops"})
    )
    assert errors == []
    assert [e.type for e in agent.perceived] == ["RequirementCreated"]
    assert [e.type for e in coordinator.state.events] == ["RequirementCreated"]
    assert coordinator.state.events[0].payload == {"item": "laptops"}


@pytest.mark.asyncio
async def test_coordinator_routes_message_intent():
    coordinator = SwarmCoordinator()
    agent = ReactAgent("a")
    coordinator.register_agent(agent)

    await coordinator.route_message(
        "supplier_search_requested",
        {"qty": 500},
        sender="requirement_agent",
        receiver="supplier_discovery",
        correlation_id="CONV-7",
    )

    message_event = coordinator.state.events[0]
    assert message_event.message is not None
    assert message_event.message.sender == "requirement_agent"
    assert message_event.message.receiver == "supplier_discovery"
    assert message_event.message.intent == "supplier_search_requested"
    assert message_event.message.payload == {"qty": 500}
    assert message_event.message.correlation_id == "CONV-7"
    assert message_event.correlation_id == "CONV-7"
    assert agent.perceived[-1].message is not None


@pytest.mark.asyncio
async def test_coordinator_seeds_correlation_id_for_unknown_conversation():
    coordinator = SwarmCoordinator()
    agent = ReactAgent("a")
    coordinator.register_agent(agent)

    await coordinator.route_message("requirement.requested", {"text": "find laptops"})

    message_event = coordinator.state.events[0]
    assert message_event.correlation_id is not None
    assert len(message_event.correlation_id) == 32
    assert message_event.message is not None
    assert message_event.message.correlation_id == message_event.correlation_id


@pytest.mark.asyncio
async def test_coordinator_unregister_stops_delivery():
    coordinator = SwarmCoordinator()
    agent = ReactAgent("a")
    coordinator.register_agent(agent)
    await coordinator.receive_event(Event(type="x", source="s"))
    assert len(agent.perceived) == 1

    coordinator.unregister_agent("a")
    assert coordinator.agents == {}
    await coordinator.receive_event(Event(type="x", source="s"))
    assert len(agent.perceived) == 1


@pytest.mark.asyncio
async def test_coordinator_restricts_event_types():
    coordinator = SwarmCoordinator()
    agent = ReactAgent("a")
    coordinator.register_agent(agent, event_types=["supplier_search_requested"])

    await coordinator.receive_event(Event(type="unrelated", source="s"))
    await coordinator.receive_event(Event(type="supplier_search_requested", source="s"))

    assert [e.type for e in agent.perceived] == ["supplier_search_requested"]
    assert [e.type for e in coordinator.state.events] == ["unrelated", "supplier_search_requested"]


@pytest.mark.asyncio
async def test_coordinator_state_is_updated_by_agent_act():
    coordinator = SwarmCoordinator()
    agent = ReactAgent("a")
    coordinator.register_agent(agent)

    await coordinator.receive_event(Event(type="x", source="s"))
    await agent.step(Event(type="x", source="s"))

    assert agent.acted
    assert coordinator.state.results["a_acted"] is True


@pytest.mark.asyncio
async def test_coordinator_replay_does_not_mutate_canonical_state():
    coordinator = SwarmCoordinator()
    agent = ReactAgent("a")
    coordinator.register_agent(agent)
    await coordinator.receive_event(Event(type="x", source="s"))
    assert [e.type for e in coordinator.state.events] == ["x"]
    assert len(agent.perceived) == 1

    await coordinator.replay()

    assert [e.type for e in coordinator.state.events] == ["x"]
    assert len(agent.perceived) == 2
