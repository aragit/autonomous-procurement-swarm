"""Unit tests for BaseAgent creation and lifecycle (Phase 1)."""

from collections.abc import Iterable

import pytest

from swarm import AgentStatus, BaseAgent, Capability, Event, SwarmState
from swarm.core.event import EventBus, SwarmEventType


class TrackingAgent(BaseAgent):
    """Test agent that records its lifecycle calls."""

    def __init__(
        self,
        name: str = "tracker",
        *,
        description: str = "tracking test agent",
        capabilities: Iterable[str] | None = None,
    ) -> None:
        super().__init__(name=name, description=description, capabilities=capabilities)
        self.perceived: list[Event] = []
        self.reason_calls = 0
        self.act_calls = 0

    async def perceive(self, event: Event) -> None:
        self.perceived.append(event)

    async def reason(self, state: SwarmState) -> None:
        self.reason_calls += 1

    async def act(self, state: SwarmState) -> None:
        self.act_calls += 1
        state.results[f"acted_{self.name}"] = True


def test_agent_creation_uses_defaults():
    agent = TrackingAgent()
    assert agent.name == "tracker"
    assert agent.description == "tracking test agent"
    assert agent.capabilities == []
    assert agent.status == AgentStatus.CREATED


def test_agent_creation_accepts_overrides():
    agent = TrackingAgent(
        "requirement_agent",
        description="parses requests",
        capabilities=["parsing"],
    )
    assert agent.name == "requirement_agent"
    assert agent.description == "parses requests"
    assert agent.capability_names == ["parsing"]


def test_agent_capabilities_are_schema_objects():
    agent = TrackingAgent(
        "discovery",
        capabilities=[Capability(name="supplier_discovery", description="finds suppliers")],
    )
    assert agent.capabilities == [
        Capability(name="supplier_discovery", description="finds suppliers")
    ]
    assert agent.capability_names == ["supplier_discovery"]
    assert agent.has_capability("supplier_discovery")
    assert not agent.has_capability("negotiation")


def test_agent_capabilities_accept_string_shorthand():
    agent = TrackingAgent("a", capabilities=["pricing", "logistics"])
    assert [capability.name for capability in agent.capabilities] == ["pricing", "logistics"]
    assert agent.has_capability("pricing")


@pytest.mark.asyncio
async def test_agent_lifecycle_created_initialized_terminated():
    agent = TrackingAgent()
    assert agent.status == AgentStatus.CREATED
    await agent.initialize()
    assert agent.status == AgentStatus.INITIALIZED
    await agent.shutdown()
    assert agent.status == AgentStatus.TERMINATED
    with pytest.raises(RuntimeError):
        await agent.initialize()


@pytest.mark.asyncio
async def test_agent_step_with_event_runs_full_cycle():
    agent = TrackingAgent()
    state = SwarmState()
    agent.state = state
    event = Event(type="some.event", source="user")
    await agent.step(event)
    assert agent.perceived == [event]
    assert agent.reason_calls == 1
    assert agent.act_calls == 1
    assert state.results["acted_tracker"] is True


@pytest.mark.asyncio
async def test_agent_step_requires_shared_state():
    agent = TrackingAgent()
    with pytest.raises(RuntimeError):
        await agent.step(Event(type="some.event", source="user"))


@pytest.mark.asyncio
async def test_agent_step_replayed_event_skips_reason_and_act():
    agent = TrackingAgent()
    state = SwarmState()
    agent.state = state
    event = Event(type="some.event", source="user", replayed=True)
    await agent.step(event)
    assert agent.perceived == [event]
    assert agent.reason_calls == 0
    assert agent.act_calls == 0


@pytest.mark.asyncio
async def test_agent_send_publishes_message_event():
    bus = EventBus()
    agent = TrackingAgent()
    await agent.send(
        bus,
        "supplier_search_requested",
        {"qty": 500},
        receiver="supplier_discovery",
        correlation_id="CONV-1",
    )
    log = bus.event_log()
    assert len(log) == 1
    event = log[0]
    assert event.type == SwarmEventType.MESSAGE
    assert event.message is not None
    assert event.message.sender == "tracker"
    assert event.message.intent == "supplier_search_requested"
    assert event.message.payload == {"qty": 500}
    assert event.message.receiver == "supplier_discovery"
    assert event.message.correlation_id == "CONV-1"
    assert event.correlation_id == "CONV-1"
