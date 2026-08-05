"""Unit tests for the swarm runtime core (Phase 1)."""

from collections.abc import Iterable

import pytest

from swarm import (
    ANY_EVENT,
    AgentRegistry,
    AgentStatus,
    Artifact,
    BaseAgent,
    Event,
    EventBus,
    Message,
    Swarm,
    SwarmEventType,
    SwarmState,
)


class EchoAgent(BaseAgent):
    """Test agent that records perceived events and echoes to shared state."""

    name = "echo"

    def __init__(
        self,
        name: str | None = None,
        *,
        description: str = "",
        capabilities: Iterable[str] | None = None,
    ) -> None:
        super().__init__(name=name, description=description, capabilities=capabilities)
        self.seen: list[Event] = []

    async def perceive(self, event: Event) -> None:
        self.seen.append(event)

    async def reason(self, state: SwarmState) -> None:
        pass

    async def act(self, state: SwarmState) -> None:
        state.results[f"last_{self.name}"] = len(self.seen)


# ─── Message ────────────────────────────────────────────────────────────────


def test_message_carries_intent_and_payload():
    msg = Message(
        sender="requirement_agent",
        receiver="supplier_discovery",
        intent="supplier_search_requested",
        payload={"quantity": 500, "item": "laptops", "budget_usd": 500_000, "delivery_days": 30},
    )
    assert msg.sender == "requirement_agent"
    assert msg.receiver == "supplier_discovery"
    assert msg.intent == "supplier_search_requested"
    assert msg.payload["quantity"] == 500
    assert msg.payload["budget_usd"] == 500_000
    assert msg.metadata == {}


def test_message_broadcast_has_no_receiver():
    msg = Message(
        sender="requirement_agent",
        intent="supplier_search_requested",
        payload={"quantity": 500},
    )
    assert msg.receiver is None
    assert msg.metadata == {}


# ─── Event ──────────────────────────────────────────────────────────────────


def test_event_carries_required_fields():
    event = Event(type="supplier_search_requested", source="requirement_agent", payload={})
    assert event.type == "supplier_search_requested"
    assert event.source == "requirement_agent"
    assert event.payload == {}
    assert event.id
    assert event.timestamp


def test_event_has_unique_id():
    e1 = Event(type="supplier_search_requested", source="requirement_agent", payload={})
    e2 = Event(type="supplier_search_requested", source="requirement_agent", payload={})
    assert e1.id != e2.id


def test_event_from_message():
    msg = Message(sender="a", receiver="b", intent="rfq", payload={"qty": 1})
    event = Event.from_message(msg)
    assert event.type == SwarmEventType.MESSAGE
    assert event.source == "a"
    assert event.message is msg
    assert event.payload == {"intent": "rfq", "receiver": "b"}


# ─── EventBus ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_event_bus_routes_by_type():
    bus = EventBus()
    received: list[Event] = []

    async def on_message(event: Event) -> None:
        received.append(event)

    bus.subscribe(SwarmEventType.MESSAGE, on_message)
    bus.subscribe("custom", on_message)

    await bus.publish(Event(type=SwarmEventType.MESSAGE, source="s"))
    await bus.publish(Event(type="custom", source="s"))
    await bus.publish(Event(type="unrelated", source="s"))

    assert len(received) == 2
    assert len(bus.event_log()) == 3


@pytest.mark.asyncio
async def test_event_bus_wildcard_subscription():
    bus = EventBus()
    seen: list[str] = []

    async def on_any(event: Event) -> None:
        seen.append(event.type)

    bus.subscribe(ANY_EVENT, on_any)
    await bus.publish(Event(type="a", source="s"))
    await bus.publish(Event(type="b", source="s"))
    assert seen == ["a", "b"]


@pytest.mark.asyncio
async def test_event_bus_isolates_handler_failures():
    bus = EventBus()
    reached = []

    async def bad(_event: Event) -> None:
        raise ValueError("boom")

    async def good(event: Event) -> None:
        reached.append(event.type)

    bus.subscribe("x", bad)
    bus.subscribe("x", good)

    errors = await bus.publish(Event(type="x", source="s"))
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    assert reached == ["x"]


@pytest.mark.asyncio
async def test_event_bus_unsubscribe():
    bus = EventBus()
    seen: list[str] = []

    async def on_x(event: Event) -> None:
        seen.append(event.type)

    bus.subscribe("x", on_x)
    await bus.publish(Event(type="x", source="s"))
    bus.unsubscribe("x", on_x)
    await bus.publish(Event(type="x", source="s"))
    assert seen == ["x"]


@pytest.mark.asyncio
async def test_event_bus_no_subscribers_returns_no_errors():
    bus = EventBus()
    assert await bus.publish(Event(type="nobody", source="s")) == []


# ─── SwarmState ─────────────────────────────────────────────────────────────


def test_swarm_state_is_mutable():
    state = SwarmState(request_id="REQ-001", goal="source 500 laptops")
    assert state.request_id == "REQ-001"
    assert state.goal == "source 500 laptops"

    state.put_artifact(Artifact(kind="requirement", name="requirement", data={"quantity": 500}))
    state.results["winner"] = "supplier_a"
    state.events.append(Event(type="supplier_search_requested", source="requirement_agent"))

    assert state.get_artifact("requirement").data["quantity"] == 500
    assert state.results["winner"] == "supplier_a"
    assert len(state.events) == 1


def test_swarm_state_serializable_roundtrip():
    state = SwarmState(request_id="REQ-001", goal="g")
    state.put_artifact(Artifact(kind="requirement", name="requirement", data={"qty": 500}))
    state.events.append(Event(type="x", source="s"))

    raw = state.to_dict()
    assert raw["request_id"] == "REQ-001"
    assert raw["goal"] == "g"
    assert raw["artifacts"][0]["name"] == "requirement"

    restored = SwarmState.from_dict(raw)
    assert restored == state
    assert isinstance(restored.artifacts[0], Artifact)
    assert isinstance(restored.events[0], Event)


def test_swarm_state_defaults():
    state = SwarmState()
    assert state.request_id == ""
    assert state.goal == ""
    assert state.artifacts == []
    assert state.events == []
    assert state.results == {}
    assert state.expectations == {}
    assert state.completions == {}


def test_expect_and_complete_artifact_default_to_request_id():
    state = SwarmState(request_id="REQ-1")
    state.expect_artifact("evaluation", count=3)
    assert state.expected_count("REQ-1", "evaluation") == 3

    state.complete_artifact("evaluation")
    assert state.is_group_completed("REQ-1", "evaluation")
    state.complete_artifact("evaluation")
    assert state.completions["REQ-1"] == ["evaluation"]


def test_expect_and_complete_artifact_require_correlation():
    state = SwarmState()
    with pytest.raises(ValueError):
        state.expect_artifact("evaluation")
    with pytest.raises(ValueError):
        state.complete_artifact("evaluation")


def test_completed_artifact_count_respects_correlation_id():
    state = SwarmState(request_id="REQ-1")
    state.put_artifact(Artifact(kind="evaluation", name="e1", data={}, correlation_id="A"))
    state.put_artifact(Artifact(kind="evaluation", name="e2", data={}, correlation_id="A"))
    state.put_artifact(Artifact(kind="evaluation", name="e3", data={}, correlation_id="B"))
    assert state.completed_artifact_count("A", "evaluation") == 2
    assert state.completed_artifact_count("B", "evaluation") == 1
    assert state.completed_artifact_count("C", "evaluation") == 0


def test_get_execution_trace_reports_events_artifacts_and_agent_actions():
    state = SwarmState(request_id="REQ-1")
    state.events.append(Event(type="msg", source="user", correlation_id="CONV-1"))
    state.events.append(
        Event(type="SupplierEvaluated", source="evaluation_agent", correlation_id="CONV-1")
    )
    state.events.append(Event(type="swarm.started", source="swarm", correlation_id="CONV-1"))
    state.put_artifact(
        Artifact(
            kind="evaluation",
            name="evaluation_s1",
            data={"score": 0.9},
            correlation_id="CONV-1",
            created_by="evaluation_agent",
        )
    )
    state.put_artifact(
        Artifact(kind="evaluation", name="evaluation_other", data={}, correlation_id="CONV-2")
    )

    trace = state.get_execution_trace("CONV-1")

    assert trace["correlation_id"] == "CONV-1"
    assert [event["type"] for event in trace["events"]] == [
        "msg",
        "SupplierEvaluated",
        "swarm.started",
    ]
    assert [artifact["name"] for artifact in trace["artifacts"]] == ["evaluation_s1"]

    actions = trace["agent_actions"]
    # Agent artifact creation is captured with lineage.
    assert any(
        action["action"] == "artifact_created"
        and action["kind"] == "evaluation"
        and action["name"] == "evaluation_s1"
        and action["agent"] == "evaluation_agent"
        for action in actions
    )
    # Agent event publications are captured.
    assert any(
        action["action"] == "event_published"
        and action["event_type"] == "SupplierEvaluated"
        and action["agent"] == "evaluation_agent"
        for action in actions
    )
    # Runtime-sourced events (swarm/coordinator) are filtered from the audit trail.
    assert not any(action.get("event_type") == "swarm.started" for action in actions)
    # Agent actions are chronological.
    assert [action["timestamp"] for action in actions] == sorted(
        action["timestamp"] for action in actions
    )


# ─── AgentRegistry ──────────────────────────────────────────────────────────


def test_registry_register_and_lookup():
    reg = AgentRegistry()
    agent = EchoAgent()
    reg.register(agent)
    assert reg.get("echo") is agent
    assert reg.require("echo") is agent
    assert len(reg) == 1
    assert "echo" in reg
    assert reg.names() == ["echo"]

    with pytest.raises(ValueError):
        reg.register(EchoAgent())

    assert reg.unregister("echo") is agent
    assert reg.get("echo") is None
    with pytest.raises(KeyError):
        reg.require("echo")


def test_registry_by_capability():
    reg = AgentRegistry()
    reg.register(EchoAgent("a", capabilities=["negotiation", "pricing"]))
    reg.register(EchoAgent("b", capabilities=["pricing"]))
    reg.register(EchoAgent("c", capabilities=["scoring"]))
    assert [a.name for a in reg.by_capability("pricing")] == ["a", "b"]
    assert [a.name for a in reg.by_capability("negotiation")] == ["a"]
    assert reg.by_capability("logistics") == []


def test_base_agent_is_abstract():
    with pytest.raises(TypeError):
        BaseAgent()  # type: ignore[abstract]


# ─── BaseAgent lifecycle ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_base_agent_lifecycle():
    agent = EchoAgent()
    assert agent.status == AgentStatus.CREATED
    assert agent.capabilities == []
    await agent.initialize()
    assert agent.status == AgentStatus.INITIALIZED
    await agent.shutdown()
    assert agent.status == AgentStatus.TERMINATED
    with pytest.raises(RuntimeError):
        await agent.initialize()


@pytest.mark.asyncio
async def test_base_agent_send_message():
    bus = EventBus()
    agent = EchoAgent()
    seen: list[Event] = []

    async def on_message(event: Event) -> None:
        seen.append(event)

    bus.subscribe(SwarmEventType.MESSAGE, on_message)
    await agent.send(bus, "supplier_search_requested", {"quantity": 500})
    assert len(seen) == 1
    assert seen[0].message is not None
    assert seen[0].message.sender == "echo"
    assert seen[0].message.intent == "supplier_search_requested"
    assert seen[0].message.payload == {"quantity": 500}


@pytest.mark.asyncio
async def test_agent_step_runs_perceive_reason_act():
    agent = EchoAgent()
    state = SwarmState()
    agent.state = state
    await agent.step(Event(type="x", source="s"))
    assert len(agent.seen) == 1
    assert state.results["last_echo"] == 1


# ─── Swarm orchestration ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_swarm_orchestrates_agent_lifecycle():
    swarm = Swarm(request_id="REQ-001", goal="source 500 laptops")
    agent = EchoAgent()
    swarm.register(agent)
    assert agent.status == AgentStatus.CREATED
    assert swarm.state.request_id == "REQ-001"
    assert swarm.state.goal == "source 500 laptops"

    await swarm.start()
    assert agent.status == AgentStatus.INITIALIZED
    assert swarm.started

    errors = await swarm.send_message("supplier_search_requested", {"quantity": 500})
    assert errors == []
    # initialized + swarm.started + message
    assert [e.type for e in agent.seen] == [
        SwarmEventType.AGENT_INITIALIZED,
        SwarmEventType.SWARM_STARTED,
        SwarmEventType.MESSAGE,
    ]

    await swarm.shutdown()
    assert agent.status == AgentStatus.TERMINATED
    assert not swarm.started


@pytest.mark.asyncio
async def test_swarm_unregister_stops_delivery():
    swarm = Swarm()
    agent = EchoAgent()
    swarm.register(agent)
    await swarm.send_message("supplier_search_requested", {})
    assert len(agent.seen) == 1

    assert swarm.unregister("echo") is agent
    await swarm.send_message("supplier_search_requested", {})
    assert len(agent.seen) == 1


@pytest.mark.asyncio
async def test_swarm_dispatch_injects_external_event():
    swarm = Swarm()
    agent = EchoAgent()
    swarm.register(agent)
    errors = await swarm.dispatch(Event(type="external", source="env", payload={"v": 1}))
    assert errors == []
    assert [e.type for e in agent.seen] == ["external"]
