"""Unit tests for the Event model and its serialization (Phase 1)."""

import pytest

from swarm import ANY_EVENT, Event, EventBus, Message, SwarmEventType


def test_event_carries_required_fields():
    event = Event(
        type="supplier_search_requested",
        source="requirement_agent",
        payload={"quantity": 500, "budget_usd": 500_000},
    )
    assert event.type == "supplier_search_requested"
    assert event.source == "requirement_agent"
    assert event.payload["quantity"] == 500
    assert event.id
    assert event.timestamp
    assert event.correlation_id is None
    assert event.message is None


def test_event_ids_are_unique():
    e1 = Event(type="RequirementCreated", source="requirement_agent")
    e2 = Event(type="RequirementCreated", source="requirement_agent")
    assert e1.id != e2.id


def test_event_serialization_roundtrip():
    event = Event(
        type="RequirementCreated",
        source="requirement_agent",
        payload={"item": "laptops", "audience": "engineering team"},
        correlation_id="CORR-1",
    )
    raw = event.model_dump()
    restored = Event.model_validate(raw)
    assert restored == event
    assert restored.id == event.id
    assert restored.timestamp == event.timestamp
    assert restored.correlation_id == "CORR-1"


def test_event_from_message_wraps_message():
    msg = Message(sender="a", receiver="b", intent="rfq", payload={"qty": 1})
    event = Event.from_message(msg, source="requirement_agent")
    assert event.type == SwarmEventType.MESSAGE
    assert event.source == "requirement_agent"
    assert event.message is msg
    assert event.payload == {"intent": "rfq", "receiver": "b"}


def test_event_from_message_propagates_correlation_id():
    msg = Message(
        sender="a",
        intent="rfq",
        payload={"qty": 1},
        correlation_id="CONV-42",
    )
    event = Event.from_message(msg)
    assert event.correlation_id == "CONV-42"
    assert event.message.correlation_id == "CONV-42"


def test_event_without_correlation_id_defaults_to_none():
    event = Event(type="RequirementCreated", source="requirement_agent")
    assert event.correlation_id is None


def test_event_with_message_serialization_roundtrip():
    msg = Message(
        sender="requirement_agent",
        receiver="supplier_discovery",
        intent="supplier_search_requested",
        payload={"qty": 500},
    )
    event = Event.from_message(msg, source="requirement_agent")
    raw = event.model_dump()
    restored = Event.model_validate(raw)
    assert restored.message == msg
    assert restored.message.sender == "requirement_agent"
    assert restored.message.intent == "supplier_search_requested"


# ─── EventBus replay ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_event_bus_replay_delivers_history_to_new_subscribers():
    bus = EventBus()
    await bus.publish(Event(type="a", source="s", payload={"n": 1}))
    await bus.publish(Event(type="b", source="s", payload={"n": 2}))

    received: list[Event] = []

    async def on_any(event: Event) -> None:
        received.append(event)

    bus.subscribe(ANY_EVENT, on_any)
    await bus.replay()

    assert [event.payload for event in received] == [{"n": 1}, {"n": 2}]
    assert len(bus.event_log()) == 2


@pytest.mark.asyncio
async def test_event_bus_replay_filters_by_type():
    bus = EventBus()
    await bus.publish(Event(type="a", source="s"))
    await bus.publish(Event(type="b", source="s"))

    received: list[str] = []

    async def on_any(event: Event) -> None:
        received.append(event.type)

    bus.subscribe(ANY_EVENT, on_any)
    await bus.replay(event_type="a")

    assert received == ["a"]


@pytest.mark.asyncio
async def test_event_bus_replay_does_not_redeliver_to_unsubscribed_agents():
    bus = EventBus()
    received: list[str] = []

    async def on_any(event: Event) -> None:
        received.append(event.type)

    bus.subscribe(ANY_EVENT, on_any)
    await bus.publish(Event(type="a", source="s"))
    bus.unsubscribe(ANY_EVENT, on_any)
    await bus.replay()

    assert received == ["a"]


@pytest.mark.asyncio
async def test_event_bus_replay_read_only_marks_events():
    bus = EventBus()
    await bus.publish(Event(type="a", source="s"))

    received: list[Event] = []

    async def on_any(event: Event) -> None:
        received.append(event)

    bus.subscribe(ANY_EVENT, on_any)
    await bus.replay()
    await bus.replay(mode="deliver")

    assert len(received) == 2
    assert received[0].replayed is True
    assert received[1].replayed is False
