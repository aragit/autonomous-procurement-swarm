"""Unit tests for the Phase 3 CompletionTracker."""

import pytest

from swarm import CompletionTracker, Event, EventBus, SwarmState
from swarm.core.artifact import Artifact

EVALUATION_DONE = "EvaluationCompleted"
QUOTES_DONE = "QuotesCompleted"


def build() -> tuple[SwarmState, EventBus, CompletionTracker]:
    state = SwarmState(request_id="REQ-1")
    bus = EventBus()
    tracker = CompletionTracker(
        state,
        bus,
        completion_events={"evaluation": EVALUATION_DONE, "quote": QUOTES_DONE},
    )
    bus.subscribe("*", tracker.handler)
    return state, bus, tracker


async def emit_evaluations(
    state: SwarmState,
    bus: EventBus,
    count: int,
    correlation_id: str = "CONV-1",
) -> None:
    """Create ``count`` evaluation artifacts and announce each one."""
    for index in range(count):
        state.put_artifact(
            Artifact(
                kind="evaluation",
                name=f"evaluation_s{index}",
                data={},
                correlation_id=correlation_id,
                created_by="evaluation_agent",
            )
        )
        await bus.publish(
            Event(
                type="SupplierEvaluated",
                source="evaluation_agent",
                payload={"supplier_id": f"s{index}"},
                correlation_id=correlation_id,
            )
        )


@pytest.mark.asyncio
async def test_completion_tracker_does_not_fire_before_expected_count():
    state, bus, _ = build()
    state.expect_artifact(kind="evaluation", count=3, correlation_id="CONV-1")
    events: list[Event] = []
    bus.subscribe(EVALUATION_DONE, lambda event: events.append(event) or None)

    await emit_evaluations(state, bus, 2)
    assert events == []
    assert not state.is_group_completed("CONV-1", "evaluation")


@pytest.mark.asyncio
async def test_completion_tracker_fires_completion_event_once_when_count_reached():
    state, bus, _ = build()
    state.expect_artifact(kind="evaluation", count=3, correlation_id="CONV-1")
    events: list[Event] = []
    bus.subscribe(EVALUATION_DONE, lambda event: events.append(event))

    await emit_evaluations(state, bus, 3)
    # Trigger more deliveries to prove the group fires exactly once.
    await emit_evaluations(state, bus, 2)

    assert state.is_group_completed("CONV-1", "evaluation")
    assert len(events) == 1
    assert events[0].type == EVALUATION_DONE
    assert events[0].correlation_id == "CONV-1"
    assert events[0].source == "completion_tracker"
    assert events[0].payload["count"] == 3


@pytest.mark.asyncio
async def test_completion_tracker_does_not_fire_without_expectation():
    state, bus, _ = build()
    events: list[Event] = []
    bus.subscribe(EVALUATION_DONE, lambda event: events.append(event))

    await emit_evaluations(state, bus, 5)
    assert events == []


@pytest.mark.asyncio
async def test_completion_tracker_ignores_replayed_events():
    state, bus, _ = build()
    state.expect_artifact(kind="evaluation", count=1, correlation_id="CONV-1")
    state.put_artifact(
        Artifact(kind="evaluation", name="e", data={}, correlation_id="CONV-1", created_by="a")
    )
    await bus.publish(
        Event(
            type="SupplierEvaluated",
            source="evaluation_agent",
            correlation_id="CONV-1",
            replayed=True,
        )
    )
    assert not state.is_group_completed("CONV-1", "evaluation")


@pytest.mark.asyncio
async def test_completion_tracker_counts_artifacts_per_correlation_id():
    state, bus, _ = build()
    state.expect_artifact(kind="quote", count=2, correlation_id="CONV-1")
    state.expect_artifact(kind="quote", count=1, correlation_id="CONV-2")

    state.put_artifact(
        Artifact(kind="quote", name="q1", data={}, correlation_id="CONV-2", created_by="a")
    )
    await bus.publish(
        Event(type="QuoteGenerated", source="negotiation_agent", correlation_id="CONV-2")
    )
    assert state.is_group_completed("CONV-2", "quote")
    assert not state.is_group_completed("CONV-1", "quote")


@pytest.mark.asyncio
async def test_completion_tracker_marks_group_without_completion_event():
    state, bus, tracker = build()
    state.expect_artifact(kind="evaluation", count=1, correlation_id="CONV-1")
    state.put_artifact(
        Artifact(kind="evaluation", name="e", data={}, correlation_id="CONV-1", created_by="a")
    )
    await bus.publish(
        Event(type="SupplierEvaluated", source="evaluation_agent", correlation_id="CONV-1")
    )
    assert state.is_group_completed("CONV-1", "evaluation")
    assert tracker.to_dict()["published"] == [("CONV-1", "evaluation")]
