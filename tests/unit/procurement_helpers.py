"""Shared helpers for Phase 2 domain-agent unit tests."""

from swarm.core.agent import BaseAgent
from swarm.core.event import Event, EventBus
from swarm.core.state import SwarmState


async def drive(
    agent: BaseAgent,
    state: SwarmState,
    event: Event,
    *,
    bus: EventBus | None = None,
) -> EventBus:
    """Step ``agent`` through perceive → reason → act with a bus attached.

    Mirrors the orchestration contract (agents publish domain events through
    their assigned bus and read shared state through their assigned state): a
    fresh ``EventBus`` and the supplied ``state`` are attached when the agent
    has none, so unit tests can drive agents directly while keeping the
    runtime's no-bus-is-an-error behavior for deliberate exception tests.
    """
    if agent.bus is None:
        agent.bus = bus or EventBus()
    if agent.state is None:
        agent.state = state
    await agent.step(event)
    assert agent.bus is not None
    return agent.bus
