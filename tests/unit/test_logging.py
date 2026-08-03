"""Tests for structured logging in the swarm runtime (Phase 1.5)."""

import asyncio

import pytest
from structlog.testing import capture_logs

from swarm import AgentRegistry, BaseAgent, Event, EventBus, SwarmState
from swarm.core.logging import resolve_log_level
from swarm.orchestration.coordinator import SwarmCoordinator


class QuietAgent(BaseAgent):
    """Minimal agent that performs no work."""

    def __init__(self, name: str) -> None:
        super().__init__(name=name)

    async def perceive(self, event: Event) -> None:
        pass

    async def reason(self, state: SwarmState) -> None:
        pass

    async def act(self, state: SwarmState) -> None:
        pass


def test_event_bus_publish_is_logged_structured():
    bus = EventBus()
    with capture_logs() as captured:
        asyncio.run(bus.publish(Event(type="some.event", source="user", correlation_id="C-1")))
    entry = captured[0]
    assert entry["event"] == "event_published"
    assert entry["event_type"] == "some.event"
    assert entry["source"] == "user"
    assert entry["correlation_id"] == "C-1"
    assert entry["log_level"] == "debug"


def test_registry_registration_is_logged():
    registry = AgentRegistry()
    with capture_logs() as captured:
        registry.register(QuietAgent("q"))
    assert [entry.get("event") for entry in captured] == ["agent_registered"]
    assert captured[0]["agent"] == "q"


def test_coordinator_message_routing_is_logged():
    coordinator = SwarmCoordinator(request_id="REQ-9")
    with capture_logs() as captured:
        asyncio.run(coordinator.route_message("x", {"k": "v"}, correlation_id="C-1"))
    routed = next(entry for entry in captured if entry.get("event") == "message_routed")
    assert routed["intent"] == "x"
    assert routed["correlation_id"] == "C-1"
    assert routed["request_id"] == "REQ-9"


def test_coordinator_lifecycle_is_logged():
    coordinator = SwarmCoordinator()
    coordinator.register_agent(QuietAgent("q"))
    with capture_logs() as captured:
        asyncio.run(coordinator.start())
        asyncio.run(coordinator.shutdown())
    events = [entry.get("event") for entry in captured]
    assert "swarm_started" in events
    assert "swarm_stopped" in events


def test_log_level_defaults_to_info(monkeypatch):
    monkeypatch.delenv("SWARM_LOG_LEVEL", raising=False)
    assert resolve_log_level() == "INFO"


def test_log_level_reads_env_and_normalizes(monkeypatch):
    monkeypatch.setenv("SWARM_LOG_LEVEL", "debug")
    assert resolve_log_level() == "DEBUG"


def test_log_level_explicit_value_overrides_env(monkeypatch):
    monkeypatch.setenv("SWARM_LOG_LEVEL", "DEBUG")
    assert resolve_log_level("warning") == "WARNING"


def test_log_level_rejects_unsupported_value():
    with pytest.raises(ValueError):
        resolve_log_level("verbose")
