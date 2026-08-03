"""Swarm runtime core — reusable primitives for autonomous multi-agent swarms.

Provides the foundation for future agents to communicate without direct
coupling:

- :class:`BaseAgent` / :class:`AgentStatus` — agent abstraction + lifecycle
- :class:`Message` — agent-to-agent message (sender / receiver / intent /
  payload / metadata / correlation_id)
- :class:`Event`, :class:`EventBus` — event-driven communication with replay
- :class:`Artifact` — typed, versioned working data shared between agents
- :class:`SwarmState` — shared, serializable swarm context over artifacts
- :class:`Capability` — declarative agent capability schema
- :class:`AgentRegistry` — agent registration and capability lookup
- :class:`Swarm` — minimal orchestration layer tying the primitives together
"""

from collections.abc import Callable, Iterable
from typing import Any
from uuid import uuid4

import structlog

from swarm.core.agent import AgentStatus, BaseAgent, drive_on_event, route_on_event
from swarm.core.artifact import Artifact
from swarm.core.capability import Capability, to_capability
from swarm.core.event import ANY_EVENT, Event, EventBus, EventHandler, SwarmEventType
from swarm.core.message import Message
from swarm.core.registry import AgentRegistry
from swarm.core.state import SwarmState

logger = structlog.get_logger(__name__)


class Swarm:
    """Public orchestration interface for the swarm runtime.

    Role contract:
    - ``Swarm`` is the public entry point that applications embed. It contains
      **no business logic** — only lifecycle, routing and state access.
    - :class:`SwarmCoordinator` is the internal runtime engine. External code
      drives the swarm through ``Swarm`` and must not call the coordinator
      directly.
    - Start execution via :meth:`start`, feed work via :meth:`send_message` /
      :meth:`dispatch`, inspect/replay history via the event bus, and stop via
      :meth:`shutdown`.

    Composes a shared :class:`EventBus`, a :class:`SwarmState` and an
    :class:`AgentRegistry` and drives the agent lifecycle. Agents exchange
    messages and events exclusively through the bus, so no agent holds a
    direct reference to any other.
    """

    def __init__(self, *, request_id: str = "", goal: str = "") -> None:
        self.bus = EventBus()
        self.registry = AgentRegistry()
        self.state = SwarmState(request_id=request_id, goal=goal)
        self._subscriptions: dict[str, list[str]] = {}
        self._handlers: dict[str, EventHandler] = {}
        self._started = False
        self.bus.subscribe(ANY_EVENT, self._record_event)

    async def _record_event(self, event: Event) -> None:
        """Keep a canonical event history in shared state.

        Replayed events are ignored: replay is a debugging aid and must not
        mutate canonical state.
        """
        if event.replayed:
            return
        self.state.events.append(event)

    def register(
        self,
        agent: BaseAgent,
        event_types: str | Iterable[str] | None = None,
        *,
        route: Callable[[Event], BaseAgent | None] | None = None,
    ) -> BaseAgent:
        """Register ``agent`` and subscribe it to the event bus.

        By default the agent perceives every event. Pass ``event_types`` to
        restrict perception to specific event types. The agent is subscribed
        with an auto-stepping handler, so a delivered event runs its whole
        perceive → reason → act cycle, and the shared bus and state are
        assigned to ``agent.bus`` / ``agent.state`` so it can publish domain
        events and read artifacts.

        Pass ``route`` to subscribe a routing handler instead: ``route``
        inspects each event and returns the single agent that should run (e.g.
        ``registry.best_for_capability(...)``), enabling capability/tag-based
        specialization without every specialized agent reacting to every event.
        """
        self.registry.register(agent)
        agent.bus = self.bus
        agent.state = self.state
        if isinstance(event_types, str):
            types = [event_types]
        else:
            types = list(event_types) if event_types is not None else [ANY_EVENT]
        self._subscriptions[agent.name] = types
        if route is not None:
            handler = route_on_event(route, self.state)
        else:
            handler = drive_on_event(agent, self.state)
        self._handlers[agent.name] = handler
        for event_type in types:
            self.bus.subscribe(event_type, handler)
        return agent

    def unregister(self, name: str) -> BaseAgent | None:
        """Unregister ``name`` and stop delivering events to it."""
        agent = self.registry.unregister(name)
        if agent is not None:
            handler = self._handlers.pop(name, None)
            for event_type in self._subscriptions.pop(name, []):
                if handler is not None:
                    self.bus.unsubscribe(event_type, handler)
        return agent

    async def start(self) -> None:
        """Initialize every registered agent and announce the swarm is live."""
        for agent in self.registry.all():
            await agent.initialize()
            await self.bus.publish(Event(type=SwarmEventType.AGENT_INITIALIZED, source=agent.name))
        self._started = True
        await self.bus.publish(Event(type=SwarmEventType.SWARM_STARTED, source="swarm"))
        logger.info("swarm_started", request_id=self.state.request_id)

    async def dispatch(self, event: Event) -> list[Exception]:
        """Inject an external event into the swarm."""
        return await self.bus.publish(event)

    async def send_message(
        self,
        intent: str,
        payload: dict[str, Any],
        *,
        sender: str = "swarm",
        receiver: str | None = None,
        correlation_id: str | None = None,
    ) -> list[Exception]:
        """Publish a :class:`Message` to the swarm on behalf of ``sender``.

        When no ``correlation_id`` is supplied one is generated, so every
        external request seeds a traceable conversation chain.
        """
        if correlation_id is None:
            correlation_id = uuid4().hex
        message = Message(
            sender=sender,
            receiver=receiver,
            intent=intent,
            payload=payload,
            correlation_id=correlation_id,
        )
        return await self.bus.publish(Event.from_message(message, source=sender))

    async def replay(self, event_type: str | None = None) -> None:
        """Re-deliver the audit log to current subscribers, read-only.

        Replay is a debugging aid: nothing is re-recorded and events are
        flagged as replayed so state recorders skip them.
        """
        await self.bus.replay(event_type=event_type, mode="read_only")

    async def shutdown(self) -> None:
        """Terminate every registered agent and announce the swarm stopped."""
        for agent in self.registry.all():
            await agent.shutdown()
            await self.bus.publish(Event(type=SwarmEventType.AGENT_TERMINATED, source=agent.name))
        await self.bus.publish(Event(type=SwarmEventType.SWARM_STOPPED, source="swarm"))
        self._started = False
        logger.info("swarm_stopped", request_id=self.state.request_id)

    @property
    def started(self) -> bool:
        return self._started

    def get_execution_trace(self, correlation_id: str) -> dict[str, Any]:
        """Ordered events, artifacts and agent actions for one conversation.

        The trace is read-only: it derives its content from the canonical
        event log and artifact history and never mutates shared state.
        """
        return self.state.get_execution_trace(correlation_id)

    def expect_artifact(
        self,
        kind: str,
        *,
        count: int = 1,
        correlation_id: str | None = None,
    ) -> None:
        """Declare that ``count`` artifacts of ``kind`` are expected for a request."""
        self.state.expect_artifact(kind, count=count, correlation_id=correlation_id)

    def complete_artifact(self, kind: str, *, correlation_id: str | None = None) -> None:
        """Record that the ``kind`` group is complete for a request (idempotent)."""
        self.state.complete_artifact(kind, correlation_id=correlation_id)


__all__ = [
    "ANY_EVENT",
    "AgentRegistry",
    "AgentStatus",
    "Artifact",
    "BaseAgent",
    "Capability",
    "Event",
    "EventBus",
    "Message",
    "Swarm",
    "SwarmEventType",
    "SwarmState",
    "to_capability",
]
