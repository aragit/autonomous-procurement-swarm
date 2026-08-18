"""Central coordinator for the swarm runtime.

.. deprecated::
    Use :class:`mesh.cluster.ProcurementCluster` (V2 mesh runtime) instead.
    The SwarmCoordinator is preserved here for backward compatibility with
    existing tests and downstream consumers.
"""

import warnings

warnings.warn(
    "swarm.orchestration.coordinator is deprecated. Use "
    "mesh.cluster.ProcurementCluster (V2 mesh runtime) instead. For the "
    "legacy implementation, import from legacy.coordinator directly.",
    DeprecationWarning,
    stacklevel=2,
)

from collections.abc import Callable, Iterable  # noqa: E402
from typing import Any  # noqa: E402
from uuid import uuid4  # noqa: E402

import structlog  # noqa: E402

from swarm.core.agent import BaseAgent, drive_on_event, route_on_event  # noqa: E402
from swarm.core.event import ANY_EVENT, Event, EventBus, EventHandler, SwarmEventType  # noqa: E402
from swarm.core.message import Message  # noqa: E402
from swarm.core.registry import AgentRegistry  # noqa: E402
from swarm.core.state import SwarmState  # noqa: E402

logger = structlog.get_logger(__name__)


class SwarmCoordinator:
    """Internal runtime engine behind the public ``Swarm`` facade.

    Registers agents, routes events and maintains the shared swarm state. This
    is the implementation detail of the runtime: applications should drive the
    swarm through ``Swarm`` and must not depend on this class directly. It
    performs no planning and spawns no agents — it only moves events and state.

    The coordinator records a canonical event history in ``state.events``.
    Replays are strictly read-only with respect to that history: events
    delivered by :meth:`replay` are flagged ``replayed=True`` and skipped by
    the recorder, so debugging via replay can never corrupt shared state.

    1. A request arrives via :meth:`receive_event` or :meth:`route_message`.
    2. The event is recorded in the shared state and fanned out to every agent
       subscribed to its type (by default, all agents).
    3. Agents mutate shared state and publish follow-up events, which are again
       recorded and routed to whoever is interested.
    """

    def __init__(self, *, request_id: str = "", goal: str = "") -> None:
        self.bus = EventBus()
        self.registry = AgentRegistry()
        self.state = SwarmState(request_id=request_id, goal=goal)
        self._subscriptions: dict[str, set[str]] = {}
        self._handlers: dict[str, EventHandler] = {}
        self.bus.subscribe(ANY_EVENT, self._record_event)

    async def _record_event(self, event: Event) -> None:
        """Keep a canonical event history in shared state.

        Replayed events are ignored: replay is a debugging aid and must not
        mutate canonical state.
        """
        if event.replayed:
            return
        self.state.events.append(event)

    def register_agent(
        self,
        agent: BaseAgent,
        event_types: str | Iterable[str] | None = None,
        *,
        route: Callable[[Event], BaseAgent | None] | None = None,
    ) -> BaseAgent:
        """Register ``agent`` and subscribe it to ``event_types``.

        By default the agent perceives every event on the bus. Pass a type or
        an iterable of types to restrict perception to specific events. The
        agent is subscribed with an auto-stepping handler, so a delivered
        event runs its whole perceive → reason → act cycle, and the shared bus
        and state are assigned to ``agent.bus`` / ``agent.state`` so it can
        publish domain events and read artifacts.

        Pass ``route`` to subscribe a routing handler instead: ``route`` picks
        the single agent to run for each event, enabling capability/tag-based
        specialization.
        """
        self.registry.register(agent)
        agent.bus = self.bus
        agent.state = self.state
        if isinstance(event_types, str):
            types = {event_types}
        else:
            types = set(event_types) if event_types is not None else {ANY_EVENT}
        self._subscriptions[agent.name] = types
        if route is not None:
            handler = route_on_event(route, self.state)
        else:
            handler = drive_on_event(agent, self.state)
        self._handlers[agent.name] = handler
        for event_type in types:
            self.bus.subscribe(event_type, handler)
        logger.info(
            "agent_registered",
            agent=agent.name,
            request_id=self.state.request_id,
            event_types=sorted(types),
        )
        return agent

    def unregister_agent(self, name: str) -> BaseAgent | None:
        """Remove ``name`` from the swarm and stop delivering events to it."""
        agent = self.registry.unregister(name)
        if agent is not None:
            handler = self._handlers.pop(name, None)
            for event_type in self._subscriptions.pop(name, set()):
                if handler is not None:
                    self.bus.unsubscribe(event_type, handler)
        return agent

    @property
    def agents(self) -> dict[str, BaseAgent]:
        """Every registered agent, mapped by name."""
        return self.registry.list_agents()

    async def start(self) -> None:
        """Initialize every registered agent and announce the swarm is live."""
        for agent in self.registry.all():
            await agent.initialize()
        await self.receive_event(Event(type=SwarmEventType.SWARM_STARTED, source="coordinator"))
        logger.info("swarm_started", request_id=self.state.request_id)

    async def shutdown(self) -> None:
        """Terminate every registered agent and announce the swarm stopped."""
        for agent in self.registry.all():
            await agent.shutdown()
        await self.receive_event(Event(type=SwarmEventType.SWARM_STOPPED, source="coordinator"))
        logger.info("swarm_stopped", request_id=self.state.request_id)

    async def receive_event(self, event: Event) -> list[Exception]:
        """Record ``event`` in shared state and route it to interested agents.

        Returns any exceptions raised by the receiving agents; one failing
        agent never blocks delivery to the others.
        """
        logger.debug(
            "event_received",
            event_type=event.type,
            source=event.source,
            correlation_id=event.correlation_id,
            request_id=self.state.request_id,
        )
        return await self.bus.publish(event)

    async def replay(self, event_type: str | None = None) -> None:
        """Re-deliver the event history without mutating canonical state.

        Delivered events are flagged as replayed, so the recorder ignores them
        and ``state.events`` stays untouched. Agents still perceive the events
        — that is what makes replay useful for rebuilding derived state after a
        (re)join or during debugging.
        """
        await self.bus.replay(event_type=event_type, mode="read_only")

    async def route_message(
        self,
        intent: str,
        payload: dict[str, Any],
        *,
        sender: str = "coordinator",
        receiver: str | None = None,
        correlation_id: str | None = None,
    ) -> list[Exception]:
        """Send a message event through the swarm on behalf of ``sender``.

        With no ``receiver`` the message is broadcast on its ``intent``, so
        callers never need to know which agent will handle it. When no
        ``correlation_id`` is supplied one is generated, so every external
        request seeds a traceable conversation chain.
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
        logger.info(
            "message_routed",
            intent=intent,
            sender=sender,
            receiver=receiver,
            correlation_id=correlation_id,
            request_id=self.state.request_id,
        )
        return await self.receive_event(Event.from_message(message, source=sender))
