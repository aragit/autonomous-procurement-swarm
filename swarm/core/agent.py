"""Abstract agent interface and lifecycle for the swarm runtime."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from enum import StrEnum
from typing import Any

import structlog

from swarm.core.capability import Capability, to_capability
from swarm.core.event import Event, EventBus, EventHandler
from swarm.core.message import Message
from swarm.core.state import SwarmState

logger = structlog.get_logger(__name__)


class AgentStatus(StrEnum):
    """Lifecycle status of a swarm agent."""

    CREATED = "created"
    INITIALIZED = "initialized"
    TERMINATED = "terminated"
    FAILED = "failed"


class BaseAgent(ABC):
    """Lightweight abstract agent interface.

    Every agent has a unique ``name``, a ``description``, an optional set of
    ``capabilities``, an optional ``memory`` and optional specialization
    ``tags`` (e.g. ``{"region": "EU"}``). Capabilities are declared as
    :class:`Capability` objects (plain strings are accepted as shorthand) so
    the coordinator can discover and route to agents by what they can do. The
    perceive-reason-act lifecycle is driven by the orchestration layer via
    :meth:`step`; agents communicate exclusively through the shared event bus,
    never directly.
    """

    name: str
    description: str
    capabilities: list[Capability]
    memory: Any | None
    status: AgentStatus
    tags: dict[str, str] = {}
    bus: EventBus | None = None
    state: SwarmState | None = None

    def __init__(
        self,
        name: str | None = None,
        *,
        description: str | None = None,
        capabilities: Iterable[str | Capability] | None = None,
        memory: Any | None = None,
    ) -> None:
        declared_name: str | None = getattr(type(self), "name", None)
        if name is None:
            name = declared_name
        if not name:
            raise ValueError("BaseAgent requires a non-empty name")
        self.name = name
        self.description = (
            description if description is not None else getattr(type(self), "description", "")
        )
        declared_capabilities = getattr(type(self), "capabilities", None)
        raw_capabilities: Iterable[str | Capability]
        if capabilities is not None:
            raw_capabilities = capabilities
        elif declared_capabilities is not None:
            raw_capabilities = declared_capabilities
        else:
            raw_capabilities = ()
        self.capabilities = [to_capability(spec) for spec in raw_capabilities]
        declared_memory = getattr(type(self), "memory", None)
        self.memory = memory if memory is not None else declared_memory
        self.tags = dict(getattr(type(self), "tags", None) or {})
        self.status = AgentStatus.CREATED

    @property
    def capability_names(self) -> list[str]:
        """Names of every declared capability."""
        return [capability.name for capability in self.capabilities]

    def has_capability(self, name: str) -> bool:
        """Whether this agent advertises the capability ``name``."""
        return any(capability.name == name for capability in self.capabilities)

    async def initialize(self) -> None:
        """Prepare the agent for work. Sets status to INITIALIZED."""
        if self.status == AgentStatus.TERMINATED:
            raise RuntimeError(f"Agent '{self.name}' has already been shut down")
        self.status = AgentStatus.INITIALIZED
        logger.info("agent_initialized", agent=self.name)

    @abstractmethod
    async def perceive(self, event: Event) -> None:
        """React to an event observed on the swarm bus."""

    @abstractmethod
    async def reason(self, state: SwarmState) -> None:
        """Derive an intention from the shared swarm state."""

    @abstractmethod
    async def act(self, state: SwarmState) -> None:
        """Execute the chosen action, mutating shared state as needed."""

    async def step(self, event: Event) -> None:
        """Run one full perceive → reason → act cycle for ``event``.

        The shared :class:`SwarmState` is taken from :attr:`state`, which the
        orchestration layer assigns at registration (and test helpers assign
        when driving agents directly). Replayed events are *perceived* but
        never re-executed, so a replay stays strictly read-only.
        """
        await self.perceive(event)
        if event.replayed:
            return
        if self.state is None:
            raise RuntimeError(
                f"Agent '{self.name}' has no shared state; register it with a Swarm first"
            )
        await self.reason(self.state)
        await self.act(self.state)

    async def publish_event(self, event: Event) -> None:
        """Publish a domain ``event`` to the swarm bus.

        The orchestration layer assigns :attr:`bus` when the agent is
        registered with a :class:`Swarm` or :class:`SwarmCoordinator`, so
        agents announce domain facts (e.g. ``RequirementCreated``) through
        this method and never hold a direct reference to any peer.
        """
        if self.bus is None:
            raise RuntimeError(
                f"Agent '{self.name}' has no event bus; register it with a Swarm first"
            )
        logger.info(
            "event_published_by_agent",
            agent=self.name,
            event_type=event.type,
            correlation_id=event.correlation_id,
        )
        await self.bus.publish(event)

    async def send(
        self,
        bus: EventBus,
        intent: str,
        payload: dict[str, Any],
        *,
        receiver: str | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """Publish a message to the swarm bus without naming a concrete receiver.

        Pass ``correlation_id`` to continue the logical conversation this agent
        is part of (usually taken from the event that triggered it).
        """
        message = Message(
            sender=self.name,
            receiver=receiver,
            intent=intent,
            payload=payload,
            correlation_id=correlation_id,
        )
        logger.debug(
            "message_sent",
            agent=self.name,
            intent=intent,
            receiver=receiver,
            correlation_id=correlation_id,
        )
        await bus.publish(Event.from_message(message, source=self.name))

    async def shutdown(self) -> None:
        """Release resources and mark the agent as TERMINATED."""
        self.status = AgentStatus.TERMINATED
        logger.info("agent_terminated", agent=self.name)


def drive_on_event(agent: BaseAgent, state: SwarmState) -> EventHandler:
    """Return a bus handler that drives one full perceive → reason → act cycle.

    The orchestration layer subscribes agents to the event bus with this
    handler so a delivered event runs the agent's whole lifecycle
    automatically. The agent is bound to ``state`` (assigned if the agent does
    not hold one yet), and replayed events are *perceived* (observe-only) but
    never re-executed, so replay stays strictly read-only: it cannot re-write
    artifacts or re-publish domain events.
    """

    async def handle(event: Event) -> None:
        if agent.state is None:
            agent.state = state
        await agent.step(event)

    return handle


def route_on_event(select: Callable[[Event], BaseAgent | None], state: SwarmState) -> EventHandler:
    """Return a bus handler that routes an event to the best matching agent.

    ``select`` inspects the event and returns the single agent that should run
    (e.g. the best agent for a capability plus any tag-based specialization
    conditions). Only that agent is stepped, so specialized agents can share a
    capability without every one of them reacting to the same event. The chosen
    agent is bound to ``state`` and replayed events are perceived but never
    re-executed.
    """

    async def handle(event: Event) -> None:
        agent = select(event)
        if agent is None:
            return
        if agent.state is None:
            agent.state = state
        await agent.step(event)

    return handle
