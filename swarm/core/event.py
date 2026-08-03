"""Event model and event bus for the swarm runtime.

Events announce facts (e.g. ``supplier_search_requested``) and travel through
the bus. The bus is fully async so it can support future messaging patterns
(request/response, fan-out, durable queues) without changing the event model.
"""

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

from swarm.core.message import Message

logger = structlog.get_logger(__name__)

ANY_EVENT = "*"


class SwarmEventType(StrEnum):
    """Well-known lifecycle events emitted by the runtime itself."""

    AGENT_REGISTERED = "agent.registered"
    AGENT_UNREGISTERED = "agent.unregistered"
    AGENT_INITIALIZED = "agent.initialized"
    AGENT_TERMINATED = "agent.terminated"
    MESSAGE = "message"
    SWARM_STARTED = "swarm.started"
    SWARM_STOPPED = "swarm.stopped"


class Event(BaseModel):
    """A notification broadcast on the swarm event bus."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    type: str
    source: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    correlation_id: str | None = None
    message: Message | None = None
    replayed: bool = False

    @classmethod
    def from_message(cls, message: Message, *, source: str | None = None) -> "Event":
        """Wrap a :class:`Message` so it can travel over the event bus.

        The message's ``correlation_id`` is carried onto the event so a logical
        conversation stays traceable across the message/event boundary.
        """
        return cls(
            type=SwarmEventType.MESSAGE,
            source=source or message.sender,
            payload={"intent": message.intent, "receiver": message.receiver},
            message=message,
            correlation_id=message.correlation_id,
        )


EventHandler = Callable[[Event], Awaitable[None]]


class EventBus:
    """In-process publish/subscribe bus.

    Subscribers register handlers for an event type; :meth:`publish` fans an
    event out to every matching handler concurrently. ``ANY_EVENT`` subscribes
    a handler to every event. A failing handler never blocks delivery to the
    other subscribers — its exception is collected and returned to the caller.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, set[EventHandler]] = defaultdict(set)
        self._event_log: list[Event] = []
        self._lock = asyncio.Lock()

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Register ``handler`` to receive events of ``event_type``."""
        self._subscribers[event_type].add(handler)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Remove a previously registered handler. No-op if absent."""
        self._subscribers[event_type].discard(handler)

    def subscriber_count(self, event_type: str) -> int:
        """Number of handlers subscribed to ``event_type`` (exact matches only)."""
        return len(self._subscribers[event_type])

    def _matching(self, event_type: str) -> set[EventHandler]:
        handlers = set(self._subscribers.get(event_type, ()))
        if event_type != ANY_EVENT:
            handlers |= self._subscribers.get(ANY_EVENT, set())
        return handlers

    async def publish(self, event: Event) -> list[Exception]:
        """Dispatch ``event`` to all matching subscribers.

        Returns any exceptions raised by handlers. Every event is recorded in
        the audit log in publish order regardless of handler outcomes.
        """
        async with self._lock:
            self._event_log.append(event)
        logger.debug(
            "event_published",
            event_id=event.id,
            event_type=event.type,
            source=event.source,
            correlation_id=event.correlation_id,
            subscribers=len(self._matching(event.type)),
        )
        return await self._deliver(event)

    async def _deliver(self, event: Event) -> list[Exception]:
        """Fan ``event`` out to current subscribers without recording it."""
        handlers = list(self._matching(event.type))
        if not handlers:
            return []

        results = await asyncio.gather(
            *(handler(event) for handler in handlers),
            return_exceptions=True,
        )
        return [result for result in results if isinstance(result, Exception)]

    async def replay(
        self,
        event_type: str | None = None,
        *,
        mode: Literal["read_only", "deliver"] = "read_only",
    ) -> None:
        """Re-deliver recorded events to the current subscribers.

        Every event in the audit log (optionally filtered by ``event_type``)
        is delivered in original order to whatever handlers are subscribed
        *now*. Nothing is re-recorded into the log.

        ``mode="read_only"`` (default) marks every delivered event with
        ``replayed=True`` so state recorders can recognize and skip them —
        replay then never mutates canonical state. ``mode="deliver"`` delivers
        the originals untouched, which is useful when deliberately replaying
        history into a fresh runtime.
        """
        events = [
            event
            for event in self._event_log
            if event_type is None or event.type == event_type
        ]
        logger.info(
            "event_replay",
            count=len(events),
            event_type=event_type or ANY_EVENT,
            mode=mode,
        )
        for event in events:
            if mode == "deliver":
                await self._deliver(event)
            else:
                await self._deliver(event.model_copy(update={"replayed": True}))

    def event_log(self) -> list[Event]:
        """Snapshot of every published event, in publish order."""
        return list(self._event_log)

    def clear(self) -> None:
        """Remove all subscribers and reset the audit log."""
        self._subscribers.clear()
        self._event_log.clear()
