"""Completion tracking for multi-stage swarm flows.

Agents publish one per-item event per artifact (e.g. ``SupplierEvaluated`` for
each supplier) instead of one big batch event. To know when a *phase* is done,
the swarm needs to count those artifacts against a declared expectation. That
is what :class:`CompletionTracker` does: it consumes every event, closes a
group once the expected number of artifacts exists, and publishes the group's
completion event exactly once.
"""

from collections.abc import Mapping
from typing import Any

import structlog

from swarm.core.event import Event, EventBus, EventHandler
from swarm.core.state import SwarmState

logger = structlog.get_logger(__name__)

COMPLETION_SOURCE = "completion_tracker"


class CompletionTracker:
    """Consumes every event and closes completion groups once they are met.

    A producer agent declares expectations up front via
    :meth:`SwarmState.expect_artifact` (e.g. "this request expects 5
    ``evaluation`` artifacts"). The tracker subscribes to ``ANY_EVENT`` and,
    after each delivery, checks every undeclared-complete group: once
    ``SwarmState.completed_artifact_count`` reaches the expected size it marks
    the group complete and publishes the mapped completion event exactly once.
    Groups without a mapped completion event are still marked complete, which
    keeps later expectations keyed on the same group deterministic.
    """

    def __init__(
        self,
        state: SwarmState,
        bus: EventBus,
        *,
        completion_events: Mapping[str, str] | None = None,
    ) -> None:
        self._state = state
        self._bus = bus
        self._completion_events = dict(completion_events or {})
        self._published: set[tuple[str, str]] = set()

    @property
    def handler(self) -> EventHandler:
        """Subscribe this handler to ``ANY_EVENT`` on the shared bus."""
        return self._on_event

    def _mark_if_complete(self, correlation_id: str, kind: str) -> str | None:
        """Close ``kind`` for a request when its expectation is met, else None.

        Fully synchronous, so concurrent deliveries can never double-close or
        double-publish a group.
        """
        key = (correlation_id, kind)
        if key in self._published:
            return None
        expected = self._state.expected_count(correlation_id, kind)
        if expected is None:
            return None
        if self._state.completed_artifact_count(correlation_id, kind) < expected:
            return None
        self._state.complete_artifact(kind, correlation_id=correlation_id)
        self._published.add(key)
        return kind

    async def _on_event(self, event: Event) -> None:
        """Re-check every expected group after a new event is delivered."""
        if event.replayed:
            return
        for correlation_id, groups in list(self._state.expectations.items()):
            for kind in list(groups):
                if self._mark_if_complete(correlation_id, kind) is None:
                    continue
                event_type = self._completion_events.get(kind)
                if event_type is None:
                    continue
                logger.info(
                    "group_completed",
                    group=kind,
                    correlation_id=correlation_id,
                    count=groups[kind],
                    event_type=event_type,
                )
                await self._bus.publish(
                    Event(
                        type=event_type,
                        source=COMPLETION_SOURCE,
                        payload={
                            "group": kind,
                            "count": groups[kind],
                            "artifacts": [
                                artifact.name
                                for artifact in self._state.find_artifacts(
                                    kind=kind, correlation_id=correlation_id
                                )
                            ],
                        },
                        correlation_id=correlation_id,
                    )
                )

    def to_dict(self) -> dict[str, Any]:
        """Serializable view of tracker state, for diagnostics."""
        return {
            "published": sorted(self._published),
            "completion_events": dict(self._completion_events),
        }
