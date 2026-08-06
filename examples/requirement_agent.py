"""Minimal requirement agent demonstrating the swarm runtime.

``SimpleRequirementAgent`` subscribes to ``requirement.requested`` messages,
turns free-text requests like "Find laptops for engineering team" into a
structured requirement, writes it into shared state as an :class:`Artifact`
and announces a ``RequirementCreated`` event for downstream agents. It uses no
LLM: the parser is a tiny deterministic heuristic so the example stays
self-contained.
"""

from typing import Any

from swarm import Artifact, Capability
from swarm.core.agent import BaseAgent
from swarm.core.event import Event, EventBus
from swarm.core.state import SwarmState

REQUIREMENT_REQUESTED = "requirement.requested"
REQUIREMENT_CREATED = "RequirementCreated"


class SimpleRequirementAgent(BaseAgent):
    """Parses procurement requests and emits a structured requirement."""

    name = "requirement_agent"
    description = "Parses free-text procurement requests into structured requirements"
    capabilities = [
        Capability(name="requirements_analysis", description="Analyzes procurement requests"),
        Capability(
            name="requirement_parsing",
            description="Parses free text into structured specs",
        ),
    ]

    def __init__(self, bus: EventBus) -> None:
        super().__init__()
        self.bus = bus
        self._pending_payload: dict[str, Any] = {}
        self._correlation_id: str | None = None
        self._parsed: dict[str, str] | None = None
        self._published = False

    async def perceive(self, event: Event) -> None:
        if event.message is not None and event.message.intent == REQUIREMENT_REQUESTED:
            self._pending_payload = dict(event.message.payload)
            self._correlation_id = event.correlation_id or event.message.correlation_id
            self._published = False

    async def reason(self, state: SwarmState) -> None:
        if self._published:
            return
        text = str(self._pending_payload.get("text", ""))
        self._parsed = self._parse(text) or None

    async def act(self, state: SwarmState) -> None:
        if self._published or self._parsed is None:
            return
        self._published = True
        state.put_artifact(
            Artifact(
                kind="requirement",
                name="requirement",
                data=self._parsed,
                created_by=self.name,
                correlation_id=self._correlation_id,
            )
        )
        if self.bus is None:
            raise RuntimeError(f"Agent '{self.name}' has no event bus")
        await self.bus.publish(
            Event(
                type=REQUIREMENT_CREATED,
                source=self.name,
                payload=self._parsed,
                correlation_id=self._correlation_id,
            )
        )

    @staticmethod
    def _parse(text: str) -> dict[str, str]:
        """Extract ``item`` and ``audience`` from "Find <item> for <audience>"."""
        stripped = text.strip()
        if not stripped.lower().startswith("find "):
            return {}
        rest = stripped[len("find ") :]
        if " for " in rest:
            item, audience = rest.split(" for ", 1)
        else:
            item, audience = rest, ""
        return {"item": item.strip(), "audience": audience.strip()}
