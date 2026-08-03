"""Minimal supplier discovery agent demonstrating the swarm runtime.

``SimpleSupplierDiscoveryAgent`` reacts to ``RequirementCreated`` events, looks
up a small static candidate list for the required item and records the matches
in shared state as an :class:`Artifact`. It is a stub on purpose: real
discovery (catalog queries, LLM-based matching) is a later phase.
"""

from typing import Any

from swarm import Artifact, Capability
from swarm.core.agent import BaseAgent
from swarm.core.event import Event
from swarm.core.state import SwarmState

REQUIREMENT_CREATED = "RequirementCreated"


class SimpleSupplierDiscoveryAgent(BaseAgent):
    """Finds candidate suppliers for a published requirement."""

    name = "supplier_discovery_agent"
    description = "Finds candidate suppliers for a published requirement"
    capabilities = [Capability(name="supplier_discovery", description="Matches suppliers to items")]

    def __init__(self) -> None:
        super().__init__()
        self._requirement: dict[str, Any] = {}
        self._correlation_id: str | None = None
        self._candidates: list[dict[str, Any]] | None = None

    async def perceive(self, event: Event) -> None:
        if event.type == REQUIREMENT_CREATED:
            self._requirement = dict(event.payload)
            self._correlation_id = event.correlation_id
            self._candidates = None

    async def reason(self, state: SwarmState) -> None:
        if self._candidates is not None or not self._requirement:
            return
        item = str(self._requirement.get("item", "unknown"))
        self._candidates = self._candidates_for(item)

    async def act(self, state: SwarmState) -> None:
        if self._candidates is None:
            return
        state.put_artifact(
            Artifact(
                kind="supplier_shortlist",
                name="suppliers_found",
                data={"candidates": self._candidates},
                created_by=self.name,
                correlation_id=self._correlation_id,
            )
        )

    @staticmethod
    def _candidates_for(item: str) -> list[dict[str, Any]]:
        """Return static candidate suppliers for ``item``."""
        return [
            {"supplier": "LaptopCorp_A", "item": item, "match_score": 0.92},
            {"supplier": "TechDistrib_B", "item": item, "match_score": 0.87},
        ]
