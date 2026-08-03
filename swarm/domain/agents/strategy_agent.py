"""StrategyAgent — selects the execution strategy for a requirement.

Reacts to ``RequirementCreated`` and picks a :class:`Strategy` from the
requirement's constraints with a pure rule (:func:`select_strategy`): a strict
carbon constraint yields ``low_carbon``, a tight budget yields
``cost_optimized``, otherwise ``balanced``. Publishes a ``StrategySelected``
event and a :class:`StrategyArtifact` that the evaluation agent reads, so the
strategy artifact always exists before any supplier is evaluated.
"""

import structlog

from swarm.core.agent import BaseAgent
from swarm.core.capability import Capability
from swarm.core.event import Event
from swarm.core.state import SwarmState
from swarm.domain.artifacts import REQUIREMENT_ARTIFACT_NAME, StrategyArtifact
from swarm.domain.events import ProcurementEventType
from swarm.domain.strategy import Strategy, select_strategy

logger = structlog.get_logger(__name__)


class StrategyAgent(BaseAgent):
    """Selects a deterministic execution strategy for a requirement."""

    name = "strategy_agent"
    description = "Selects the execution strategy for a requirement"
    capabilities = [
        Capability(
            name="strategy.select",
            description="Chooses a scoring strategy from the requirement constraints",
        )
    ]

    def __init__(self) -> None:
        super().__init__()
        self._correlation_id: str | None = None
        self._requirement_artifact: str = REQUIREMENT_ARTIFACT_NAME
        self._strategy: Strategy | None = None
        self._pending = False

    async def perceive(self, event: Event) -> None:
        if event.replayed:
            return
        if event.type == ProcurementEventType.REQUIREMENT_CREATED:
            self._pending = True
            self._correlation_id = event.correlation_id
            self._requirement_artifact = str(
                event.payload.get("artifact", REQUIREMENT_ARTIFACT_NAME)
            )
            self._strategy = None

    async def reason(self, state: SwarmState) -> None:
        if not self._pending:
            return
        requirement = state.get_artifact(self._requirement_artifact)
        if requirement is None:
            self._pending = False
            return
        constraints = requirement.data.get("constraints", {})
        self._strategy = select_strategy(constraints)
        logger.info(
            "strategy_selected",
            agent=self.name,
            strategy_name=self._strategy.name,
            correlation_id=self._correlation_id,
        )

    async def act(self, state: SwarmState) -> None:
        if not self._pending or self._strategy is None:
            return
        artifact = StrategyArtifact(
            data={
                "strategy_name": self._strategy.name,
                "description": self._strategy.description,
                "weights": self._strategy.as_weights(),
            },
            parent_ids=[self._requirement_artifact],
            created_by=self.name,
            correlation_id=self._correlation_id,
        )
        state.put_artifact(artifact)
        logger.info(
            "artifact_created",
            agent=self.name,
            kind=artifact.kind,
            name=artifact.name,
            correlation_id=self._correlation_id,
        )
        await self.publish_event(
            Event(
                type=ProcurementEventType.STRATEGY_SELECTED,
                source=self.name,
                payload={
                    "artifact": artifact.name,
                    "strategy_name": self._strategy.name,
                    "weights": self._strategy.as_weights(),
                },
                correlation_id=self._correlation_id,
            )
        )
        self._pending = False
        self._strategy = None
