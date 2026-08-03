"""DecisionAgent — selects the best supplier from quotes and evaluation scores.

Filters the quotes through the existing deterministic ``PolicyEngine``
compliance rules, then ranks the survivors by evaluation score (descending) and
unit price (ascending) to pick the winner. Produces a :class:`DecisionArtifact`.

Phase 3: the agent reacts only to the ``QuotesCompleted`` phase-gate event
published by the :class:`CompletionTracker`, so it always decides on the full
quote set rather than whatever quotes happened to arrive first.
"""

from typing import Any

import structlog

from configs.settings import settings
from core.protocol.policy_engine import PolicyContext, PolicyEngine
from swarm.core.agent import BaseAgent
from swarm.core.capability import Capability
from swarm.core.event import Event
from swarm.core.state import SwarmState
from swarm.domain.artifacts import (
    REQUIREMENT_ARTIFACT_NAME,
    DecisionArtifact,
)
from swarm.domain.events import ProcurementEventType
from swarm.domain.pricing import DEFAULT_BID_BOND_PCT, bid_bond_amount

logger = structlog.get_logger(__name__)


class DecisionAgent(BaseAgent):
    """Selects the best supplier using evaluation score and price."""

    name = "decision_agent"
    description = "Selects the best supplier from quotes and evaluation scores"
    capabilities = [
        Capability(
            name="supplier.select",
            description="Picks the winning supplier from evaluated quotes",
        )
    ]

    def __init__(self) -> None:
        super().__init__()
        self._policy = PolicyEngine()
        self._correlation_id: str | None = None
        self._decision: dict[str, Any] | None = None
        self._quote_names: list[str] = []
        self._pending = False

    async def perceive(self, event: Event) -> None:
        if event.replayed:
            return
        if event.type == ProcurementEventType.QUOTES_COMPLETED:
            self._pending = True
            self._correlation_id = event.correlation_id
            self._decision = None

    async def reason(self, state: SwarmState) -> None:
        if not self._pending:
            return
        quotes = state.find_artifacts(kind="quote", correlation_id=self._correlation_id)
        if not quotes:
            self._pending = False
            return
        self._quote_names = [quote.name for quote in quotes]
        requirement = state.get_artifact(REQUIREMENT_ARTIFACT_NAME)
        constraints = requirement.data.get("constraints", {}) if requirement is not None else {}
        quantity = int(constraints.get("quantity") or 1000)
        budget = float(constraints.get("budget") or 500_000.0)
        max_unit_price = constraints.get("max_unit_price")
        material = str(constraints.get("material") or "steel")

        max_carbon_per_unit = settings.evaluation.esg_baselines.get(material, 10_000.0)
        policy_ctx = PolicyContext(
            buyer_max_budget_total=budget,
            max_unit_price=float(max_unit_price) if max_unit_price else float("inf"),
            max_carbon_limit_kg=max_carbon_per_unit * quantity,
        )

        ranked: list[dict[str, Any]] = []
        for quote in quotes:
            price = float(quote.data["price"])
            evaluation = state.get_artifact(f"evaluation_{quote.data['supplier_id']}")
            score = float(evaluation.data["score"]) if evaluation is not None else 0.0
            bid = {
                "supplier_id": quote.data["supplier_id"],
                "unit_price": price,
                "quantity": quantity,
                "carbon_footprint_kg": quote.data["metadata"]["carbon_footprint_kg"],
                "bid_bond_amount": bid_bond_amount(price, quantity, DEFAULT_BID_BOND_PCT),
            }
            passed, reason = self._policy.evaluate_bid(bid, policy_ctx)
            ranked.append(
                {
                    "supplier_id": quote.data["supplier_id"],
                    "score": score,
                    "price": price,
                    "policy_passed": passed,
                    "policy_reason": reason,
                }
            )

        ranked.sort(key=lambda entry: (-entry["score"], entry["price"]))
        compliant = [entry for entry in ranked if entry["policy_passed"]]
        selected = compliant[0]["supplier_id"] if compliant else None

        self._decision = {
            "selected_supplier": selected,
            "reasoning": {
                "criteria": "evaluation score (descending), unit price (ascending)",
                "ranked": ranked,
            },
        }
        logger.info(
            "agent_executing",
            agent=self.name,
            phase="decision_made",
            selected_supplier=selected,
            correlation_id=self._correlation_id,
        )

    async def act(self, state: SwarmState) -> None:
        if not self._pending or self._decision is None:
            return
        artifact = DecisionArtifact(
            data=self._decision,
            parent_ids=list(self._quote_names),
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
                type=ProcurementEventType.DECISION_MADE,
                source=self.name,
                payload={
                    "artifact": artifact.name,
                    "selected_supplier": self._decision["selected_supplier"],
                },
                correlation_id=self._correlation_id,
            )
        )
        self._pending = False
        self._decision = None
