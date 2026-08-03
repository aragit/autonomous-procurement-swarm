"""DecisionAgent — selects the best supplier from quotes and evaluation scores.

Filters the quotes through the existing deterministic ``PolicyEngine``
compliance rules, then ranks the survivors by evaluation score (descending) and
unit price (ascending) to pick the winner. Produces a :class:`DecisionArtifact`
and a human-readable :class:`DecisionExplanationArtifact` describing the "why"
of the selection.

Phase 3: the agent reacts only to the ``QuotesCompleted`` phase-gate event
published by the :class:`CompletionTracker`, so it always decides on the full
quote set rather than whatever quotes happened to arrive first.

Phase 4: the explanation records the active strategy and deterministic reasons
for every rejected supplier, so selections are auditable without an LLM.
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
    DECISION_EXPLANATION_ARTIFACT_NAME,
    REQUIREMENT_ARTIFACT_NAME,
    STRATEGY_ARTIFACT_NAME,
    DecisionArtifact,
    DecisionExplanationArtifact,
)
from swarm.domain.events import ProcurementEventType
from swarm.domain.pricing import DEFAULT_BID_BOND_PCT, bid_bond_amount
from swarm.domain.strategy import BALANCED_STRATEGY, DEFAULT_STRATEGIES

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
        self._explanation: dict[str, Any] | None = None
        self._quote_names: list[str] = []
        self._pending = False

    async def perceive(self, event: Event) -> None:
        if event.replayed:
            return
        if event.type == ProcurementEventType.QUOTES_COMPLETED:
            self._pending = True
            self._correlation_id = event.correlation_id
            self._decision = None
            self._explanation = None

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

        strategy_name = self._strategy_name(state)
        self._decision = {
            "selected_supplier": selected,
            "reasoning": {
                "criteria": "evaluation score (descending), unit price (ascending)",
                "ranked": ranked,
            },
        }
        self._explanation = self._build_explanation(ranked, selected, strategy_name)
        logger.info(
            "agent_executing",
            agent=self.name,
            phase="decision_made",
            selected_supplier=selected,
            correlation_id=self._correlation_id,
        )
        logger.info(
            "explanation_generated",
            agent=self.name,
            selected_supplier=selected,
            strategy_used=strategy_name,
            rejected_count=len(self._explanation["rejected_suppliers"]),
            correlation_id=self._correlation_id,
        )

    @staticmethod
    def _strategy_name(state: SwarmState) -> str:
        """The active strategy name, defaulting to balanced."""
        artifact = state.get_artifact(STRATEGY_ARTIFACT_NAME)
        if artifact is None:
            return BALANCED_STRATEGY.name
        name = str(artifact.data.get("strategy_name") or BALANCED_STRATEGY.name)
        return DEFAULT_STRATEGIES.get(name, BALANCED_STRATEGY).name

    def _build_explanation(
        self,
        ranked: list[dict[str, Any]],
        selected: str | None,
        strategy_name: str,
    ) -> dict[str, Any]:
        """A deterministic, human-readable explanation of the selection."""
        explanation: dict[str, Any] = {
            "selected_supplier": selected,
            "strategy_used": strategy_name,
            "top_factors": [],
            "rejected_suppliers": [],
        }
        strategy = DEFAULT_STRATEGIES.get(strategy_name, BALANCED_STRATEGY)
        explanation["top_factors"].append(
            f"Selection followed the {strategy_name} strategy: {strategy.description}."
        )
        if selected is None:
            explanation["top_factors"].append(
                "No supplier satisfied the policy constraints, so no award was made."
            )
            for entry in ranked:
                explanation["rejected_suppliers"].append(
                    self._rejected_entry(
                        entry, reason=entry["policy_reason"] or "Failed policy compliance"
                    )
                )
            return explanation

        winner = next(entry for entry in ranked if entry["supplier_id"] == selected)
        explanation["top_factors"].append(
            f"{selected} scored highest (composite {winner['score']:.4f}, "
            f"unit price {winner['price']:.2f}) among policy-compliant suppliers."
        )
        for entry in ranked:
            if entry["supplier_id"] == selected:
                continue
            if not entry["policy_passed"]:
                reason = entry["policy_reason"] or "Failed policy compliance"
            elif entry["score"] < winner["score"]:
                reason = "Lower composite score than the selected supplier"
            elif entry["score"] == winner["score"] and entry["price"] > winner["price"]:
                reason = "Equal composite score but higher unit price"
            else:
                reason = f"Not selected under the {strategy_name} strategy"
            explanation["rejected_suppliers"].append(self._rejected_entry(entry, reason=reason))
        return explanation

    @staticmethod
    def _rejected_entry(entry: dict[str, Any], reason: str) -> dict[str, Any]:
        return {
            "supplier_id": entry["supplier_id"],
            "score": entry["score"],
            "price": entry["price"],
            "policy_passed": entry["policy_passed"],
            "policy_reason": entry["policy_reason"],
            "reason": reason,
        }

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
        if self._explanation is not None:
            explanation_artifact = DecisionExplanationArtifact(
                name=DECISION_EXPLANATION_ARTIFACT_NAME,
                data=self._explanation,
                parent_ids=[artifact.name],
                created_by=self.name,
                correlation_id=self._correlation_id,
            )
            state.put_artifact(explanation_artifact)
        self._pending = False
        self._decision = None
        self._explanation = None
