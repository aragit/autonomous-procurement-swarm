"""EvaluatorActor — evaluates suppliers and writes SCORE and RISK channels.

Wraps the deterministic EvaluationAgent and RiskAssessmentAgent logic.
"""

from __future__ import annotations

from typing import Any, cast

import ray
import structlog
from ray.actor import ActorHandle

from configs.settings import settings
from core.evaluator.scoring import EvaluationWeights, MultiCriteriaEvaluator
from core.protocol.schema import BidPayload
from mesh.actors.base import MeshActor
from mesh.channels import ChannelType
from swarm.domain.pricing import (
    DEFAULT_BID_BOND_PCT,
    bid_bond_amount,
    carbon_footprint,
    floor_price,
    lead_time_days,
)
from swarm.domain.risk import (
    classify_risk,
    compute_risk_scores,
)
from swarm.domain.strategy import BALANCED_STRATEGY, Strategy
from swarm.memory import SupplierMemoryStore

logger = structlog.get_logger(__name__)

DEFAULT_DELIVERY_DATE = "2026-08-15"


@ray.remote(max_restarts=3, max_task_retries=3)
class EvaluatorActor(MeshActor):
    """Scores suppliers and assesses risk.

    Reads from DISCOVERY channel, computes evaluation scores and risk assessments,
    writes to SCORE and RISK channels.
    """

    def __init__(
        self,
        actor_id: str,
        blackboard: ActorHandle[Any],
        kernel: ActorHandle[Any] | None = None,
        memory: SupplierMemoryStore | None = None,
    ) -> None:
        super().__init__(actor_id, "evaluator", blackboard, kernel)
        self._evaluator = MultiCriteriaEvaluator(
            weights=EvaluationWeights(),
            esg_baselines=dict(settings.evaluation.esg_baselines),
        )
        legacy = settings.evaluation.scoring_weights
        self._lead_time_weight = float(legacy.get("lead_time", 0.25))
        self._reliability_weight = float(legacy.get("reliability", 0.15))
        self._quality_weight_total = self._lead_time_weight + self._reliability_weight
        self._memory = memory
        self._processed_suppliers: set[tuple[str, str]] = set()  # (correlation_id, supplier_id)

    async def perceive(self, blackboard: ActorHandle[Any]) -> dict[str, Any]:
        """Read DISCOVERY channel for new supplier discoveries."""
        traces = await self.read_channel(ChannelType.DISCOVERY, limit=50)
        return {"discoveries": traces}

    async def reason(self, perception: dict[str, Any]) -> dict[str, Any]:
        """Process discoveries and compute evaluations + risk."""
        discoveries = perception.get("discoveries", [])
        proposals = []

        for trace in discoveries:
            payload = trace.get("payload", {})
            if payload.get("type") != "supplier_discovered":
                continue

            correlation_id = trace.get("correlation_id", "")
            supplier_id = str(payload.get("supplier_id") or "")
            trace_id = trace.get("id", "")
            pool_trace_id = payload.get("pool_trace_id", "")

            if not supplier_id or (correlation_id, supplier_id) in self._processed_suppliers:
                continue

            # Need to read the pool to get supplier details
            pool_traces = await self.read_channel(ChannelType.DISCOVERY, limit=20)
            pool_trace = next(
                (
                    t
                    for t in pool_traces
                    if t.get("id") == pool_trace_id
                    and t.get("payload", {}).get("type") == "supplier_list"
                ),
                None,
            )
            if not pool_trace:
                # Try to find by correlation_id
                pool_trace = next(
                    (
                        t
                        for t in pool_traces
                        if t.get("correlation_id") == correlation_id
                        and t.get("payload", {}).get("type") == "supplier_list"
                    ),
                    None,
                )

            if not pool_trace:
                logger.warning(
                    "evaluator_pool_not_found",
                    actor_id=self.actor_id,
                    correlation_id=correlation_id,
                    supplier_id=supplier_id,
                )
                continue

            pool_data = pool_trace.get("payload", {}).get("data", {})
            supplier = self._find_supplier(pool_data, supplier_id)
            if not supplier:
                continue

            # Also need requirement for constraints
            req_traces = await self.read_channel(ChannelType.REQUIREMENT, limit=5)
            req_trace = next(
                (t for t in req_traces if t.get("correlation_id") == correlation_id),
                None,
            )
            requirement_data = (
                req_trace.get("payload", {}).get("requirement", {}) if req_trace else {}
            )

            # Read strategy if available
            # For now, use balanced strategy (the strategy agent would write to
            # the REQUIREMENT channel in a full implementation)
            strategy = BALANCED_STRATEGY

            # Compute evaluation
            evaluation = self._compute_evaluation(
                supplier=supplier,
                pool_data=pool_data,
                requirement_data=requirement_data,
                strategy=strategy,
                correlation_id=correlation_id,
            )

            # Compute risk
            risk = self._compute_risk(
                supplier=supplier,
                pool_data=pool_data,
                requirement_data=requirement_data,
                evaluation_score=evaluation["score"],
                correlation_id=correlation_id,
            )

            proposals.append(
                {
                    "correlation_id": correlation_id,
                    "supplier_id": supplier_id,
                    "discovery_trace_id": trace_id,
                    "pool_trace_id": pool_trace_id,
                    "evaluation": evaluation,
                    "risk": risk,
                    "confidence": 1.0,
                }
            )
            self._processed_suppliers.add((correlation_id, supplier_id))

        return {"proposals": proposals}

    async def act(self, blackboard: ActorHandle[Any], proposal: dict[str, Any]) -> None:
        """Write evaluation to SCORE channel and risk to RISK channel."""
        for prop in proposal.get("proposals", []):
            correlation_id = prop["correlation_id"]
            supplier_id = prop["supplier_id"]
            discovery_trace_id = prop["discovery_trace_id"]
            prop["pool_trace_id"]

            # Write to SCORE channel
            eval_trace_id = await self.write_channel(
                ChannelType.SCORE,
                {
                    "type": "evaluation",
                    "data": prop["evaluation"],
                    "correlation_id": correlation_id,
                    "supplier_id": supplier_id,
                },
                parent_ids=[discovery_trace_id],
            )

            # Write to RISK channel
            risk_trace_id = await self.write_channel(
                ChannelType.RISK,
                {
                    "type": "risk_assessment",
                    "data": prop["risk"],
                    "correlation_id": correlation_id,
                    "supplier_id": supplier_id,
                },
                parent_ids=[eval_trace_id],
            )

            logger.info(
                "evaluator_completed",
                actor_id=self.actor_id,
                correlation_id=correlation_id,
                supplier_id=supplier_id,
                score=prop["evaluation"]["score"],
                risk_level=prop["risk"]["risk_level"],
                eval_trace_id=eval_trace_id,
                risk_trace_id=risk_trace_id,
            )

    def _compute_evaluation(
        self,
        supplier: dict[str, Any],
        pool_data: dict[str, Any],
        requirement_data: dict[str, Any],
        strategy: Strategy,
        correlation_id: str,
    ) -> dict[str, Any]:
        """Compute multi-criteria evaluation score."""
        material = str(pool_data.get("material") or "steel")
        quantity = int(pool_data.get("quantity") or 1000)
        target_lead = int(pool_data.get("target_lead_time_days") or 30)
        spot = float(pool_data.get("spot_price") or 100.0)
        session_id = (correlation_id or "procurement")[:32]
        index = self._supplier_index(pool_data, supplier_id=str(supplier["supplier_id"]))

        bid = self._synthetic_bid(supplier, quantity, target_lead, index, session_id)
        price = self._evaluator._score_price(bid.unit_price, spot)
        lead_time = self._evaluator._score_lead_time(bid.lead_time_days, target_lead)
        esg = self._evaluator._score_esg(bid.carbon_footprint_kg, material, quantity)
        reliability = self._evaluator._score_reliability(bid.reliability_score)
        quality = self._quality_score(lead_time, reliability)

        base_score = round(
            strategy.price_weight * price
            + strategy.score_weight * quality
            + strategy.carbon_weight * esg,
            4,
        )

        history = self._supplier_history(supplier_id=str(supplier["supplier_id"]))
        adjustment = SupplierMemoryStore.history_adjustment(history) if history is not None else 0.0
        score = round(max(0.0, min(1.0, base_score + adjustment)), 4)

        return {
            "supplier_id": str(supplier["supplier_id"]),
            "score": score,
            "breakdown": {
                "price": price,
                "lead_time": lead_time,
                "esg": esg,
                "reliability": reliability,
                "quality": quality,
            },
            "strategy": {
                "strategy_name": strategy.name,
                "weights": strategy.as_weights(),
            },
            "history": {
                "applied": history is not None,
                "adjustment": adjustment,
                "reliability": round(history.delivery_reliability, 4)
                if history is not None
                else None,
                "total_orders": history.total_orders if history is not None else 0,
            },
            "bid": bid.model_dump(),
        }

    def _compute_risk(
        self,
        supplier: dict[str, Any],
        pool_data: dict[str, Any],
        requirement_data: dict[str, Any],
        evaluation_score: float,
        correlation_id: str,
    ) -> dict[str, Any]:
        """Compute deterministic risk assessment."""
        supplier_id = str(supplier["supplier_id"])
        decision_id = correlation_id  # Use correlation_id as decision_id for traceability

        constraints = requirement_data.get("constraints", {})
        quantity = int(constraints.get("quantity") or 1000)
        max_carbon_per_unit = constraints.get("max_carbon_per_unit")
        material = str(constraints.get("material") or "steel")
        constraints.get("max_unit_price")
        float(constraints.get("budget") or 500_000.0)

        # Purchase amount from quote (will be generated by negotiator)
        # For now, use floor_price * quantity as estimate
        unit_price = floor_price(supplier)
        purchase_amount = round(unit_price * quantity, 2)

        carbon_per_unit = carbon_footprint(supplier, quantity)
        if quantity > 0:
            carbon_per_unit = round(carbon_per_unit / quantity, 4)

        baseline = float(settings.evaluation.esg_baselines.get(material, 10_000.0))

        # Policy limits
        max_purchase_amount = 5_000_000.0  # Standard policy

        performance = self._supplier_history(supplier_id)

        scores = compute_risk_scores(
            purchase_amount=purchase_amount,
            max_purchase_amount=max_purchase_amount,
            performance=performance,
            evaluation_score=evaluation_score,
            carbon_per_unit=carbon_per_unit,
            max_carbon_per_unit=float(max_carbon_per_unit) if max_carbon_per_unit else None,
            esg_baseline=baseline,
        )

        risk_level = classify_risk(scores["overall_risk_score"])

        return {
            "supplier_id": supplier_id,
            "decision_id": decision_id,
            "risk_id": risk_level.value.lower() + "_" + supplier_id,
            "purchase_amount": purchase_amount,
            "risk_scores": scores,
            "risk_level": risk_level.value,
            "policy_name": "standard",
        }

    def _quality_score(self, lead_time: float, reliability: float) -> float:
        return (
            self._lead_time_weight * lead_time + self._reliability_weight * reliability
        ) / self._quality_weight_total

    def _supplier_history(self, supplier_id: str) -> Any:
        if self._memory is None:
            return None
        return self._memory.get_supplier_performance(supplier_id)

    @staticmethod
    def _find_supplier(pool: dict[str, Any], supplier_id: str) -> dict[str, Any] | None:
        for supplier in pool.get("suppliers", []):
            if str(supplier["supplier_id"]) == supplier_id:
                return cast("dict[str, Any]", supplier)
        return None

    @staticmethod
    def _supplier_index(pool: dict[str, Any], supplier_id: str) -> int:
        for index, supplier in enumerate(pool.get("suppliers", [])):
            if str(supplier["supplier_id"]) == supplier_id:
                return index
        return 0

    def _synthetic_bid(
        self,
        supplier: dict[str, Any],
        quantity: int,
        target_lead: int,
        index: int,
        session_id: str,
    ) -> BidPayload:
        supplier_id = str(supplier["supplier_id"])
        unit_price = floor_price(supplier)
        return BidPayload(
            session_id=session_id,
            supplier_id=supplier_id,
            unit_price=unit_price,
            lead_time_days=lead_time_days(supplier, target_lead, index),
            carbon_footprint_kg=carbon_footprint(supplier, quantity),
            reliability_score=float(supplier["reliability_score"]),
            bid_bond_amount=bid_bond_amount(unit_price, quantity, DEFAULT_BID_BOND_PCT),
            delivery_date=DEFAULT_DELIVERY_DATE,
            justification=f"Synthetic pre-quote profile for {supplier_id}",
        )
