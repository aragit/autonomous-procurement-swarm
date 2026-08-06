"""EvaluationAgent — scores each discovered supplier with the core evaluator.

Builds a deterministic pre-quote ``BidPayload`` per supplier from its
CostModel-shaped profile, then reuses ``MultiCriteriaEvaluator`` (with the
application's evaluation weights and ESG baselines) to produce one
``EvaluationArtifact`` per supplier.

Phase 3: the agent reacts to each ``SupplierDiscovered`` event individually and
publishes one ``SupplierEvaluated`` event per supplier, so suppliers are
evaluated as soon as they are discovered — the evaluation stage runs in
parallel instead of waiting for the whole pool.

Phase 4: the composite score is blended with the strategy weights from the
:class:`StrategyArtifact` (price / score / carbon). The ``balanced`` strategy
reproduces the Phase 3 scoring exactly; when no strategy artifact exists the
balanced strategy is assumed (unit tests, direct drives).

Phase 5: the composite is further adjusted by the supplier's deterministic
history (:class:`SupplierMemoryStore`): strong reliability raises the score and
poor reliability lowers it. When no supplier memory is attached, or the supplier
has no recorded history, the adjustment is zero and Phase 4 scores are unchanged.
"""

from typing import Any, cast

import structlog

from configs.settings import settings
from core.evaluator.scoring import EvaluationWeights, MultiCriteriaEvaluator
from core.protocol.schema import BidPayload
from swarm.core.agent import BaseAgent
from swarm.core.capability import Capability
from swarm.core.event import Event
from swarm.core.state import SwarmState
from swarm.domain.artifacts import (
    STRATEGY_ARTIFACT_NAME,
    SUPPLIER_LIST_ARTIFACT_NAME,
    EvaluationArtifact,
)
from swarm.domain.events import ProcurementEventType
from swarm.domain.pricing import bid_bond_amount, carbon_footprint, floor_price, lead_time_days
from swarm.domain.strategy import BALANCED_STRATEGY, DEFAULT_STRATEGIES, Strategy
from swarm.learning.adaptive_policy import (
    apply_strategy_weights,
    get_strategy_type,
    get_strategy_weights,
)
from swarm.memory import SupplierMemoryStore

logger = structlog.get_logger(__name__)

DEFAULT_DELIVERY_DATE = "2026-08-15"


class EvaluationAgent(BaseAgent):
    """Scores one discovered supplier using the multi-criteria evaluator."""

    name = "evaluation_agent"
    description = "Scores discovered suppliers using the multi-criteria evaluator"
    capabilities = [
        Capability(
            name="supplier.evaluate",
            description="Scores supplier bids against the requirement",
        )
    ]

    def __init__(
        self,
        *,
        weights: EvaluationWeights | None = None,
        esg_baselines: dict[str, float] | None = None,
        memory: Any = None,
    ) -> None:
        super().__init__(memory=memory)
        self._evaluator = MultiCriteriaEvaluator(
            weights=weights or EvaluationWeights(),
            esg_baselines=(
                esg_baselines
                if esg_baselines is not None
                else dict(settings.evaluation.esg_baselines)
            ),
        )
        # Legacy weights used to derive the quality sub-score from the
        # lead-time and reliability components of the core evaluator.
        legacy = settings.evaluation.scoring_weights
        self._lead_time_weight = float(legacy.get("lead_time", 0.25))
        self._reliability_weight = float(legacy.get("reliability", 0.15))
        self._quality_weight_total = self._lead_time_weight + self._reliability_weight
        self._correlation_id: str | None = None
        self._list_artifact: str = SUPPLIER_LIST_ARTIFACT_NAME
        self._supplier_id: str = ""
        self._evaluation: dict[str, Any] | None = None
        self._pending = False

    async def perceive(self, event: Event) -> None:
        if event.replayed:
            return
        if event.type == ProcurementEventType.SUPPLIER_DISCOVERED:
            self._pending = True
            self._correlation_id = event.correlation_id
            self._supplier_id = str(event.payload.get("supplier_id") or "")
            self._list_artifact = str(event.payload.get("artifact", SUPPLIER_LIST_ARTIFACT_NAME))
            self._evaluation = None

    async def reason(self, state: SwarmState) -> None:
        if not self._pending:
            return
        pool = state.get_artifact(self._list_artifact)
        if pool is None:
            self._pending = False
            return
        supplier = self._find_supplier(pool, self._supplier_id)
        if supplier is None:
            self._pending = False
            return
        material = str(pool.data.get("material") or "steel")
        quantity = int(pool.data.get("quantity") or 1000)
        target_lead = int(pool.data.get("target_lead_time_days") or 30)
        spot = float(pool.data.get("spot_price") or 100.0)
        session_id = (self._correlation_id or "procurement")[:32]
        index = self._supplier_index(pool, self._supplier_id)

        strategy = self._read_strategy(state)
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
        history = self._supplier_history(state)
        adjustment = SupplierMemoryStore.history_adjustment(history) if history is not None else 0.0
        score = round(max(0.0, min(1.0, base_score + adjustment)), 4)
        breakdown = {
            "price": price,
            "lead_time": lead_time,
            "esg": esg,
            "reliability": reliability,
            "quality": quality,
        }
        self._evaluation = {
            "supplier_id": self._supplier_id,
            "score": score,
            "breakdown": breakdown,
            "strategy": {
                "strategy_name": strategy.name,
                "weights": strategy.as_weights(),
            },
            "history": {
                "applied": history is not None,
                "adjustment": adjustment,
                "reliability": (
                    round(history.delivery_reliability, 4) if history is not None else None
                ),
                "total_orders": history.total_orders if history is not None else 0,
            },
            "bid": bid.model_dump(),
        }
        logger.info(
            "agent_executing",
            agent=self.name,
            phase="supplier_evaluated",
            supplier_id=self._supplier_id,
            score=score,
            strategy=strategy.name,
            correlation_id=self._correlation_id,
        )

    def _quality_score(self, lead_time: float, reliability: float) -> float:
        """Blend lead time and reliability into a single quality sub-score.

        Uses the relative legacy weights, so the ``balanced`` strategy's
        ``score_weight * quality`` exactly reproduces the Phase 3 composite.
        """
        return (
            self._lead_time_weight * lead_time + self._reliability_weight * reliability
        ) / self._quality_weight_total

    def _supplier_history(self, state: SwarmState) -> Any:
        """The supplier's performance record from the attached memory store.

        Returns ``None`` (→ zero adjustment, Phase 4 behavior) when no memory
        store is attached or the supplier has no recorded history.
        """
        memory = getattr(self, "memory", None)
        if not isinstance(memory, SupplierMemoryStore):
            return None
        return memory.get_supplier_performance(self._supplier_id)

    @staticmethod
    def _read_strategy(state: SwarmState) -> Strategy:
        """The active strategy artifact, defaulting to balanced.

        v1.1 Step 22: when strategy weights are pinned (via the
        ``override_strategy_weights`` ContextVar — set by the policy learner
        during candidate evaluation, and by the active promoted policy), those
        weights override the canonical artifact weights so candidate policies
        are evaluated with full replay parity. When no override is pinned
        (live operation or no promoted policy) the canonical, artifact-derived
        strategy is used unchanged.

        v1.1 Step 23: the strategy *type* is resolved via
        :func:`get_strategy_type` with the same priority (override → active
        policy → config default). If the resolved type differs from
        ``"balanced"``, the weight transformation is applied deterministically
        to the base weights, then re-clamped and normalized.

        v1.1 Step 24: when the active policy uses a routing strategy, the
        strategy type is selected per-trace via context routing:
        ``context → strategy``. Context is extracted from the trace pool data
        and passed to :func:`get_strategy_type` for routing rule evaluation.

        v1.1 Step 25: param overrides from the routing strategy are applied to
        the resolved weights, adjusting thresholds and weight deltas per-trace.
        """
        from swarm.learning.adaptive_policy import get_param_overrides
        from swarm.learning.context import extract_context as _extract_context
        from swarm.learning.policy import clamp_weights
        from swarm.learning.routing import apply_param_overrides

        artifact = state.get_artifact(STRATEGY_ARTIFACT_NAME)
        if artifact is None:
            base = BALANCED_STRATEGY
        else:
            name = str(artifact.data.get("strategy_name") or "balanced")
            base = DEFAULT_STRATEGIES.get(name, BALANCED_STRATEGY)

        # Extract context from state for routing (Step 24).
        context = EvaluationAgent._extract_context_from_state(state, _extract_context)
        strat_type = get_strategy_type(context=context)

        # Resolve base weights (override or canonical).
        override = get_strategy_weights()
        if override is None:
            if strat_type == "balanced":
                resolved_weights = base.as_weights()
            else:
                transformed = apply_strategy_weights(strat_type, base.as_weights())
                resolved_weights = transformed
        else:
            if strat_type == "balanced":
                resolved_weights = dict(override)
            else:
                resolved_weights = apply_strategy_weights(strat_type, override)

        # Apply param overrides on weights (Step 25).
        param_overrides = get_param_overrides(context)
        weight_overrides = {
            k: v
            for k, v in param_overrides.items()
            if k in ("price_weight", "score_weight", "carbon_weight")
        }
        if weight_overrides:
            thresholds_override = dict.fromkeys(
                ("confidence_threshold", "stability_threshold", "trust_threshold"), 0.7
            )
            _, resolved_weights = apply_param_overrides(
                thresholds_override, clamp_weights(resolved_weights), weight_overrides
            )

        return Strategy(
            name=base.name,
            description=base.description,
            price_weight=resolved_weights["price_weight"],
            score_weight=resolved_weights["score_weight"],
            carbon_weight=resolved_weights["carbon_weight"],
        )

    @staticmethod
    def _extract_context_from_state(
        state: SwarmState, extract_fn: Any
    ) -> dict[str, Any]:
        """Extract routing context from the state's supplier pool artifact.

        Builds a trace-input-compatible dict from the pool artifact and applies
        :func:`extract_context` to produce the routing context deterministically.
        """
        pool = state.get_artifact(SUPPLIER_LIST_ARTIFACT_NAME)
        if pool is None:
            return {"budget_level": "medium", "urgency": "low", "supplier_count": 3}
        pool_data = pool.data if hasattr(pool, "data") else {}
        trace_input = {
            "material": str(pool_data.get("material", "steel")),
            "quantity": int(pool_data.get("quantity", 1000)),
            "budget": float(pool_data.get("budget", 0.0)),
            "target_lead_time_days": int(pool_data.get("target_lead_time_days", 30)),
            "max_carbon_per_unit": pool_data.get("max_carbon_per_unit"),
            "goal": pool_data.get("goal"),
            "supplier_count": len(pool_data.get("suppliers", [])) or int(
                pool_data.get("supplier_count", 3)
            ),
        }
        result: dict[str, Any] = extract_fn(trace_input)
        return result

    async def act(self, state: SwarmState) -> None:
        if not self._pending or self._evaluation is None:
            return
        artifact = EvaluationArtifact(
            name=f"evaluation_{self._supplier_id}",
            data=self._evaluation,
            parent_ids=[self._list_artifact],
            tags={"supplier": self._supplier_id},
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
                type=ProcurementEventType.SUPPLIER_EVALUATED,
                source=self.name,
                payload={
                    "supplier_id": self._supplier_id,
                    "artifact": artifact.name,
                    "score": self._evaluation["score"],
                },
                correlation_id=self._correlation_id,
            )
        )
        self._pending = False
        self._evaluation = None

    @staticmethod
    def _find_supplier(pool: Any, supplier_id: str) -> dict[str, Any] | None:
        """The pool entry matching ``supplier_id``, or None."""
        for supplier in pool.data["suppliers"]:
            if str(supplier["supplier_id"]) == supplier_id:
                return cast("dict[str, Any]", supplier)
        return None

    @staticmethod
    def _supplier_index(pool: Any, supplier_id: str) -> int:
        """Position of ``supplier_id`` in the pool (stable ordering)."""
        for index, supplier in enumerate(pool.data["suppliers"]):
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
        """A deterministic pre-quote bid derived from the supplier's profile."""
        supplier_id = str(supplier["supplier_id"])
        unit_price = floor_price(supplier)
        return BidPayload(
            session_id=session_id,
            supplier_id=supplier_id,
            unit_price=unit_price,
            lead_time_days=lead_time_days(supplier, target_lead, index),
            carbon_footprint_kg=carbon_footprint(supplier, quantity),
            reliability_score=float(supplier["reliability_score"]),
            bid_bond_amount=bid_bond_amount(unit_price, quantity),
            delivery_date=DEFAULT_DELIVERY_DATE,
            justification=f"Synthetic pre-quote profile for {supplier_id}",
        )
