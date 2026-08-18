"""NegotiatorActor — generates quotes and writes to DEAL channel.

Wraps the deterministic NegotiationAgent logic.
"""

from __future__ import annotations

from typing import Any, cast

import ray
import structlog

from mesh.actors.base import MeshActor
from mesh.blackboard import DistributedBlackboard
from mesh.channels import ChannelType
from mesh.neuro import (
    BanditContext,
    LLMConfig,
    NegotiationStrategy,
    NegotiatorProposal,
    NeuroSymbolicBridge,
    OpenAICompatibleBackend,
    compute_reward,
    get_persistent_bandit,
)
from mesh.neuro.backend import StructuredBackend
from mesh.neuro.types import NeuralProposal, SymbolicVerdict
from swarm.domain.pricing import (
    DEFAULT_PAYMENT_TERMS,
    carbon_footprint,
    floor_price,
    lead_time_days,
)

logger = structlog.get_logger(__name__)


@ray.remote(max_restarts=3, max_task_retries=3)
class NegotiatorActor(MeshActor):
    """Generates deterministic quotes for evaluated suppliers.

    Reads from SCORE channel (and RISK for context), generates quotes,
    writes to DEAL channel.

    When a ``neuro_bridge`` (or ``llm_config``) is supplied the actor uses
    schema-constrained LLM quote generation validated by the SafetyKernelActor
    with an auto-correction retry loop.  If the LLM path is exhausted or
    unavailable it transparently falls back to the deterministic
    :meth:`_generate_quote`.
    """

    def __init__(
        self,
        actor_id: str,
        blackboard: ray.actor.ActorHandle,
        kernel: ray.actor.ActorHandle | None = None,
        neuro_bridge: NeuroSymbolicBridge | None = None,
        llm_config: LLMConfig | None = None,
        neuro_max_retries: int = 3,
    ) -> None:
        super().__init__(actor_id, "negotiator", blackboard, kernel)
        self._processed_suppliers: set[tuple[str, str]] = set()  # (correlation_id, supplier_id)
        self._neuro_bridge = neuro_bridge or self._build_neuro_bridge(llm_config, neuro_max_retries)
        self._bandit = get_persistent_bandit()
        self._current_strategy: NegotiationStrategy | None = None
        self._current_context: Any | None = None
        self._negotiation_rounds = 0

    def _build_neuro_bridge(
        self,
        llm_config: LLMConfig | None,
        max_retries: int,
    ) -> NeuroSymbolicBridge | None:
        """Construct the neuro bridge from an LLM config, if provided."""
        if llm_config is None:
            return None

        async def _kernel_validate(proposal: NeuralProposal) -> SymbolicVerdict:
            return await self.kernel.validate.remote(proposal)  # type: ignore[no-any-return]

        backend: StructuredBackend = OpenAICompatibleBackend(llm_config)
        return NeuroSymbolicBridge(
            backend=backend,
            validator=_kernel_validate,
            max_retries=max_retries,
            raise_on_exhaustion=False,
        )

    async def perceive(self, blackboard: DistributedBlackboard) -> dict[str, Any]:
        """Read SCORE channel for evaluated suppliers."""
        score_traces = await self.read_channel(ChannelType.SCORE, limit=50)
        return {"evaluations": score_traces}

    async def reason(self, perception: dict[str, Any]) -> dict[str, Any]:
        """Process evaluations and generate quotes."""
        evaluations = perception.get("evaluations", [])
        proposals = []

        for trace in evaluations:
            payload = trace.get("payload", {})
            if payload.get("type") != "evaluation":
                continue

            correlation_id = trace.get("correlation_id", "")
            supplier_id = str(payload.get("supplier_id") or "")
            trace_id = trace.get("id", "")

            if not supplier_id or (correlation_id, supplier_id) in self._processed_suppliers:
                continue

            # Need pool data from DISCOVERY channel
            disc_traces = await self.read_channel(ChannelType.DISCOVERY, limit=20)
            pool_trace = next(
                (
                    t
                    for t in disc_traces
                    if t.get("correlation_id") == correlation_id
                    and t.get("payload", {}).get("type") == "supplier_list"
                ),
                None,
            )
            if not pool_trace:
                logger.warning(
                    "negotiator_pool_not_found",
                    actor_id=self.actor_id,
                    correlation_id=correlation_id,
                    supplier_id=supplier_id,
                )
                continue

            pool_data = pool_trace.get("payload", {}).get("data", {})
            supplier = self._find_supplier(pool_data, supplier_id)
            if not supplier:
                continue

            # Requirement for constraints
            req_traces = await self.read_channel(ChannelType.REQUIREMENT, limit=5)
            req_trace = next(
                (t for t in req_traces if t.get("correlation_id") == correlation_id),
                None,
            )
            requirement_data = (
                req_trace.get("payload", {}).get("requirement", {}) if req_trace else {}
            )
            budget = requirement_data.get("constraints", {}).get("budget")

            # Generate quote (neuro path with deterministic fallback)
            quote = await self._neuro_quote(
                supplier=supplier,
                pool_data=pool_data,
                requirement_data=requirement_data,
                correlation_id=correlation_id,
                budget=budget,
            )
            if quote is None:
                quote = self._generate_quote(
                    supplier=supplier,
                    pool_data=pool_data,
                    requirement_data=requirement_data,
                    correlation_id=correlation_id,
                )

            proposals.append(
                {
                    "correlation_id": correlation_id,
                    "supplier_id": supplier_id,
                    "eval_trace_id": trace_id,
                    "pool_trace_id": pool_trace.get("id", ""),
                    "quote": quote,
                    "confidence": 1.0,
                }
            )
            self._processed_suppliers.add((correlation_id, supplier_id))

        return {"proposals": proposals}

    async def act(self, blackboard: DistributedBlackboard, proposal: dict[str, Any]) -> None:
        """Write quote to DEAL channel."""
        for prop in proposal.get("proposals", []):
            correlation_id = prop["correlation_id"]
            supplier_id = prop["supplier_id"]
            eval_trace_id = prop["eval_trace_id"]
            pool_trace_id = prop["pool_trace_id"]

            trace_id = await self.write_channel(
                ChannelType.DEAL,
                {
                    "type": "quote",
                    "data": prop["quote"],
                    "correlation_id": correlation_id,
                    "supplier_id": supplier_id,
                },
                parent_ids=[eval_trace_id, pool_trace_id],
            )

            logger.info(
                "negotiator_quoted",
                actor_id=self.actor_id,
                correlation_id=correlation_id,
                supplier_id=supplier_id,
                price=prop["quote"]["price"],
                trace_id=trace_id,
            )

    def _build_bandit_context(
        self,
        supplier: dict[str, Any],
        pool_data: dict[str, Any],
        requirement_data: dict[str, Any],
    ) -> BanditContext:
        """Build context vector for bandit from current negotiation state."""
        return BanditContext.from_requirement(
            requirement=requirement_data,
            supplier=supplier,
            pool_data=pool_data,
            round_num=self._negotiation_rounds,
        )

    def _apply_strategy_hints(
        self,
        strategy: NegotiationStrategy,
        messages: list[dict[str, str]],
        supplier: dict[str, Any],
        pool_data: dict[str, Any],
        requirement_data: dict[str, Any],
        budget: float | None,
    ) -> list[dict[str, str]]:
        """Apply strategy-specific hints to the prompt messages."""
        from mesh.neuro import DEFAULT_STRATEGIES
        config = DEFAULT_STRATEGIES[strategy]

        hints = config.to_prompt_hints()

        # Modify the user message with strategy hints
        strategy_guidance = (
            f"\n\nNEGOTIATION STRATEGY: {strategy.value}\n"
            f"- Anchor multiplier: {hints['anchor_multiplier']:.2f}\n"
            f"- Concession rate: {hints['concession_rate']:.2f} per round\n"
            f"- Preferred payment terms: {', '.join(hints['preferred_payment_terms'])}\n"
            f"- Risk tolerance: {hints['risk_tolerance']:.2f}\n"
            f"- Relationship weight: {hints['relationship_weight']:.2f}\n"
        )

        enhanced_messages = [msg.copy() for msg in messages]
        enhanced_messages[-1]["content"] += strategy_guidance
        return enhanced_messages

    async def _neuro_quote(
        self,
        supplier: dict[str, Any],
        pool_data: dict[str, Any],
        requirement_data: dict[str, Any],
        correlation_id: str,
        budget: float | None,
    ) -> dict[str, Any] | None:
        """Generate a quote via structured LLM generation + kernel retry loop.

        Returns the validated quote dict (matching :meth:`_generate_quote`'s
        shape), or ``None`` if the neuro path is unavailable/exhausted so the
        caller falls back to the deterministic quote.
        """
        bridge = self._neuro_bridge
        if bridge is None:
            return None

        # Build context and select strategy via bandit
        context = self._build_bandit_context(supplier, pool_data, requirement_data)
        strategy = self._bandit.select_action(context)
        self._current_strategy = strategy
        self._current_context = context

        material = str(pool_data.get("material") or "steel")
        quantity = int(pool_data.get("quantity") or 1000)
        supplier_id = str(supplier["supplier_id"])
        spot_price = float(pool_data.get("spot_price") or 0.0)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a procurement supplier-quote agent. Generate a single "
                    "structured JSON quote for the named supplier. The quote price "
                    "must respect the budget ceiling (price * quantity <= budget "
                    "when provided), use net_30/net_60/cod/letter_of_credit terms, "
                    "and return ONLY valid JSON conforming to the NegotiatorQuote schema."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Quote for supplier {supplier_id} on material={material}, "
                    f"quantity={quantity}, spot_price={spot_price}, budget={budget}. "
                    "Provide price, payment terms and a metadata block with "
                    "quantity, lead_time_days (1-365), carbon_footprint_kg and "
                    "reliability_score (0-1)."
                ),
            },
        ]

        # Apply strategy hints to prompt
        messages = self._apply_strategy_hints(
            strategy, messages, supplier, pool_data, requirement_data, budget
        )

        def _payload_builder(model: NegotiatorProposal) -> dict[str, Any]:
            return model.to_kernel_payload(material=material, quantity=quantity, budget=budget)

        try:
            result = await bridge.safe_propose(
                archetype="negotiator",
                response_model=NegotiatorProposal,
                messages=messages,
                payload_builder=_payload_builder,
                correlation_id=correlation_id,
                confidence=1.0,
            )
        except Exception as exc:
            logger.warning(
                "negotiator_neuro_generation_error",
                actor_id=self.actor_id,
                supplier_id=supplier_id,
                error=str(exc),
            )
            return None

        if not result.verdict.approved:
            logger.warning(
                "negotiator_neuro_exhausted",
                actor_id=self.actor_id,
                correlation_id=correlation_id,
                supplier_id=supplier_id,
                reason=result.verdict.reason,
                attempts=result.attempts,
            )
            return None

        proposal: NegotiatorProposal = result.model_instance  # type: ignore[assignment]
        return proposal.quote.model_dump()

    def update_bandit_from_decision(
        self,
        decision: dict[str, Any],
        quote: dict[str, Any],
        requirement_data: dict[str, Any],
    ) -> None:
        """Update bandit with reward from completed negotiation.

        Called when a deal is awarded (BuyerActor writes DECISION).
        """
        if self._current_strategy is None or self._current_context is None:
            logger.debug(
                "bandit_update_skipped_no_context",
                actor_id=self.actor_id,
            )
            return

        reward = compute_reward(
            decision=decision,
            quote=quote,
            requirement=requirement_data,
            negotiation_rounds=self._negotiation_rounds,
        )

        self._bandit.update(self._current_strategy, self._current_context, reward)

        logger.info(
            "bandit_updated",
            actor_id=self.actor_id,
            strategy=self._current_strategy.value,
            reward=reward,
            action_counts=self._bandit.action_counts[self._current_strategy],
        )

        # Reset for next negotiation
        self._current_strategy = None
        self._current_context = None
        self._negotiation_rounds = 0

    def increment_round(self) -> None:
        """Increment negotiation round counter."""
        self._negotiation_rounds += 1

    @staticmethod
    def _generate_quote(
        supplier: dict[str, Any],
        pool_data: dict[str, Any],
        requirement_data: dict[str, Any],
        correlation_id: str,
    ) -> dict[str, Any]:
        """Generate deterministic quote from supplier profile."""
        constraints = requirement_data.get("constraints", {})
        quantity = int(pool_data.get("quantity") or constraints.get("quantity") or 1000)
        target_lead = int(
            pool_data.get("target_lead_time_days") or constraints.get("target_lead_time_days") or 30
        )
        index = 0  # Supplier index in pool (for deterministic lead time)

        return {
            "supplier_id": str(supplier["supplier_id"]),
            "price": floor_price(supplier),
            "terms": DEFAULT_PAYMENT_TERMS,
            "metadata": {
                "quantity": quantity,
                "lead_time_days": lead_time_days(supplier, target_lead, index),
                "carbon_footprint_kg": carbon_footprint(supplier, quantity),
                "reliability_score": supplier["reliability_score"],
            },
        }

    @staticmethod
    def _find_supplier(pool: dict[str, Any], supplier_id: str) -> dict[str, Any] | None:
        for supplier in pool.get("suppliers", []):
            if str(supplier["supplier_id"]) == supplier_id:
                return cast("dict[str, Any]", supplier)
        return None
