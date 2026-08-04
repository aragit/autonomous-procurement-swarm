"""SupplierAnalysisLLMAgent — a read-only LLM analysis agent (v0.9 Step 2).

This agent performs supplier comparison analysis via an LLM, but is strictly
bounded:

- **No side effects**: it makes zero connector calls and triggers no
  downstream agent.
- **No authority**: it does not mutate decisions, governance, or execution.
- **No events published**: it is purely observational.
- **Replay-safe**: identical inputs produce identical LLMArtifacts (deduped
  via :func:`compute_llm_input_hash`).

It is triggered on ``QuotesCompleted`` — the point at which all supplier
responses (quotes + evaluations) have been collected — and records two
:class:`LLMArtifact` objects: one for the prompt (``llm_prompt``) and one for
the completion (``llm_completion``), both appearing under the ``cognitive``
phase in the timeline.

The LLM call itself is stubbed for now; the agent records the analysis result
deterministically so the timeline always shows the cognitive layer without
trusting or executing against it.
"""

from __future__ import annotations

from typing import Any

import structlog

from swarm.core.agent import BaseAgent
from swarm.core.capability import Capability
from swarm.core.event import Event
from swarm.core.state import SwarmState
from swarm.domain.events import ProcurementEventType
from swarm.utils.llm_hash import record_llm_artifact

logger = structlog.get_logger(__name__)

SUPPLIER_ANALYSIS_PROMPT = """
You are a procurement analysis assistant. Compare the suppliers based on
price, delivery time, reliability, and carbon footprint. Return your analysis
as a structured JSON object with:
- "summary": a brief textual summary of the supplier comparison
- "risks": a list of risk factors per supplier
- "tradeoffs": a list of key tradeoffs between suppliers

Do NOT make a final sourcing decision. Your output is advisory and
observational only.
""".strip()


#: Number of independent "model" completions to produce per analysis.
#: Each variant is deterministic and shares the same input_hash, enabling
#: multi-LLM consensus in Step 5.
NUM_COMPLETION_VARIANTS = 3

#: Deterministic adjustment offsets per variant (±0.01 around the base).
_VARIANT_OFFSETS: dict[int, tuple[float, float]] = {
    0: (-0.01, 0.01),
    1: (0.00, 0.00),
    2: (+0.01, -0.01),
}


class SupplierAnalysisLLMAgent(BaseAgent):
    """A read-only LLM agent that performs supplier comparison analysis.

    Subscribes to ``QuotesCompleted`` and records :class:`LLMArtifact` objects
    (prompt + completion) in shared state. Produces **no** domain events and
    invokes **no** external systems — it is purely part of the cognitive
    layer's audit trail for future Step 3 consumption.
    """

    name = "supplier_analysis_llm_agent"
    description = (
        "Performs read-only supplier comparison analysis via LLM; "
        "records LLMArtifacts without side effects or authority"
    )
    capabilities = [
        Capability(
            name="supplier.analyze",
            description="LLM-powered supplier comparison analysis (read-only)",
        )
    ]

    def __init__(self) -> None:
        super().__init__()
        self._correlation_id: str | None = None
        self._pending = False
        self._analyzed_for: set[str] = set()

    async def perceive(self, event: Event) -> None:
        if event.replayed:
            return
        if event.type == ProcurementEventType.QUOTES_COMPLETED:
            self._correlation_id = event.correlation_id
            if self._correlation_id is not None and self._correlation_id in self._analyzed_for:
                self._pending = False
                return
            self._pending = True

    async def reason(self, state: SwarmState) -> None:
        if not self._pending or self._correlation_id is None:
            return

        requirement_name = "requirement"
        requirement = state.get_artifact(requirement_name)
        if requirement is None:
            self._pending = False
            return

        quotes = state.find_artifacts(kind="quote", correlation_id=self._correlation_id)
        if not quotes:
            self._pending = False
            return

        constraints = requirement.data.get("constraints", {})
        self._input_payload: dict[str, Any] = self._build_input_payload(constraints, quotes)

    async def act(self, state: SwarmState) -> None:
        if not self._pending or self._correlation_id is None:
            return

        prompt_artifact = record_llm_artifact(
            state,
            model="stub",
            prompt=SUPPLIER_ANALYSIS_PROMPT,
            parameters={"payload": self._input_payload},
            kind="llm_prompt",
            correlation_id=self._correlation_id,
            by=self.name,
        )

        parent_id = prompt_artifact.id
        for variant in range(NUM_COMPLETION_VARIANTS):
            llm_output = self._call_llm_stub(
                SUPPLIER_ANALYSIS_PROMPT, self._input_payload, variant
            )
            record_llm_artifact(
                state,
                model="stub",
                prompt=SUPPLIER_ANALYSIS_PROMPT,
                parameters={"payload": self._input_payload},
                output=llm_output,
                kind="llm_completion",
                variant=variant,
                parent_ids=[parent_id],
                correlation_id=self._correlation_id,
                by=self.name,
            )

        if self._correlation_id is not None:
            self._analyzed_for.add(self._correlation_id)
        self._pending = False
        self._input_payload = {}

    @staticmethod
    def _build_input_payload(
        constraints: dict[str, Any],
        quotes: list[Any],
    ) -> dict[str, Any]:
        """Build a deterministic, minimal input payload from quotes + requirement.

        No timestamps, no randomness — suppliers are sorted by ``supplier_id``
        for stable hashing and reproducible LLMArtifact dedup.
        """
        suppliers = sorted(
            (
                {
                    "supplier_id": str(quote.data.get("supplier_id", "")),
                    "price": quote.data.get("price"),
                    "terms": quote.data.get("terms"),
                    "metadata": dict(quote.data.get("metadata", {})),
                }
                for quote in quotes
            ),
            key=lambda s: s["supplier_id"],
        )
        return {
            "requirement": {
                "material": constraints.get("material"),
                "quantity": int(constraints.get("quantity") or 0),
                "target_lead_time_days": int(
                    constraints.get("target_lead_time_days") or 0
                ),
                "budget": float(constraints.get("budget") or 0.0),
                "max_unit_price": constraints.get("max_unit_price"),
            },
            "suppliers": suppliers,
        }

    @staticmethod
    def _call_llm_stub(
        prompt: str,
        payload: dict[str, Any],
        variant: int = 0,
    ) -> dict[str, Any]:
        """Stubbed LLM invocation — returns a deterministic analysis.

        This is a placeholder. When the real LLM integration lands in a later
        step, this method will be replaced with the actual call. The return
        shape is stable: a dict with ``summary``, ``risks``,
        ``tradeoffs``, and optionally ``suggested_adjustments`` keys.

        ``variant`` introduces deterministic, bounded variation in the suggested
        adjustments so multi-LLM consensus can detect agreement vs. noise.
        """
        supplier_ids = [s["supplier_id"] for s in payload.get("suppliers", [])]
        price_delta, delivery_delta = _VARIANT_OFFSETS.get(variant, (0.0, 0.0))
        return {
            "summary": f"Analyzed {len(supplier_ids)} suppliers: {', '.join(supplier_ids)[:100]}",
            "risks": [
                {
                    "supplier_id": sid,
                    "risk_factors": ["price_variance", "delivery_reliability"],
                }
                for sid in supplier_ids
            ],
            "tradeoffs": [
                "Lower price may imply higher delivery risk",
                "Higher reliability suppliers charge premium pricing",
            ],
            "suggested_adjustments": {
                "price_weight_delta": -0.05 + price_delta,
                "delivery_weight_delta": 0.05 + delivery_delta,
            },
        }
