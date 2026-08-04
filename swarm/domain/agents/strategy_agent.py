"""StrategyAgent — selects the execution strategy for a requirement.

Reacts to ``RequirementCreated`` and picks a :class:`Strategy` from the
requirement's constraints with a pure rule (:func:`select_strategy`): a strict
carbon constraint yields ``low_carbon``, a tight budget yields
``cost_optimized``, otherwise ``balanced``. Publishes a ``StrategySelected``
event and a :class:`StrategyArtifact` that the evaluation agent reads, so the
strategy artifact always exists before any supplier is evaluated.

v0.9 Step 3: the agent also reads any existing ``llm_completion`` artifact
(via :func:`swarm.utils.llm_reader.get_latest_llm_completion`) and attaches
its signals as advisory ``llm_context`` on the :class:`StrategyArtifact`.
This is non-authoritative: the strategy selection logic is unchanged — LLM
data is context only, never logic input.

v0.9 Step 4: the agent additionally reads ``suggested_adjustments`` from the
LLM completion, validates them through
:func:`swarm.utils.llm_validation.validate_strategy_adjustments`, and records
the bounded result as ``adjusted_weights`` + ``llm_influence`` on the
artifact. The base ``weights`` field remains the canonical strategy weights
unchanged, so downstream agents are unaffected. This creates a safe audit
trail of influence before any weight is ever honored.

v0.9 Step 5: the agent aggregates all ``llm_completion`` artifacts via
:func:`swarm.utils.llm_consensus.compute_llm_consensus` and only applies
adjustments when the consensus ``confidence`` exceeds
:data:`~swarm.utils.llm_consensus.CONFIDENCE_THRESHOLD` (0.7). Low-confidence
LLM outputs produce zero influence — the system requires multi-model agreement
before any suggestion is considered.

v0.9 Step 6: the agent records each consensus result as a replay-safe history
artifact, then computes **temporal stability** — how much the LLM's suggestions
drift across time. Trust is defined as ``confidence × stability``. Influence
is only applied when ``trust >= TRUST_THRESHOLD`` (0.7). This prevents the system
from over-trusting an LLM that produces high-confidence but volatile suggestions,
and avoids over-trusting early (single-data-point) consensus that has no
temporal track record.
"""

from typing import Any

import structlog

from swarm.core.agent import BaseAgent
from swarm.core.capability import Capability
from swarm.core.event import Event
from swarm.core.state import SwarmState
from swarm.domain.artifacts import (
    REQUIREMENT_ARTIFACT_NAME,
    STRATEGY_ARTIFACT_NAME,
    StrategyArtifact,
)
from swarm.domain.events import ProcurementEventType
from swarm.domain.strategy import Strategy, select_strategy
from swarm.utils.llm_consensus import compute_llm_consensus
from swarm.utils.llm_explainer import build_llm_explanation
from swarm.utils.llm_memory import get_llm_consensus_history, record_llm_consensus
from swarm.utils.llm_reader import get_all_llm_completions, get_latest_llm_completion
from swarm.utils.llm_stability import TRUST_THRESHOLD, compute_temporal_stability
from swarm.utils.llm_validation import validate_strategy_adjustments

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
        self._llm_context: dict[str, Any] | None = None
        self._validated_adjustments: dict[str, float] = {}
        self._raw_adjustments: dict[str, float] = {}
        self._adjusted_weights: dict[str, float] | None = None
        self._consensus: dict[str, Any] = {}
        self._stability: float = 0.0
        self._trust_score: float = 0.0
        self._history_depth: int = 0
        self._explanation: dict[str, Any] = {}
        self._pending = False
        self._is_re_evaluation = False
        self._strategy_selected_event_published = False

    async def perceive(self, event: Event) -> None:
        if event.replayed:
            return
        if event.type in (
            ProcurementEventType.REQUIREMENT_CREATED,
            ProcurementEventType.QUOTES_COMPLETED,
        ):
            self._pending = True
            self._correlation_id = event.correlation_id
            self._requirement_artifact = str(
                event.payload.get("artifact", REQUIREMENT_ARTIFACT_NAME)
            )
            self._is_re_evaluation = (
                event.type == ProcurementEventType.QUOTES_COMPLETED
            )
            self._strategy = None
            self._llm_context = None
            self._validated_adjustments = {}
            self._raw_adjustments = {}
            self._adjusted_weights = None
            self._consensus = {}
            self._stability = 0.0
            self._trust_score = 0.0
            self._history_depth = 0

    async def reason(self, state: SwarmState) -> None:
        if not self._pending:
            return
        requirement = state.get_artifact(self._requirement_artifact)
        if requirement is None:
            self._pending = False
            return
        constraints = requirement.data.get("constraints", {})
        self._strategy = select_strategy(constraints)

        # Advisory-only: read latest LLM completion output if one exists.
        # The strategy selection logic above is NOT affected by this.
        llm_output = get_latest_llm_completion(state, correlation_id=self._correlation_id)
        completions = get_all_llm_completions(state, correlation_id=self._correlation_id)

        # Compute consensus across all LLM completions (Step 5)
        self._consensus = compute_llm_consensus(completions)

        if llm_output is not None:
            llm_risks = llm_output.get("risks", [])
            risks = llm_risks if isinstance(llm_risks, list) else []
            llm_tradeoffs = llm_output.get("tradeoffs", [])
            tradeoffs = llm_tradeoffs if isinstance(llm_tradeoffs, list) else []
            self._llm_context = {
                "used": True,
                "risk_hints": risks[:3],
                "tradeoff_hints": tradeoffs[:3],
                "adjustments_applied": self._trust_score >= TRUST_THRESHOLD,
            }
        else:
            self._llm_context = {"used": False, "adjustments_applied": False}

        # Step 6: Record consensus history and compute temporal stability + trust.
        # Only record when there are actual LLM completions (non-empty consensus).
        if completions:
            existing_history = get_llm_consensus_history(
                state,
                correlation_id=self._correlation_id or "",
            )
            round_number = len(existing_history) + 1

            record_llm_consensus(
                state,
                correlation_id=self._correlation_id or "",
                consensus=self._consensus,
                round_number=round_number,
                parent_ids=[c.id for c in state.find_artifacts(
                    kind="llm",
                    correlation_id=self._correlation_id,
                )],
                by=self.name,
            )

            # Read full history (including the just-recorded consensus).
            history = get_llm_consensus_history(
                state,
                correlation_id=self._correlation_id or "",
            )
            self._history_depth = len(history)

            self._stability = compute_temporal_stability(history)
            confidence = self._consensus.get("confidence", 0.0)
            self._trust_score = round(confidence * self._stability, 4)
        else:
            self._history_depth = 0
            self._stability = 0.0
            self._trust_score = 0.0

         # Gate: only apply adjustments if trust >= threshold.
        # Trust = confidence × stability, so both dimensions must be high.
        # The aggregated adjustments from consensus go through validation again
        # (defense-in-depth) before being applied.
        if self._trust_score >= TRUST_THRESHOLD:
            aggregated = self._consensus.get("aggregated_adjustments", {})
            self._raw_adjustments = aggregated
            self._validated_adjustments = validate_strategy_adjustments(
                aggregated
            )
        else:
            self._raw_adjustments = {}
            self._validated_adjustments = {}

        # Apply bounded adjustments to the strategy weights.
        self._adjusted_weights = self._apply_adjustments()

        # Deterministic explainability: build an explanation dict that records
        # the decision rationale without affecting any logic.
        self._explanation = build_llm_explanation(
            confidence=self._consensus.get("confidence", 0.0),
            stability=self._stability,
            trust=self._trust_score,
            threshold=TRUST_THRESHOLD,
            adjustments=self._validated_adjustments,
        )

        logger.info(
            "strategy_selected",
            agent=self.name,
            strategy_name=self._strategy.name,
            llm_context_used=bool(llm_output),
            llm_confidence=self._consensus.get("confidence", 0.0),
            llm_stability=self._stability,
            llm_trust_score=self._trust_score,
            adjustments_applied=bool(self._validated_adjustments),
            correlation_id=self._correlation_id,
        )

    def _apply_adjustments(self) -> dict[str, float]:
        """Apply bounded LLM adjustments to the base strategy weights.

        Maps ``delivery_weight_delta`` to ``score_weight`` (delivery time is
        a component of the evaluation score). All weights are clamped to
        [0, 1] and then normalized to sum to exactly 1.0.

        The strategy NAME is never affected — only the numeric weights
        carried on the artifact for consumers.
        """
        base = self._strategy.as_weights() if self._strategy is not None else {}
        if not self._validated_adjustments or not base:
            return base

        price = base["price_weight"] + self._validated_adjustments.get(
            "price_weight_delta", 0.0
        )
        score = base["score_weight"] + self._validated_adjustments.get(
            "delivery_weight_delta", 0.0
        )
        carbon = base["carbon_weight"]

        # Clamp each weight to [0, 1] (hard constraint: never negative)
        price = max(0.0, min(1.0, price))
        score = max(0.0, min(1.0, score))
        carbon = max(0.0, min(1.0, carbon))

        # Normalize so weights sum to exactly 1.0
        total = price + score + carbon
        if total <= 0.0:
            return dict(base)
        return {
            "price_weight": price / total,
            "score_weight": score / total,
            "carbon_weight": carbon / total,
        }

    async def act(self, state: SwarmState) -> None:
        if not self._pending or self._strategy is None:
            return
        base_weights = self._strategy.as_weights()
        artifact_data = {
            "strategy_name": self._strategy.name,
            "description": self._strategy.description,
            "weights": base_weights,
            "adjusted_weights": self._adjusted_weights or base_weights,
            "llm_context": self._llm_context or {"used": False},
            "llm_influence": {
                "used": bool(self._validated_adjustments),
                "raw_adjustments": self._raw_adjustments,
                "validated_adjustments": self._validated_adjustments,
                "adjustments_applied": bool(self._validated_adjustments),
                "llm_consensus": self._consensus,
            },
            "llm_trust": {
                "confidence": self._consensus.get("confidence", 0.0),
                "stability": self._stability,
                "trust_score": self._trust_score,
                "history_depth": self._history_depth,
            },
            "llm_explanation": self._explanation,
        }

        # On re-evaluation (QuotesCompleted), update the existing strategy
        # artifact instead of creating a new one and publishing again.
        if self._is_re_evaluation:
            existing = state.get_artifact(STRATEGY_ARTIFACT_NAME)
            if existing is not None:
                artifact = existing.update(artifact_data, by=self.name)
            else:
                artifact = StrategyArtifact(
                    data=artifact_data,
                    parent_ids=[self._requirement_artifact],
                    created_by=self.name,
                    correlation_id=self._correlation_id,
                )
        else:
            artifact = StrategyArtifact(
                data=artifact_data,
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

        # Only publish StrategySelected on the initial selection (RequirementCreated),
        # not on re-evaluations (QuotesCompleted), to avoid triggering duplicate
        # supplier discovery.
        if not self._is_re_evaluation:
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
        self._is_re_evaluation = False
        self._strategy = None
        self._llm_context = None
        self._validated_adjustments = {}
        self._raw_adjustments = {}
        self._adjusted_weights = None
        self._consensus = {}
        self._stability = 0.0
        self._trust_score = 0.0
        self._history_depth = 0
        self._explanation = {}
