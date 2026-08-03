"""Procurement domain layer for the Phase 2 adapter swarm.

Contains the domain artifacts, event types and the deterministic agents
(requirement, strategy, supplier discovery, evaluation, negotiation, decision)
that wrap the existing ``core/`` procurement logic. The agents communicate
exclusively through the swarm runtime (``Message → Event → Artifact``) and
never call each other directly.
"""

from swarm.domain.agents import (
    DecisionAgent,
    EvaluationAgent,
    NegotiationAgent,
    OutcomeAgent,
    RequirementAgent,
    StrategyAgent,
    SupplierDiscoveryAgent,
    SupplierIntelligenceAgent,
)
from swarm.domain.artifacts import (
    DECISION_ARTIFACT_NAME,
    DECISION_EXPLANATION_ARTIFACT_NAME,
    OUTCOME_ARTIFACT_NAME,
    REQUIREMENT_ARTIFACT_NAME,
    STRATEGY_ARTIFACT_NAME,
    SUPPLIER_LIST_ARTIFACT_NAME,
    SUPPLIER_PERFORMANCE_ARTIFACT_NAME,
    DecisionArtifact,
    DecisionExplanationArtifact,
    EvaluationArtifact,
    OutcomeArtifact,
    QuoteArtifact,
    RequirementArtifact,
    StrategyArtifact,
    SupplierListArtifact,
    SupplierPerformanceArtifact,
    evaluation_artifact_name,
    quote_artifact_name,
)
from swarm.domain.events import (
    CREATE_REQUIREMENT_INTENT,
    RECORD_OUTCOME_INTENT,
    ProcurementEventType,
)
from swarm.domain.pricing import (
    DEFAULT_BID_BOND_PCT,
    DEFAULT_PAYMENT_TERMS,
    bid_bond_amount,
    carbon_footprint,
    floor_price,
    lead_time_days,
)
from swarm.domain.strategy import (
    BALANCED_STRATEGY,
    DEFAULT_STRATEGIES,
    Strategy,
    select_strategy,
)

__all__ = [
    "BALANCED_STRATEGY",
    "CREATE_REQUIREMENT_INTENT",
    "DECISION_ARTIFACT_NAME",
    "DECISION_EXPLANATION_ARTIFACT_NAME",
    "DEFAULT_BID_BOND_PCT",
    "DEFAULT_PAYMENT_TERMS",
    "DEFAULT_STRATEGIES",
    "DecisionAgent",
    "DecisionArtifact",
    "DecisionExplanationArtifact",
    "EvaluationAgent",
    "EvaluationArtifact",
    "NegotiationAgent",
    "OUTCOME_ARTIFACT_NAME",
    "OutcomeAgent",
    "OutcomeArtifact",
    "ProcurementEventType",
    "QuoteArtifact",
    "RECORD_OUTCOME_INTENT",
    "RECORD_OUTCOME_INTENT",
    "REQUIREMENT_ARTIFACT_NAME",
    "RequirementAgent",
    "RequirementArtifact",
    "STRATEGY_ARTIFACT_NAME",
    "SupplierDiscoveryAgent",
    "SupplierIntelligenceAgent",
    "SupplierListArtifact",
    "SUPPLIER_LIST_ARTIFACT_NAME",
    "SUPPLIER_PERFORMANCE_ARTIFACT_NAME",
    "SupplierPerformanceArtifact",
    "Strategy",
    "StrategyAgent",
    "StrategyArtifact",
    "bid_bond_amount",
    "carbon_footprint",
    "evaluation_artifact_name",
    "floor_price",
    "lead_time_days",
    "quote_artifact_name",
    "select_strategy",
]
