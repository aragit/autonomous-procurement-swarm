"""Procurement domain layer for the Phase 2 adapter swarm.

Contains the domain artifacts, event types and the five deterministic agents
(requirement, supplier discovery, evaluation, negotiation, decision) that wrap
the existing ``core/`` procurement logic. The agents communicate exclusively
through the swarm runtime (``Message → Event → Artifact``) and never call each
other directly.
"""

from swarm.domain.agents import (
    DecisionAgent,
    EvaluationAgent,
    NegotiationAgent,
    RequirementAgent,
    SupplierDiscoveryAgent,
)
from swarm.domain.artifacts import (
    DECISION_ARTIFACT_NAME,
    REQUIREMENT_ARTIFACT_NAME,
    SUPPLIER_LIST_ARTIFACT_NAME,
    DecisionArtifact,
    EvaluationArtifact,
    QuoteArtifact,
    RequirementArtifact,
    SupplierListArtifact,
    evaluation_artifact_name,
    quote_artifact_name,
)
from swarm.domain.events import (
    CREATE_REQUIREMENT_INTENT,
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

__all__ = [
    "CREATE_REQUIREMENT_INTENT",
    "DECISION_ARTIFACT_NAME",
    "DEFAULT_BID_BOND_PCT",
    "DEFAULT_PAYMENT_TERMS",
    "DecisionAgent",
    "DecisionArtifact",
    "EvaluationAgent",
    "EvaluationArtifact",
    "NegotiationAgent",
    "ProcurementEventType",
    "QuoteArtifact",
    "REQUIREMENT_ARTIFACT_NAME",
    "RequirementAgent",
    "RequirementArtifact",
    "SUPPLIER_LIST_ARTIFACT_NAME",
    "SupplierDiscoveryAgent",
    "SupplierListArtifact",
    "bid_bond_amount",
    "carbon_footprint",
    "evaluation_artifact_name",
    "floor_price",
    "lead_time_days",
    "quote_artifact_name",
]
