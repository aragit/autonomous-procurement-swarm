"""Domain artifacts shared between the Phase 2 procurement agents.

Each artifact subclasses the runtime :class:`Artifact` model so it stays
serializable and queryable through :class:`SwarmState`. The ``data`` contract
of each is documented on the class: downstream agents read artifacts by their
``kind``/``name``/``tags``, never by peeking at another agent's internals.
"""

from typing import Literal

from swarm.core.artifact import Artifact

REQUIREMENT_ARTIFACT_NAME = "requirement"
SUPPLIER_LIST_ARTIFACT_NAME = "suppliers"
STRATEGY_ARTIFACT_NAME = "strategy"
DECISION_ARTIFACT_NAME = "decision"
DECISION_EXPLANATION_ARTIFACT_NAME = "decision_explanation"
OUTCOME_ARTIFACT_NAME = "outcome"
SUPPLIER_PERFORMANCE_ARTIFACT_NAME = "supplier_performance"


def evaluation_artifact_name(supplier_id: str) -> str:
    """Stable artifact name for one supplier's evaluation."""
    return f"evaluation_{supplier_id}"


def quote_artifact_name(supplier_id: str) -> str:
    """Stable artifact name for one supplier's quote."""
    return f"quote_{supplier_id}"


class RequirementArtifact(Artifact):
    """A parsed procurement requirement.

    ``data`` contract::

        {
            "text": str,                       # original free-text request
            "constraints": {                   # structured, reusable spec
                "material": str,
                "quantity": int,
                "max_unit_price": float | None,
                "target_lead_time_days": int,
                "budget": float,
                "max_carbon_per_unit": float | None,  # strict carbon constraint
            },
            "metadata": {                      # raw input + RFQ normalization
                "raw": dict,
                "rfq": dict | None,            # core RFQPayload dump when valid
            },
        }
    """

    kind: Literal["requirement"] = "requirement"
    name: str = REQUIREMENT_ARTIFACT_NAME


class StrategyArtifact(Artifact):
    """The execution strategy selected for a requirement.

    ``data`` contract::

        {
            "strategy_name": str,              # one of the DEFAULT_STRATEGIES
            "description": str,
            "weights": {                       # sums to 1.0
                "price_weight": float,
                "score_weight": float,
                "carbon_weight": float,
            },
        }
    """

    kind: Literal["strategy"] = "strategy"
    name: str = STRATEGY_ARTIFACT_NAME


class SupplierListArtifact(Artifact):
    """The discovered supplier pool for a requirement.

    ``data`` contract::

        {
            "material": str,
            "quantity": int,
            "target_lead_time_days": int,
            "spot_price": float,               # deterministic market reference
            "suppliers": [                     # one entry per candidate, each a
                {                              # CostModel-shaped profile
                    "supplier_id": str,
                    "base_cost_per_unit": float,
                    "logistics_premium_per_unit": float,
                    "capacity_units": int,
                    "current_utilization": float,
                    "min_margin_pct": float,
                    "reliability_score": float,
                    "esg_carbon_per_unit": float,
                },
                ...
            ],
        }
    """

    kind: Literal["supplier_list"] = "supplier_list"
    name: str = SUPPLIER_LIST_ARTIFACT_NAME


class QuoteArtifact(Artifact):
    """A deterministic quote from one supplier.

    ``data`` contract::

        {
            "supplier_id": str,
            "price": float,                    # deterministic ask price
            "terms": str,                      # e.g. "net_30"
            "metadata": {
                "quantity": int,
                "lead_time_days": int,
                "carbon_footprint_kg": float,
                "reliability_score": float,
            },
        }

    ``tags``: ``{"supplier": <supplier_id>}``.
    """

    kind: Literal["quote"] = "quote"
    name: str


class EvaluationArtifact(Artifact):
    """The multi-criteria score of one supplier.

    ``data`` contract::

        {
            "supplier_id": str,
            "score": float,                    # composite score in [0, 1]
            "breakdown": {                     # weighted sub-scores
                "price": float,
                "lead_time": float,
                "esg": float,
                "reliability": float,
            },
            "bid": dict,                       # core BidPayload that was scored
        }

    ``tags``: ``{"supplier": <supplier_id>}``.
    """

    kind: Literal["evaluation"] = "evaluation"
    name: str


class DecisionArtifact(Artifact):
    """The final supplier selection.

    ``data`` contract::

        {
            "selected_supplier": str | None,
            "reasoning": {
                "criteria": str,
                "ranked": [                    # best first
                    {"supplier_id", "score", "price", "policy_passed",
                     "policy_reason"},
                    ...
                ],
            },
        }
    """

    kind: Literal["decision"] = "decision"
    name: str = DECISION_ARTIFACT_NAME


class DecisionExplanationArtifact(Artifact):
    """A human-readable explanation of the final supplier selection.

    Produced by the decision agent immediately after the decision so the "why"
    of a selection is auditable. ``data`` contract::

        {
            "selected_supplier": str | None,
            "strategy_used": str,              # name of the execution strategy
            "top_factors": [                   # why the winner was chosen
                "Highest composite score ...",
                ...
            ],
            "rejected_suppliers": [            # every non-selected supplier
                {
                    "supplier_id": str,
                    "score": float,
                    "price": float,
                    "policy_passed": bool,
                    "policy_reason": str,
                    "reason": str,             # why it lost / was rejected
                },
                ...
            ],
        }
    """

    kind: Literal["decision_explanation"] = "decision_explanation"
    name: str = DECISION_EXPLANATION_ARTIFACT_NAME


def outcome_artifact_name(decision_id: str) -> str:
    """Stable artifact name for an outcome tied to a decision."""
    return f"outcome_{decision_id}"


def supplier_performance_artifact_name(supplier_id: str) -> str:
    """Stable artifact name for one supplier's performance record."""
    return f"performance_{supplier_id}"


class OutcomeArtifact(Artifact):
    """A post-decision procurement outcome recorded after delivery.

    Published by :class:`OutcomeAgent` and consumed by
    :class:`SupplierIntelligenceAgent` to update supplier memory. Its
    ``parent_ids`` reference the originating :class:`DecisionArtifact` (by id),
    preserving lineage through the feedback loop.

    ``data`` contract::

        {
            "supplier_id": str,
            "decision_id": str,            # DecisionArtifact.id (uuid)
            "delivered_on_time": bool,
            "quality_score": float,        # 0..1
            "actual_price": float,         # per-unit, currency
            "carbon_score": float,          # per-unit kg CO₂
        }
    """

    kind: Literal["procurement_outcome"] = "procurement_outcome"
    name: str = OUTCOME_ARTIFACT_NAME


class SupplierPerformanceArtifact(Artifact):
    """A supplier's cumulative performance record as a first-class artifact.

    Produced by :class:`SupplierIntelligenceAgent` from
    :class:`OutcomeArtifact`s and read by :class:`EvaluationAgent` to apply a
    deterministic history adjustment. Its ``parent_ids`` reference the
    :class:`OutcomeArtifact` that produced it.

    ``data`` contract::

        {
            "supplier_id": str,
            "performance_metrics": {         # running averages
                "total_orders": int,
                "successful_orders": int,
                "average_delivery_score": float,
                "average_quality_score": float,
                "average_price_competitiveness": float,
                "average_carbon_score": float,
            },
            "order_count": int,
            "updated_at": str,               # ISO-8601 UTC
        }
    """

    kind: Literal["supplier_performance"] = "supplier_performance"
    name: str = SUPPLIER_PERFORMANCE_ARTIFACT_NAME
