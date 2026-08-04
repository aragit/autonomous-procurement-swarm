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
RISK_ASSESSMENT_ARTIFACT_NAME = "risk_assessment"
GOVERNANCE_DECISION_ARTIFACT_NAME = "governance_decision"
EXECUTION_AUTHORIZATION_ARTIFACT_NAME = "execution_authorization"
PURCHASE_ORDER_ARTIFACT_NAME = "purchase_order"
EXECUTION_STATUS_ARTIFACT_NAME = "execution_status"
CONTRACT_VALIDATION_ARTIFACT_NAME = "contract_validation"
EXTERNAL_CALL_ARTIFACT_NAME = "external_call"


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


class RiskAssessmentArtifact(Artifact):
    """A deterministic risk assessment for a procurement decision.

    Produced by :class:`RiskAssessmentAgent` and consumed by
    :class:`GovernanceAgent`. Its ``parent_ids`` reference the originating
    :class:`DecisionArtifact` (by id), continuing the governance lineage.

    ``data`` contract::

        {
            "decision_id": str,            # DecisionArtifact.id (uuid)
            "supplier_id": str,
            "risk_id": str,
            "purchase_amount": float,
            "risk_scores": {
                "financial_risk_score": float,
                "delivery_risk_score": float,
                "quality_risk_score": float,
                "carbon_risk_score": float,
                "overall_risk_score": float,
            },
            "risk_level": str,
            "policy_name": str,
            "created_at": str,
        }
    """

    kind: Literal["risk_assessment"] = "risk_assessment"
    name: str = RISK_ASSESSMENT_ARTIFACT_NAME


class GovernanceDecisionArtifact(Artifact):
    """The governance outcome of a risk assessment.

    Produced by :class:`GovernanceAgent` from a
    :class:`RiskAssessmentArtifact` and a :class:`GovernancePolicy`. Its
    ``parent_ids`` reference the :class:`RiskAssessmentArtifact` (by id).

    ``data`` contract::

        {
            "decision_id": str,
            "supplier_id": str,
            "risk_id": str,
            "status": str,                # APPROVED | APPROVAL_REQUIRED | REJECTED
            "policy_used": str,
            "purchase_amount": float,
            "overall_risk_score": float,
            "risk_level": str,
            "reasons": [str, ...],
            "required_approver": str | None,
        }
    """

    kind: Literal["governance_decision"] = "governance_decision"
    name: str = GOVERNANCE_DECISION_ARTIFACT_NAME


class ExecutionAuthorizationArtifact(Artifact):
    """The final authorization gate before a decision may execute.

    Produced by :class:`ApprovalAgent` from a
    :class:`GovernanceDecisionArtifact`. ``authorization_status`` is
    ``"authorized"`` (decision may execute), ``"pending"`` (human approval
    required), or ``"rejected"`` (decision blocked). Its ``parent_ids``
    reference the :class:`GovernanceDecisionArtifact` (by id).

    ``data`` contract::

        {
            "decision_id": str,
            "risk_assessment_id": str,
            "governance_decision_id": str,
            "authorization_status": str,
            "approved_by": str | None,
            "timestamp": str,             # ISO-8601 UTC
        }
    """

    kind: Literal["execution_authorization"] = "execution_authorization"
    name: str = EXECUTION_AUTHORIZATION_ARTIFACT_NAME


class PurchaseOrderArtifact(Artifact):
    """A purchase order created from an authorized decision (Phase 7).

    Produced by :class:`PurchaseOrderAgent` from an
    :class:`ExecutionAuthorizationArtifact` only when the decision is
    ``authorized`` (governance has already excluded rejected decisions). Its
    ``parent_ids`` reference the originating
    :class:`ExecutionAuthorizationArtifact` (by id), continuing the control →
    action lineage.

    ``data`` contract::

        {
            "order_id": str,
            "decision_id": str,
            "authorization_id": str,
            "supplier_id": str,
            "currency": str,
            "items": [{"material": str, "quantity": int, "unit_price": float}],
            "total_amount": float,
            "quantity": int,
            "status": str,  # CREATED | SUBMITTED | CONFIRMED | SHIPPED | DELIVERED | CANCELLED
            "created_at": str,
            "submitted_at": str | None,
        }
    """

    kind: Literal["purchase_order"] = "purchase_order"
    name: str = PURCHASE_ORDER_ARTIFACT_NAME


class ExecutionStatusArtifact(Artifact):
    """The realized execution lifecycle of a purchase order (Phase 7).

    Produced by :class:`ExecutionTrackingAgent` from a
    :class:`PurchaseOrderArtifact` by asking the :class:`SupplierConnector` for
    the order's status. Its ``parent_ids`` reference the originating
    :class:`PurchaseOrderArtifact` (by id).

    ``data`` contract::

        {
            "order_id": str,
            "purchase_order_id": str,
            "status": str,              # terminal/sampled status from the connector
            "lifecycle": [str, ...],    # deterministic stages the order progressed through
            "tracked_at": str,          # ISO-8601 UTC
        }
    """

    kind: Literal["execution_status"] = "execution_status"
    name: str = EXECUTION_STATUS_ARTIFACT_NAME


class ExternalCallArtifact(Artifact):
    """An outbound call made to an external system (Phase 8 integration layer).

    Produced by agents that talk to ERP/supplier/contract systems so every
    external interaction is auditable and replay-safe. ``data`` contract::

        {
            "system": str,                       # e.g. "coupa", "sap", "supplier_api"
            "action": str,                       # callable name, e.g. "submit_order"
            "order_id": str | None,              # linked purchase order, if any
            "decision_id": str | None,
            "request_payload": dict | None,      # serialized request (no secrets)
            "response_payload": dict | None,     # serialized response (no secrets)
            "status": str,                       # success | error | pending
            "timestamp": str,                    # ISO-8601 UTC
        }
    """

    kind: Literal["external_call"] = "external_call"
    name: str = EXTERNAL_CALL_ARTIFACT_NAME


class ContractValidationArtifact(Artifact):
    """The result of validating a decision against supplier contract(s).

    Produced by :class:`swarm.domain.agents.contract_validation_agent.
    ContractValidationAgent` and consumed by :class:`RiskAssessmentAgent`. Its
    ``parent_ids`` reference the originating :class:`DecisionArtifact` (by id).

    ``data`` contract::

        {
            "decision_id": str,
            "supplier_id": str,
            "contract_id": str | None,          # which contract was applied
            "valid": bool,                      # True → proceed to risk; False → reject
            "reason": str | None,               # human-readable outcome
            "validated_at": str,                # ISO-8601 UTC
        }
    """

    kind: Literal["contract_validation"] = "contract_validation"
    name: str = CONTRACT_VALIDATION_ARTIFACT_NAME
