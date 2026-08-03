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
DECISION_ARTIFACT_NAME = "decision"


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
            },
            "metadata": {                      # raw input + RFQ normalization
                "raw": dict,
                "rfq": dict | None,            # core RFQPayload dump when valid
            },
        }
    """

    kind: Literal["requirement"] = "requirement"
    name: str = REQUIREMENT_ARTIFACT_NAME


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
