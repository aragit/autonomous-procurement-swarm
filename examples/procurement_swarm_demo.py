"""End-to-end Phase 7 demo of the deterministic procurement swarm.

Flow: one ``CreateRequirement`` message fans out through the parallel, wired
swarm — requirement → strategy selection → per-supplier discovery →
per-supplier evaluation (weighted by the strategy) → per-supplier quoting →
completion-tracked decision → decision explanation. The decision then flows
through the deterministic control and action tiers:

- governance tail: risk assessment → governance decision → approval →
  execution authorization
- execution tier: purchase order → execution tracking → delivery status

producing a final :class:`DecisionArtifact`, a human-readable
:class:`DecisionExplanationArtifact`, :class:`RiskAssessmentArtifact`,
:class:`GovernanceDecisionArtifact`, :class:`ExecutionAuthorizationArtifact`,
:class:`PurchaseOrderArtifact` and :class:`ExecutionStatusArtifact` entirely
through event-driven collaboration (``Message → Event → Artifact``).

Run from the repository root with:

    python -m examples.procurement_swarm_demo

The aluminum baseline (LOW blended risk, purchase amount below the 2M approval
threshold) resolves through the standard policy straight to an authorized
execution and an auto-created purchase order tracked to delivery. To exercise
the approval-required branch, raise the quantity so the resulting purchase
amount exceeds the policy's ``requires_approval_above_amount`` and call
``POST /swarm/{request_id}/approve`` then ``POST /swarm/{request_id}/execute``.
"""

import asyncio

from swarm import SwarmState
from swarm.core.logging import configure_logging
from swarm.domain import (
    CREATE_REQUIREMENT_INTENT,
    ProcurementEventType,
)
from swarm.domain.wiring import build_procurement_swarm


async def run() -> SwarmState:
    """Execute the procurement flow and return the resulting shared state."""
    swarm = build_procurement_swarm(
        request_id="REQ-002", goal="Source 1000 units of aluminum"
    )
    await swarm.start()
    await swarm.send_message(
        CREATE_REQUIREMENT_INTENT,
        {
            "text": "Source 1000 units of aluminum at market price",
            "material": "aluminum",
            "quantity": 1000,
            "budget": 3_500_000.0,
            "target_lead_time_days": 30,
        },
        sender="user",
        correlation_id="REQ-002-CONV",
    )
    await swarm.shutdown()
    return swarm.state


def main() -> None:
    """Configure logging, run the flow, and summarize the result."""
    configure_logging()
    state = asyncio.run(run())

    requirement = state.get_artifact("requirement")
    strategy = state.get_artifact("strategy")
    pool = state.get_artifact("suppliers")
    evaluations = state.find_artifacts(kind="evaluation")
    quotes = state.find_artifacts(kind="quote")
    decision = state.get_artifact("decision")
    explanation = state.get_artifact("decision_explanation")

    print(f"request_id: {state.request_id}")
    print(f"goal: {state.goal}")
    print(f"requirement: {requirement.data['constraints'] if requirement else None}")
    print(
        f"strategy: {strategy.data['strategy_name'] if strategy else None} "
        f"weights={strategy.data.get('weights') if strategy else None}"
    )
    print(f"suppliers discovered: {len(pool.data['suppliers']) if pool else 0}")
    print(
        "evaluations: "
        + (
            ", ".join(
                f"{evaluation.data['supplier_id']}={evaluation.data['score']}"
                for evaluation in sorted(
                    evaluations, key=lambda item: item.data["score"], reverse=True
                )
            )
            if evaluations
            else "none"
        )
    )
    print(
        "quotes: "
        + (
            ", ".join(
                f"{quote.data['supplier_id']}@${quote.data['price']}" for quote in quotes
            )
            if quotes
            else "none"
        )
    )
    print(f"decision: {decision.data if decision else None}")
    if explanation is not None:
        data = explanation.data
        print(f"explanation.strategy_used: {data['strategy_used']}")
        print(f"explanation.selected_supplier: {data['selected_supplier']}")
        for factor in data.get("top_factors", []):
            print(f"  factor: {factor}")
        for entry in data.get("rejected_suppliers", []):
            print(
                f"  rejected: {entry['supplier_id']} "
                f"(score={entry['score']}, policy={entry['policy_passed']}, "
                f"reason={entry['reason']})"
            )

    risk = state.get_artifact("risk_assessment")
    governance = state.get_artifact("governance_decision")
    authorization = state.get_artifact("execution_authorization")
    if risk is not None:
        data = risk.data
        scores = data.get("risk_scores", {})
        components = {
            k: round(float(scores[k]), 4)
            for k in (
                "financial_risk_score",
                "delivery_risk_score",
                "quality_risk_score",
                "carbon_risk_score",
            )
            if k in scores
        }
        print(
            f"risk: level={data['risk_level']} score={float(scores['overall_risk_score']):.4f} "
            f"components={components}"
        )
    if governance is not None:
        data = governance.data
        print(
            f"governance: status={data['status']} policy={data['policy_used']} "
            f"reasons={data.get('reasons', [])}"
        )
    if authorization is not None:
        data = authorization.data
        print(
            f"authorization: status={data['authorization_status']} "
            f"approved_by={data.get('approved_by')} "
            f"risk_id={data.get('risk_assessment_id')}"
        )

    order = state.get_artifact("purchase_order")
    execution = state.get_artifact("execution_status")
    if order is not None:
        data = order.data
        print(
            f"purchase_order: id={data['order_id']} supplier={data['supplier_id']} "
            f"amount={data['total_amount']} status={data['status']}"
        )
    if execution is not None:
        data = execution.data
        print(
            f"execution: status={data['status']} lifecycle={data.get('lifecycle')}"
        )

    print(f"completions: {state.completions}")
    per_supplier = (
        ProcurementEventType.SUPPLIER_EVALUATED,
        ProcurementEventType.QUOTE_GENERATED,
    )
    print(
        f"per-supplier events: "
        f"{sum(1 for e in state.events if e.type in per_supplier)}"
    )
    print(f"event flow: {[event.type for event in state.events]}")
    print(
        "correlation ids: "
        + str(sorted({event.correlation_id for event in state.events if event.correlation_id}))
    )


if __name__ == "__main__":
    main()
