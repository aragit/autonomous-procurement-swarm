"""Integration test: the Phase 6 governance + approval flow.

Full chain through the wired swarm's deterministic governance tail:

    Decision -> RiskAssessment -> GovernanceDecision -> Approval -> Authorization

Three scenarios:
  * LOW risk  -> fully funded, clean history       -> APPROVED + authorized
  * HIGH risk -> oversized purchase, poor history  -> APPROVAL_REQUIRED (pending)
  * CRITICAL  -> extreme purchase + poor history   -> REJECTED (no authorization)
"""

from datetime import UTC, datetime, timedelta

import pytest

from swarm import Event, SwarmState
from swarm.core.artifact import Artifact
from swarm.domain import CREATE_REQUIREMENT_INTENT
from swarm.domain.contracts import Contract, ContractStatus
from swarm.domain.events import ProcurementEventType
from swarm.domain.risk import RiskLevel
from swarm.domain.supplier import SupplierPerformance
from swarm.domain.wiring import build_procurement_swarm
from swarm.memory import SupplierMemoryStore

DECISION_ID = "deadbeef-gov-integration"
CID = "REQ-GOV-01"


def build_history(
    supplier_id: str, orders: int, on_time: int, quality: float
) -> SupplierPerformance:
    perf = SupplierPerformance(supplier_id)
    for _ in range(on_time):
        perf = perf.apply_outcome(
            delivered_on_time=True,
            quality_score=quality,
            price_competitiveness=1.0,
            carbon_score=1800.0,
        )
    for _ in range(orders - on_time):
        perf = perf.apply_outcome(
            delivered_on_time=False,
            quality_score=quality,
            price_competitiveness=1.0,
            carbon_score=1800.0,
        )
    return perf


def seed_context(
    state: SwarmState,
    *,
    supplier: str = "MinerCorp_A",
    quantity: int = 1000,
    price: float = 984.0,
    carbon_unit: float = 1800.0,
    eval_score: float = 0.858,
) -> None:
    state.put_artifact(
        Artifact(
            id="req-int",
            kind="requirement",
            name="requirement",
            data={
                "text": "buy aluminum",
                "constraints": {
                    "material": "aluminum",
                    "quantity": quantity,
                    "budget": 2_000_000.0,
                    "target_lead_time_days": 30,
                    "max_carbon_per_unit": None,
                },
                "metadata": {},
            },
            created_by="requirement_agent",
            correlation_id=CID,
        )
    )
    state.put_artifact(
        Artifact(
            kind="quote",
            name=f"quote_{supplier}",
            data={
                "supplier_id": supplier,
                "price": price,
                "terms": "net_30",
                "metadata": {
                    "quantity": quantity,
                    "lead_time_days": 28,
                    "carbon_footprint_kg": round(carbon_unit * quantity, 2),
                    "reliability_score": 0.85,
                },
            },
            created_by="negotiation_agent",
            correlation_id=CID,
        )
    )
    state.put_artifact(
        Artifact(
            kind="evaluation",
            name=f"evaluation_{supplier}",
            data={
                "supplier_id": supplier,
                "score": eval_score,
                "breakdown": {
                    "price": 0.9,
                    "lead_time": 0.8,
                    "esg": 0.95,
                    "reliability": 0.85,
                },
                "strategy": {"strategy_name": "balanced", "weights": {}},
                "history": {
                    "applied": False,
                    "adjustment": 0.0,
                    "reliability": None,
                    "total_orders": 0,
                },
                "bid": {},
            },
            created_by="evaluation_agent",
            correlation_id=CID,
        )
    )
    state.put_artifact(
        Artifact(
            id=DECISION_ID,
            kind="decision",
            name="decision",
            data={
                "selected_supplier": supplier,
                "reasoning": {
                    "criteria": "score",
                    "ranked": [
                        {
                            "supplier_id": supplier,
                            "score": eval_score,
                            "price": price,
                            "policy_passed": True,
                            "policy_reason": "POLICY_PASSED",
                        }
                    ],
                },
            },
            created_by="decision_agent",
            correlation_id=CID,
        )
    )


@pytest.mark.asyncio
async def test_low_risk_full_flow_is_authorized() -> None:
    """Full end-to-end flow: a clean aluminum procurement is APPROVED + authorized."""
    store = SupplierMemoryStore()
    swarm = build_procurement_swarm(
        request_id="REQ-GOV-LOW",
        goal="low-risk procurement",
        supplier_memory=store,
    )
    await swarm.start()
    await swarm.send_message(
        CREATE_REQUIREMENT_INTENT,
        {
            "text": "Source 1000 units of aluminum",
            "material": "aluminum",
            "quantity": 1000,
            "budget": 2_000_000.0,
            "target_lead_time_days": 30,
        },
        sender="user",
        correlation_id="REQ-GOV-LOW-CONV",
    )
    await swarm.shutdown()

    state = swarm.state
    assert state.get_artifact("decision").data["selected_supplier"] == "MinerCorp_A"

    risk = state.get_artifact("risk_assessment")
    assert risk is not None
    assert risk.data["risk_level"] == RiskLevel.LOW.value

    governance = state.get_artifact("governance_decision")
    assert governance is not None
    assert governance.data["status"] == "APPROVED"

    authorization = state.get_artifact("execution_authorization")
    assert authorization is not None
    assert authorization.data["authorization_status"] == "authorized"
    assert authorization.data["approved_by"] == "governance_sim"

    # Lineage: decision -> risk -> governance -> authorization (by id).
    decision = state.get_artifact("decision")
    assert risk.parent_ids == [decision.id]
    assert governance.parent_ids == [risk.id]
    assert authorization.parent_ids == [governance.id]


@pytest.mark.asyncio
async def test_high_risk_requires_approval() -> None:
    """An oversized, poor-history award lands in APPROVAL_REQUIRED (pending)."""
    store = SupplierMemoryStore()
    store.save_performance(build_history("MinerCorp_A", orders=5, on_time=0, quality=0.4))

    swarm = build_procurement_swarm(
        request_id="REQ-GOV-HIGH",
        goal="high-risk procurement",
        supplier_memory=store,
    )
    await swarm.start()
    state = swarm.state
    # purchase_amount = 1000 * 7000 = 7,000,000 -> financial risk pushes HIGH
    seed_context(state, quantity=7000, price=1000.0)
    await swarm.dispatch(
        Event(
            type=ProcurementEventType.DECISION_MADE,
            source="decision_agent",
            payload={
                "artifact": "decision",
                "selected_supplier": "MinerCorp_A",
                "decision_id": DECISION_ID,
            },
            correlation_id="REQ-GOV-HIGH-CONV",
        )
    )
    await swarm.shutdown()

    risk = state.get_artifact("risk_assessment")
    assert risk is not None
    assert risk.data["risk_level"] == RiskLevel.HIGH.value
    assert risk.data["purchase_amount"] == 7_000_000.0

    governance = state.get_artifact("governance_decision")
    assert governance.data["status"] == "APPROVAL_REQUIRED"
    assert governance.data["required_approver"] == "governance_sim"

    authorization = state.get_artifact("execution_authorization")
    assert authorization is not None
    assert authorization.data["authorization_status"] == "pending"
    assert authorization.data["approved_by"] is None

    required = [e for e in state.events if e.type == ProcurementEventType.APPROVAL_REQUIRED]
    assert len(required) == 1


@pytest.mark.asyncio
async def test_critical_risk_is_rejected() -> None:
    """An extreme, poor-history award is REJECTED with no authorization."""
    store = SupplierMemoryStore()
    store.save_performance(build_history("MinerCorp_A", orders=5, on_time=0, quality=0.1))

    swarm = build_procurement_swarm(
        request_id="REQ-GOV-CRIT",
        goal="critical-risk procurement",
        supplier_memory=store,
    )
    await swarm.start()
    state = swarm.state
    # purchase_amount = 1000 * 12000 = 12,000,000 -> CRITICAL
    seed_context(state, quantity=12000, price=1000.0)
    await swarm.dispatch(
        Event(
            type=ProcurementEventType.DECISION_MADE,
            source="decision_agent",
            payload={
                "artifact": "decision",
                "selected_supplier": "MinerCorp_A",
                "decision_id": DECISION_ID,
            },
            correlation_id="REQ-GOV-CRIT-CONV",
        )
    )
    await swarm.shutdown()

    risk = state.get_artifact("risk_assessment")
    assert risk is not None
    assert risk.data["risk_level"] == RiskLevel.CRITICAL.value

    governance = state.get_artifact("governance_decision")
    assert governance.data["status"] == "REJECTED"
    assert governance.data["required_approver"] is None

    # Rejected decisions must not produce an execution authorization.
    assert state.get_artifact("execution_authorization") is None
    rejected = [e for e in state.events if e.type == ProcurementEventType.APPROVAL_REJECTED]
    assert len(rejected) == 1


@pytest.mark.asyncio
async def test_contract_rejection_short_circuits_to_rejected_governance() -> None:
    """The contract pre-gate is a HARD GATE (pre-risk).

    DecisionMade -> ContractRejected -> GovernanceDecision(REJECTED):
      - a valid-looking decision is still rejected because its supplier contract
        is expired;
      - risk is never assessed (RiskAssessmentAgent only subscribes to
        ContractValidated);
      - the rejected governance decision flows to approval, which publishes
        ApprovalRejected and produces NO execution authorization, purchase order
        or execution status.
    """
    expired = datetime.now(UTC) - timedelta(days=1)
    expired_contract = Contract(
        contract_id="C-REJ-01",
        supplier_id="MinerCorp_A",
        allowed_items=["aluminum"],
        pricing_rules=[],
        expiry_date=expired.isoformat(),
        compliance_flags={"active"},
        status=ContractStatus.ACTIVE,
    )

    store = SupplierMemoryStore()
    swarm = build_procurement_swarm(
        request_id="REQ-GOV-REJ",
        goal="contract-rejected procurement",
        supplier_memory=store,
        contracts={"MinerCorp_A": expired_contract},
        require_contract=True,
    )
    await swarm.start()
    state = swarm.state
    seed_context(state)
    await swarm.dispatch(
        Event(
            type=ProcurementEventType.DECISION_MADE,
            source="decision_agent",
            payload={
                "artifact": "decision",
                "selected_supplier": "MinerCorp_A",
                "decision_id": DECISION_ID,
            },
            correlation_id="REQ-GOV-REJ-CONV",
        )
    )
    await swarm.shutdown()

    events = [e.type for e in state.events]

    # The contract gate rejected the decision.
    assert ProcurementEventType.CONTRACT_REJECTED in events
    cv = state.get_artifact("contract_validation")
    assert cv is not None
    assert cv.data["valid"] is False
    assert "not active" in cv.data["reason"]
    assert cv.parent_ids == [DECISION_ID]

    # Contract = HARD GATE (pre-risk): no risk assessment is ever produced.
    assert state.get_artifact("risk_assessment") is None
    assert ProcurementEventType.RISK_ASSESSMENT_COMPLETED not in events

    # Governance still runs on the rejection and emits a REJECTED decision.
    governance = state.get_artifact("governance_decision")
    assert governance is not None
    assert governance.data["status"] == "REJECTED"
    assert governance.data["required_approver"] is None

    # Approval publishes a rejection but no authorization artifact is created.
    assert state.get_artifact("execution_authorization") is None
    rejected = [e for e in state.events if e.type == ProcurementEventType.APPROVAL_REJECTED]
    assert len(rejected) == 1
    assert rejected[0].payload["authorization_status"] == "rejected"
    assert ProcurementEventType.APPROVAL_GRANTED not in events

    # Nothing past the gate could ever execute.
    assert state.get_artifact("purchase_order") is None
    assert state.get_artifact("execution_status") is None
    assert ProcurementEventType.PURCHASE_ORDER_CREATED not in events
    assert ProcurementEventType.EXECUTION_STATUS_UPDATED not in events
