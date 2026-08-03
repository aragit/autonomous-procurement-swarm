"""Integration tests for the Phase 7 execution lifecycle (control → action)."""

import pytest

from swarm.domain import CREATE_REQUIREMENT_INTENT
from swarm.domain.agents import ApprovalAgent, ExecutionTrackingAgent, PurchaseOrderAgent
from swarm.domain.artifacts import (
    EXECUTION_AUTHORIZATION_ARTIFACT_NAME,
    EXECUTION_STATUS_ARTIFACT_NAME,
    PURCHASE_ORDER_ARTIFACT_NAME,
)
from swarm.domain.wiring import build_procurement_swarm


@pytest.mark.asyncio
async def test_authorized_purchase_creates_order_and_executes_to_delivered() -> None:
    swarm = build_procurement_swarm(request_id="REQ-EXEC-OK", goal="Source aluminum")
    await swarm.start()
    await swarm.send_message(
        CREATE_REQUIREMENT_INTENT,
        {"material": "aluminum", "quantity": 1000, "budget": 3_500_000.0},
        sender="user",
        correlation_id="REQ-EXEC-OK-CONV",
    )
    await swarm.shutdown()
    state = swarm.state

    auth = state.get_artifact(EXECUTION_AUTHORIZATION_ARTIFACT_NAME)
    assert auth is not None
    assert auth.data["authorization_status"] == "authorized"

    order = state.get_artifact(PURCHASE_ORDER_ARTIFACT_NAME)
    assert order is not None
    assert order.data["supplier_id"] == "MinerCorp_A"
    assert order.data["status"] == "SUBMITTED"
    assert order.data["total_amount"] == 984_000.0
    assert auth.id in order.parent_ids

    status = state.get_artifact(EXECUTION_STATUS_ARTIFACT_NAME)
    assert status is not None
    assert status.data["status"] == "DELIVERED"
    assert status.data["lifecycle"] == ["SUBMITTED", "CONFIRMED", "SHIPPED", "DELIVERED"]
    assert order.id in status.parent_ids


@pytest.mark.asyncio
async def test_approval_required_blocks_execution_until_approve_and_execute() -> None:
    """A purchase above the approval threshold stays pending until explicitly approved."""
    swarm = build_procurement_swarm(request_id="REQ-EXEC-PEND", goal="Source aluminum")
    await swarm.start()
    await swarm.send_message(
        CREATE_REQUIREMENT_INTENT,
        {"material": "aluminum", "quantity": 2500, "budget": 5_000_000.0},
        sender="user",
        correlation_id="REQ-EXEC-PEND-CONV",
    )
    await swarm.shutdown()
    state = swarm.state

    auth = state.get_artifact(EXECUTION_AUTHORIZATION_ARTIFACT_NAME)
    assert auth is not None
    assert auth.data["authorization_status"] == "pending"
    # No purchase order is created while authorization is pending.
    assert state.find_artifacts(kind=PURCHASE_ORDER_ARTIFACT_NAME) == []

    # Explicit approval + execution on the remembered state (POST /swarm/.../execute).
    ApprovalAgent().approve(state)
    auth = state.get_artifact(EXECUTION_AUTHORIZATION_ARTIFACT_NAME)
    assert auth.data["authorization_status"] == "authorized"

    order_agent = PurchaseOrderAgent()
    execution_agent = ExecutionTrackingAgent()
    order = order_agent.create_order(state)
    execution = execution_agent.track(state)

    assert order is not None
    assert order.data["status"] == "SUBMITTED"
    assert order.data["total_amount"] == 984.0 * 2500
    assert execution is not None
    assert execution.data["status"] == "DELIVERED"


@pytest.mark.asyncio
async def test_rejected_decision_blocks_purchase_order() -> None:
    """A rejected governance decision never produces an authorization or order."""
    from swarm.domain.governance import GovernancePolicy

    # A zero-tolerance policy: any risk above 0.1 is rejected. The aluminum
    # baseline risk (0.1178) therefore crosses the threshold end-to-end.
    rejecting_policy = GovernancePolicy(
        name="zero_tolerance",
        max_purchase_amount=5_000_000.0,
        max_risk_score=0.1,
        requires_approval_above_amount=2_000_000.0,
        requires_approval_for_high_risk=True,
    )
    swarm = build_procurement_swarm(
        request_id="REQ-EXEC-RJCT",
        goal="Source aluminum",
        governance_policy=rejecting_policy,
    )
    await swarm.start()
    await swarm.send_message(
        CREATE_REQUIREMENT_INTENT,
        {"material": "aluminum", "quantity": 1000, "budget": 3_500_000.0},
        sender="user",
        correlation_id="REQ-EXEC-RJCT-CONV",
    )
    await swarm.shutdown()
    state = swarm.state

    governance = state.get_artifact("governance_decision")
    assert governance is not None
    assert governance.data["status"] == "REJECTED"
    # Rejection never reaches authorization, so no order can be created.
    assert state.get_artifact(EXECUTION_AUTHORIZATION_ARTIFACT_NAME) is None
    assert state.find_artifacts(kind=PURCHASE_ORDER_ARTIFACT_NAME) == []
