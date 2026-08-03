"""Unit tests for the Phase 6 ApprovalAgent."""

import pytest

from swarm import Event, SwarmState
from swarm.core.artifact import Artifact
from swarm.domain.agents import ApprovalAgent
from swarm.domain.artifacts import EXECUTION_AUTHORIZATION_ARTIFACT_NAME
from swarm.domain.events import ProcurementEventType
from tests.unit.procurement_helpers import drive


def seed_governance(
    state: SwarmState, status: str, *, decision_id: str = "dec-1", risk_id: str = "r-1"
) -> Artifact:
    return state.put_artifact(
        Artifact(
            id="gov-1",
            kind="governance_decision",
            name="governance_decision",
            data={
                "decision_id": decision_id,
                "supplier_id": "MinerCorp_A",
                "risk_id": risk_id,
                "status": status,
                "policy_used": "standard",
                "purchase_amount": 984_000.0,
                "overall_risk_score": 0.12,
                "risk_level": "LOW",
                "reasons": ["within policy"],
                "required_approver": None if status != "APPROVAL_REQUIRED" else "governance_sim",
            },
            created_by="governance_agent",
            correlation_id="REQ-APP-01",
        )
    )


def governance_event(status: str) -> Event:
    return Event(
        type=ProcurementEventType.GOVERNANCE_DECISION_MADE,
        source="governance_agent",
        payload={
            "artifact": "governance_decision",
            "decision_id": "dec-1",
            "supplier_id": "MinerCorp_A",
            "status": status,
            "policy_used": "standard",
        },
        correlation_id="REQ-APP-01",
    )


@pytest.mark.asyncio
async def test_approved_decision_creates_authorized_authorization() -> None:
    state = SwarmState()
    gov = seed_governance(state, "APPROVED")
    agent = ApprovalAgent()
    bus = await drive(agent, state, governance_event("APPROVED"))

    auths = state.find_artifacts(kind="execution_authorization")
    assert len(auths) == 1
    auth = auths[0]
    assert auth.parent_ids == [gov.id]
    assert auth.data["authorization_status"] == "authorized"
    assert auth.data["approved_by"] == "governance_sim"
    assert auth.data["decision_id"] == "dec-1"

    granted = [e for e in bus.event_log() if e.type == ProcurementEventType.APPROVAL_GRANTED]
    assert len(granted) == 1


@pytest.mark.asyncio
async def test_approval_required_creates_pending_authorization() -> None:
    state = SwarmState()
    gov = seed_governance(state, "APPROVAL_REQUIRED")
    agent = ApprovalAgent()
    bus = await drive(agent, state, governance_event("APPROVAL_REQUIRED"))

    auths = state.find_artifacts(kind="execution_authorization")
    assert len(auths) == 1
    auth = auths[0]
    assert auth.parent_ids == [gov.id]
    assert auth.data["authorization_status"] == "pending"
    assert auth.data["approved_by"] is None

    required = [e for e in bus.event_log() if e.type == ProcurementEventType.APPROVAL_REQUIRED]
    assert len(required) == 1


@pytest.mark.asyncio
async def test_rejected_decision_blocks_authorization() -> None:
    state = SwarmState()
    seed_governance(state, "REJECTED")
    agent = ApprovalAgent()
    bus = await drive(agent, state, governance_event("REJECTED"))

    assert state.find_artifacts(kind="execution_authorization") == []
    rejected = [e for e in bus.event_log() if e.type == ProcurementEventType.APPROVAL_REJECTED]
    assert len(rejected) == 1
    assert rejected[0].payload["authorization_status"] == "rejected"


@pytest.mark.asyncio
async def test_manual_approve_resolves_pending_authorization() -> None:
    state = SwarmState()
    seed_governance(state, "APPROVAL_REQUIRED")
    agent = ApprovalAgent()
    await drive(agent, state, governance_event("APPROVAL_REQUIRED"))

    resolved = agent.approve(state, approver="governance_sim")
    assert resolved is not None
    assert resolved.data["authorization_status"] == "authorized"
    assert resolved.data["approved_by"] == "governance_sim"

    # A second call finds nothing pending.
    assert agent.approve(state, approver="governance_sim") is None


@pytest.mark.asyncio
async def test_manual_approve_noops_without_pending() -> None:
    state = SwarmState()
    agent = ApprovalAgent()
    assert agent.approve(state) is None
    assert state.find_artifacts(kind=EXECUTION_AUTHORIZATION_ARTIFACT_NAME) == []
