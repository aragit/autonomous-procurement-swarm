"""Unit tests for the Phase 6 deterministic governance policy and decision."""

import pytest

from swarm.domain.governance import (
    STANDARD_POLICY,
    STRICT_POLICY,
    GovernanceDecision,
    GovernanceStatus,
)
from swarm.domain.risk import RiskAssessment, RiskLevel


def _risk(
    overall: float,
    level: RiskLevel,
    purchase: float,
    risk_id: str = "r-1",
) -> RiskAssessment:
    return RiskAssessment(
        risk_id=risk_id,
        supplier_id="MinerCorp_A",
        decision_id="dec-1",
        purchase_amount=purchase,
        financial_risk_score=overall,
        delivery_risk_score=overall,
        quality_risk_score=overall,
        carbon_risk_score=overall,
        overall_risk_score=overall,
        risk_level=level,
        created_at="2026-06-01T00:00:00+00:00",
    )


def test_low_risk_within_amount_is_approved() -> None:
    decision = GovernanceDecision.from_risk(_risk(0.1, RiskLevel.LOW, 984_000.0), STANDARD_POLICY)
    assert decision.status == GovernanceStatus.APPROVED
    assert decision.required_approver is None
    assert decision.policy_used == "standard"
    assert decision.reasons


def test_medium_risk_requires_approval() -> None:
    decision = GovernanceDecision.from_risk(
        _risk(0.45, RiskLevel.MEDIUM, 984_000.0), STANDARD_POLICY
    )
    assert decision.status == GovernanceStatus.APPROVAL_REQUIRED
    assert decision.required_approver == "governance_sim"


def test_high_risk_above_amount_requires_approval() -> None:
    decision = GovernanceDecision.from_risk(
        _risk(0.65, RiskLevel.HIGH, 6_000_000.0), STANDARD_POLICY
    )
    assert decision.status == GovernanceStatus.APPROVAL_REQUIRED
    assert decision.required_approver == "governance_sim"


def test_critical_risk_is_rejected() -> None:
    decision = GovernanceDecision.from_risk(
        _risk(0.85, RiskLevel.CRITICAL, 12_000_000.0), STANDARD_POLICY
    )
    assert decision.status == GovernanceStatus.REJECTED
    assert decision.required_approver is None
    assert any("CRITICAL" in reason for reason in decision.reasons)


def test_score_above_max_risk_is_rejected() -> None:
    # HIGH level but overall exceeds the strict policy's lower max_risk_score.
    decision = GovernanceDecision.from_risk(_risk(0.7, RiskLevel.HIGH, 6_000_000.0), STRICT_POLICY)
    assert decision.status == GovernanceStatus.REJECTED


def test_strict_policy_rejects_high_risk_that_standard_would_approve() -> None:
    high = _risk(0.65, RiskLevel.HIGH, 600_000.0)
    standard = GovernanceDecision.from_risk(high, STANDARD_POLICY)
    strict = GovernanceDecision.from_risk(high, STRICT_POLICY)
    assert standard.status == GovernanceStatus.APPROVAL_REQUIRED
    assert strict.status == GovernanceStatus.REJECTED


@pytest.mark.parametrize(
    "status",
    [
        GovernanceStatus.APPROVED,
        GovernanceStatus.APPROVAL_REQUIRED,
        GovernanceStatus.REJECTED,
    ],
)
def test_default_required_approver_only_for_approval_required(status: GovernanceStatus) -> None:
    level = {
        GovernanceStatus.APPROVED: RiskLevel.LOW,
        GovernanceStatus.APPROVAL_REQUIRED: RiskLevel.MEDIUM,
        GovernanceStatus.REJECTED: RiskLevel.CRITICAL,
    }[status]
    decision = GovernanceDecision.from_risk(_risk(0.2, level, 100_000.0), STANDARD_POLICY)
    if status == GovernanceStatus.APPROVAL_REQUIRED:
        assert decision.required_approver == "governance_sim"
    else:
        assert decision.required_approver is None
