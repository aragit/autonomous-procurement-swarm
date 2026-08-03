"""Deterministic governance policies for the Phase 6 procurement swarm.

A :class:`GovernancePolicy` is a plain, serializable set of hard thresholds that
the :class:`GovernanceAgent` applies to a :class:`~swarm.domain.risk.RiskAssessment`
to produce a :class:`GovernanceDecision` outcome. Policies are static data — they
are not learned and they do not replace the existing :class:`PolicyEngine` (which
still guards per-bid compliance); governance sits *after* the decision to gate
whether the selected award is safe and authorized to execute.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from swarm.domain.risk import RiskAssessment, RiskLevel


class GovernanceStatus(StrEnum):
    """Authorization outcome of a governance check."""

    APPROVED = "APPROVED"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    REJECTED = "REJECTED"


class GovernancePolicy(BaseModel):
    """Static thresholds that turn a risk assessment into a governance decision.

    - ``max_purchase_amount``: policy ceiling on the purchase value; exceeding it
      drives the financial-risk signal and the approval threshold.
    - ``max_risk_score``: overall risk score above which a decision is rejected.
    - ``requires_approval_above_amount``: purchase value that forces a human
      approval step before the award can execute.
    - ``requires_approval_for_high_risk``: when true, any HIGH-risk decision
      additionally forces approval (CRITICAL is always rejected).
    """

    name: str = "standard"
    max_purchase_amount: float = Field(gt=0)
    max_risk_score: float = Field(ge=0.0, le=1.0)
    requires_approval_above_amount: float = Field(ge=0.0)
    requires_approval_for_high_risk: bool = True


STANDARD_POLICY = GovernancePolicy(
    name="standard",
    max_purchase_amount=5_000_000.0,
    max_risk_score=0.80,
    requires_approval_above_amount=2_000_000.0,
    requires_approval_for_high_risk=True,
)

STRICT_POLICY = GovernancePolicy(
    name="strict",
    max_purchase_amount=1_000_000.0,
    max_risk_score=0.50,
    requires_approval_above_amount=500_000.0,
    requires_approval_for_high_risk=True,
)

DEFAULT_GOVERNANCE_POLICIES: dict[str, GovernancePolicy] = {
    "standard": STANDARD_POLICY,
    "strict": STRICT_POLICY,
}


class GovernanceDecision(BaseModel):
    """The outcome of applying a policy to a risk assessment."""

    decision_id: str
    supplier_id: str
    risk_id: str
    status: GovernanceStatus
    policy_used: str
    purchase_amount: float
    overall_risk_score: float
    risk_level: str
    reasons: list[str] = Field(default_factory=list)
    required_approver: str | None = None

    @classmethod
    def from_risk(
        cls,
        risk: RiskAssessment,
        policy: GovernancePolicy,
        *,
        required_approver: str = "governance_sim",
    ) -> GovernanceDecision:
        """Apply ``policy`` to ``risk`` and produce the deterministic decision."""
        reasons: list[str] = []
        if risk.risk_level == RiskLevel.CRITICAL:
            reasons.append(f"Risk level {risk.risk_level.value} exceeds the critical threshold")
        if risk.overall_risk_score > policy.max_risk_score:
            reasons.append(
                f"Overall risk score {risk.overall_risk_score} exceeds policy limit "
                f"{policy.max_risk_score}"
            )

        if risk.risk_level == RiskLevel.CRITICAL or risk.overall_risk_score > policy.max_risk_score:
            return cls(
                decision_id=risk.decision_id,
                supplier_id=risk.supplier_id,
                risk_id=risk.risk_id,
                status=GovernanceStatus.REJECTED,
                policy_used=policy.name,
                purchase_amount=risk.purchase_amount,
                overall_risk_score=risk.overall_risk_score,
                risk_level=risk.risk_level.value,
                reasons=reasons,
                required_approver=None,
            )

        needs_approval = (
            risk.purchase_amount > policy.requires_approval_above_amount
            or (
                policy.requires_approval_for_high_risk
                and risk.risk_level == RiskLevel.HIGH
            )
            or risk.risk_level == RiskLevel.MEDIUM
        )
        if needs_approval:
            if risk.purchase_amount > policy.requires_approval_above_amount:
                reasons.append(
                    f"Purchase amount {risk.purchase_amount} exceeds approval threshold "
                    f"{policy.requires_approval_above_amount}"
                )
            if risk.risk_level == RiskLevel.MEDIUM:
                reasons.append(f"Risk level {risk.risk_level.value} requires approval")
            if risk.risk_level == RiskLevel.HIGH:
                reasons.append(
                    f"Risk level {risk.risk_level.value} requires approval under policy "
                    f"{policy.name}"
                )
            return cls(
                decision_id=risk.decision_id,
                supplier_id=risk.supplier_id,
                risk_id=risk.risk_id,
                status=GovernanceStatus.APPROVAL_REQUIRED,
                policy_used=policy.name,
                purchase_amount=risk.purchase_amount,
                overall_risk_score=risk.overall_risk_score,
                risk_level=risk.risk_level.value,
                reasons=reasons,
                required_approver=required_approver,
            )

        return cls(
            decision_id=risk.decision_id,
            supplier_id=risk.supplier_id,
            risk_id=risk.risk_id,
            status=GovernanceStatus.APPROVED,
            policy_used=policy.name,
            purchase_amount=risk.purchase_amount,
            overall_risk_score=risk.overall_risk_score,
            risk_level=risk.risk_level.value,
            reasons=[
                f"Risk level {risk.risk_level.value} and purchase amount within policy limits"
            ],
            required_approver=None,
        )

    def to_summary(self) -> dict[str, Any]:
        """Serializable summary used by the API and tests."""
        return self.model_dump()
