"""GovernanceAgent — applies policy to a risk assessment.

Reacts to ``RiskAssessmentCompleted``, loads the
:class:`RiskAssessmentArtifact` (and the originating
:class:`DecisionArtifact`), applies the active :class:`GovernancePolicy` via the
pure :meth:`GovernanceDecision.from_risk`, and writes a
:class:`GovernanceDecisionArtifact` (lineaged to the risk assessment by id). It
publishes ``GovernanceDecisionMade`` with the authorization status so the
:class:`ApprovalAgent` can act on it.

Governance is deterministic: the same risk assessment and policy always yield the
same status. It is *not* the per-bid :class:`PolicyEngine` (which still guards
compliance at the award stage); this is the post-decision safety and
authorization gate.
"""


import structlog

from swarm.core.agent import BaseAgent
from swarm.core.capability import Capability
from swarm.core.event import Event
from swarm.core.state import SwarmState
from swarm.domain.artifacts import (
    GovernanceDecisionArtifact,
)
from swarm.domain.events import ProcurementEventType
from swarm.domain.governance import (
    STANDARD_POLICY,
    GovernanceDecision,
    GovernancePolicy,
    GovernanceStatus,
)
from swarm.domain.risk import RiskAssessment

logger = structlog.get_logger(__name__)


class GovernanceAgent(BaseAgent):
    """Applies governance policy to a risk assessment."""

    name = "governance_agent"
    description = "Validates a risk assessment against governance policy"
    capabilities = [
        Capability(
            name="governance.validate",
            description="Decides approve / approval-required / reject for a decision",
        )
    ]

    def __init__(self, *, policy: GovernancePolicy | None = None) -> None:
        super().__init__()
        self._policy = policy if policy is not None else STANDARD_POLICY
        self._correlation_id: str | None = None
        self._risk_artifact: str = "risk_assessment"
        self._risk_artifact_id: str = ""
        self._pending = False
        self._contract_rejected = False
        self._reject_reason: str | None = None
        self._reject_supplier_id: str = ""
        self._reject_decision_id: str = ""
        self._decision: GovernanceDecision | None = None

    async def perceive(self, event: Event) -> None:
        if event.replayed:
            return
        if event.type == ProcurementEventType.RISK_ASSESSMENT_COMPLETED:
            self._pending = True
            self._correlation_id = event.correlation_id
            self._risk_artifact = str(event.payload.get("artifact") or "risk_assessment")
            self._contract_rejected = False
            self._decision = None
        elif event.type == ProcurementEventType.CONTRACT_REJECTED:
            self._pending = True
            self._correlation_id = event.correlation_id
            self._risk_artifact = str(event.payload.get("artifact") or "decision")
            self._contract_rejected = True
            self._reject_reason = str(event.payload.get("reason") or "contract rejected")
            self._reject_supplier_id = str(event.payload.get("supplier_id") or "")
            self._reject_decision_id = str(event.payload.get("decision_id") or "")
            self._decision = None

    async def reason(self, state: SwarmState) -> None:
        if not self._pending:
            return
        if self._contract_rejected:
            self._decision = GovernanceDecision(
                decision_id=self._reject_decision_id,
                supplier_id=self._reject_supplier_id,
                risk_id=self._reject_decision_id,
                status=GovernanceStatus.REJECTED,
                policy_used=self._policy.name,
                purchase_amount=0.0,
                overall_risk_score=0.0,
                risk_level="UNKNOWN",
                reasons=[f"Contract validation failed: {self._reject_reason}"],
                required_approver=None,
            )
            self._risk_artifact_id = self._reject_decision_id or self._risk_artifact
            return
        risk_artifact = state.get_artifact(self._risk_artifact)
        if risk_artifact is None:
            self._pending = False
            return
        self._risk_artifact_id = risk_artifact.id
        risk = RiskAssessment.from_artifact(risk_artifact.data)
        self._decision = GovernanceDecision.from_risk(
            risk, self._policy, required_approver="governance_sim"
        )
        logger.info(
            "agent_executing",
            agent=self.name,
            phase="governance_decision",
            supplier_id=self._decision.supplier_id,
            status=self._decision.status.value,
            policy_used=self._decision.policy_used,
            correlation_id=self._correlation_id,
        )

    async def act(self, state: SwarmState) -> None:
        if not self._pending or self._decision is None:
            return
        decision = self._decision
        parent = self._risk_artifact_id or self._risk_artifact
        artifact = GovernanceDecisionArtifact(
            data=decision.to_summary(),
            parent_ids=[parent] if parent else [],
            tags={
                "decision": self._decision.decision_id,
                "supplier": self._decision.supplier_id,
                "status": self._decision.status.value,
            },
            created_by=self.name,
            correlation_id=self._correlation_id,
        )
        state.put_artifact(artifact)
        logger.info(
            "artifact_created",
            agent=self.name,
            kind=artifact.kind,
            name=artifact.name,
            correlation_id=self._correlation_id,
        )
        await self.publish_event(
            Event(
                type=ProcurementEventType.GOVERNANCE_DECISION_MADE,
                source=self.name,
                payload={
                    "artifact": artifact.name,
                    "decision_id": self._decision.decision_id,
                    "supplier_id": self._decision.supplier_id,
                    "status": self._decision.status.value,
                    "policy_used": self._decision.policy_used,
                },
                correlation_id=self._correlation_id,
            )
        )
        self._pending = False
        self._contract_rejected = False
        self._decision = None
