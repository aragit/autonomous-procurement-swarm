"""ApprovalAgent — closes the governance gate with a deterministic authorization.

Reacts to ``GovernanceDecisionMade`` and, depending on the governance status:

* ``APPROVED``         — grants authorization immediately (the decision may execute).
* ``APPROVAL_REQUIRED`` — records a *pending* authorization awaiting a manual
  approval (the ``ApproveDecision`` intent / ``POST /swarm/{request_id}/approve``
  endpoint).
* ``REJECTED``         — blocks execution (no authorization artifact is produced).

For Phase 6 the human approver is a deterministic simulation: a pending
authorization is resolved by :meth:`approve` / the API endpoint into an
authorized execution authorization. There is no autonomous approval and no LLM.
"""

from datetime import UTC, datetime

import structlog

from swarm.core.agent import BaseAgent
from swarm.core.artifact import Artifact
from swarm.core.capability import Capability
from swarm.core.event import Event
from swarm.core.state import SwarmState
from swarm.domain.artifacts import (
    EXECUTION_AUTHORIZATION_ARTIFACT_NAME,
    ExecutionAuthorizationArtifact,
)
from swarm.domain.events import ProcurementEventType

logger = structlog.get_logger(__name__)

SIMULATED_APPROVER = "governance_sim"


class ApprovalAgent(BaseAgent):
    """Turns a governance decision into an execution authorization."""

    name = "approval_agent"
    description = "Authorizes or blocks execution of a governance-approved decision"
    capabilities = [
        Capability(
            name="governance.approve",
            description="Resolves governance decisions into execution authorizations",
        )
    ]

    def __init__(self, *, simulated_approver: str = SIMULATED_APPROVER) -> None:
        super().__init__()
        self._simulated_approver = simulated_approver
        self._correlation_id: str | None = None
        self._governance_artifact: str = "governance_decision"
        self._decision_payload: dict[str, str] | None = None
        self._pending = False

    async def perceive(self, event: Event) -> None:
        if event.replayed:
            return
        if event.type == ProcurementEventType.GOVERNANCE_DECISION_MADE:
            self._pending = True
            self._correlation_id = event.correlation_id
            self._governance_artifact = str(event.payload.get("artifact") or "governance_decision")
            self._decision_payload = dict(event.payload)

    async def reason(self, state: SwarmState) -> None:
        # The authorization status is fully determined by the governance payload.
        return

    async def act(self, state: SwarmState) -> None:
        if not self._pending or self._decision_payload is None:
            return
        status = str(self._decision_payload.get("status") or "")
        decision_id = str(self._decision_payload.get("decision_id") or "")
        governance = state.get_artifact(self._governance_artifact)
        if governance is None:
            self._pending = False
            self._decision_payload = None
            return

        if status == "REJECTED":
            logger.info(
                "agent_executing",
                agent=self.name,
                phase="approval_rejected",
                decision_id=decision_id,
                correlation_id=self._correlation_id,
            )
            await self.publish_event(
                Event(
                    type=ProcurementEventType.APPROVAL_REJECTED,
                    source=self.name,
                    payload={
                        "decision_id": decision_id,
                        "authorization_status": "rejected",
                        "governance_decision_id": governance.id,
                    },
                    correlation_id=self._correlation_id,
                )
            )
            self._pending = False
            self._decision_payload = None
            return

        auth_status = "pending" if status == "APPROVAL_REQUIRED" else "authorized"
        approved_by = self._simulated_approver if status == "APPROVED" else None
        authorization = ExecutionAuthorizationArtifact(
            data={
                "decision_id": decision_id or str(governance.data.get("decision_id", "")),
                "risk_assessment_id": str(governance.data.get("risk_id", "")),
                "governance_decision_id": governance.id,
                "authorization_status": auth_status,
                "approved_by": approved_by,
                "timestamp": datetime.now(UTC).isoformat(),
            },
            parent_ids=[governance.id],
            tags={"decision": str(governance.data.get("decision_id", ""))},
            created_by=self.name,
            correlation_id=self._correlation_id,
        )
        state.put_artifact(authorization)
        logger.info(
            "artifact_created",
            agent=self.name,
            kind=authorization.kind,
            name=authorization.name,
            authorization_status=auth_status,
            correlation_id=self._correlation_id,
        )

        event_type = (
            ProcurementEventType.APPROVAL_REQUIRED
            if auth_status == "pending"
            else ProcurementEventType.APPROVAL_GRANTED
        )
        await self.publish_event(
            Event(
                type=event_type,
                source=self.name,
                payload={
                    "artifact": authorization.name,
                    "decision_id": authorization.data["decision_id"],
                    "authorization_status": auth_status,
                    "approved_by": approved_by,
                },
                correlation_id=self._correlation_id,
            )
        )
        self._pending = False
        self._decision_payload = None

    def approve(
        self, state: SwarmState, *, approver: str = SIMULATED_APPROVER
    ) -> Artifact | None:
        """Resolve a *pending* authorization (deterministic simulated approval).

        Used by the ``POST /swarm/{request_id}/approve`` endpoint against a
        remembered swarm run: finds the pending
        :class:`ExecutionAuthorizationArtifact`, flips it to ``authorized`` under
        ``approver`` (preserving prior versions for audit), and returns it.
        Returns ``None`` if there is no pending authorization to resolve.

        Phase 6 simulates the human: a pending authorization (which governance
        already vetted as non-rejected) is granted deterministically.
        """
        pending = state.get_artifact(EXECUTION_AUTHORIZATION_ARTIFACT_NAME)
        if pending is None or pending.data.get("authorization_status") != "pending":
            return None
        updated = pending.update(
            {
                **pending.data,
                "authorization_status": "authorized",
                "approved_by": approver,
                "timestamp": datetime.now(UTC).isoformat(),
            },
            by=self.name,
        )
        state.put_artifact(updated)
        logger.info(
            "agent_executing",
            agent=self.name,
            phase="approval_granted",
            decision_id=pending.data.get("decision_id"),
            approver=approver,
            correlation_id=pending.correlation_id,
        )
        return updated
