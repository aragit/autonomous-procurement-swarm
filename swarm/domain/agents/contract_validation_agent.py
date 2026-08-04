"""ContractValidationAgent — applies supplier contracts to a decision.

Subscribes to ``DecisionMade`` and runs *before* the
:class:`~swarm.domain.agents.risk_agent.RiskAssessmentAgent`: a decision that
fails contract validation is short-circuited to ``ContractRejected`` so the
risk and governance stages never see it. A decision that passes (or a
no-contract situation where ``require_contract`` is False) emits
``ContractValidated`` and lets the normal risk → governance chain proceed.

This agent does **not** implement approval logic — it only enforces contract
compliance and publishes an auditable
:class:`~swarm.domain.artifacts.ContractValidationArtifact`.
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog

from swarm.core.agent import BaseAgent
from swarm.core.event import Event
from swarm.core.state import SwarmState
from swarm.domain.artifacts import (
    DECISION_ARTIFACT_NAME,
    REQUIREMENT_ARTIFACT_NAME,
    ContractValidationArtifact,
)
from swarm.domain.contracts import Contract
from swarm.domain.events import ProcurementEventType

logger = structlog.get_logger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class ContractValidationAgent(BaseAgent):
    """Validates a decision against the selected supplier's contract(s)."""

    name = "contract_validation_agent"
    description = "Validates a procurement decision against supplier contracts"
    capabilities = []

    def __init__(
        self,
        *,
        contracts: dict[str, Contract] | None = None,
        require_contract: bool = False,
    ) -> None:
        super().__init__()
        self._contracts: dict[str, Contract] = dict(contracts or {})
        self._require_contract = require_contract
        self._correlation_id: str | None = None
        self._pending = False
        self._decision_id: str = ""
        self._supplier_id: str = ""
        self._decision_artifact: str = DECISION_ARTIFACT_NAME
        self._material: str = ""
        self._unit_price: float | None = None

    async def perceive(self, event: Event) -> None:
        if event.replayed:
            return
        if event.type != ProcurementEventType.DECISION_MADE:
            return
        self._pending = True
        self._correlation_id = event.correlation_id
        self._decision_artifact = str(event.payload.get("artifact") or DECISION_ARTIFACT_NAME)
        self._supplier_id = str(event.payload.get("selected_supplier") or "")
        self._decision_id = ""
        self._material = ""
        self._unit_price = None

    async def reason(self, state: SwarmState) -> None:
        if not self._pending:
            return
        decision = state.get_artifact(self._decision_artifact)
        if decision is None:
            self._pending = False
            return
        self._decision_id = str(decision.id)
        self._supplier_id = (
            self._supplier_id
            or str(decision.data.get("selected_supplier") or "")
        )
        self._material, self._unit_price = self._line_details(state)
        logger.info(
            "agent_executing",
            agent=self.name,
            phase="contract_validation",
            supplier_id=self._supplier_id,
            decision_id=self._decision_id,
            correlation_id=self._correlation_id,
        )

    async def act(self, state: SwarmState) -> None:
        if not self._pending or not self._decision_id:
            return
        valid, contract, reason = self._evaluate()
        artifact = ContractValidationArtifact(
            data={
                "decision_id": self._decision_id,
                "supplier_id": self._supplier_id,
                "contract_id": contract.contract_id if contract else None,
                "valid": valid,
                "reason": reason,
                "validated_at": _now_iso(),
            },
            parent_ids=[self._decision_id],
            tags={"decision": self._decision_id, "supplier": self._supplier_id},
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
        if valid:
            event_type = ProcurementEventType.CONTRACT_VALIDATED
            outcome = "valid"
        else:
            event_type = ProcurementEventType.CONTRACT_REJECTED
            outcome = "rejected"
        await self.publish_event(
            Event(
                type=event_type,
                source=self.name,
                payload={
                    "artifact": DECISION_ARTIFACT_NAME,
                    "selected_supplier": self._supplier_id,
                    "decision_id": self._decision_id,
                    "supplier_id": self._supplier_id,
                    "contract_id": contract.contract_id if contract else None,
                    "valid": valid,
                    "reason": reason,
                    "outcome": outcome,
                },
                correlation_id=self._correlation_id,
            )
        )
        self._pending = False
        self._decision_id = ""

    def _line_details(self, state: SwarmState) -> tuple[str, float | None]:
        """Derive (material, unit_price) for the selected supplier from state."""
        material = "steel"
        unit_price: float | None = None
        requirement = state.get_artifact(REQUIREMENT_ARTIFACT_NAME)
        if requirement is not None:
            constraints = requirement.data.get("constraints", {}) or {}
            material = str(constraints.get("material") or "steel")
        quote = state.get_artifact(f"quote_{self._supplier_id}")
        if quote is not None:
            unit_price = float(quote.data.get("price") or 0.0)
        return material, unit_price

    def _evaluate(self) -> tuple[bool, Contract | None, str | None]:
        """Apply supplier contracts and the require_contract policy.

        Returns ``(valid, contract_or_none, reason)``.
        """
        contract = self._contracts.get(self._supplier_id)
        if contract is None:
            if self._require_contract:
                return (
                    False,
                    None,
                    f"No contract on file for supplier {self._supplier_id}",
                )
            return True, None, None
        valid, reason = contract.validate(
            supplier_id=self._supplier_id,
            material=self._material,
            unit_price=self._unit_price,
        )
        return valid, contract, reason

    def add_contract(self, contract: Contract) -> None:
        """Register or replace a contract keyed by supplier id (wiring/test hook)."""
        self._contracts[contract.supplier_id] = contract
        self._contracts = dict(self._contracts)

    @property
    def contracts(self) -> dict[str, Contract]:
        return dict(self._contracts)
