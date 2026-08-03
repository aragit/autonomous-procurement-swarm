"""RiskAssessmentAgent — deterministic risk assessment for a decision.

Reacts to the ``DecisionMade`` event, loads the originating
:class:`DecisionArtifact` together with the requirement, the selected supplier's
quote, its :class:`SupplierPerformance` history and the active
:class:`GovernancePolicy`, then computes a deterministic
:class:`RiskAssessment` and writes a :class:`RiskAssessmentArtifact` (lineaged
to the decision by id). It publishes ``RiskAssessmentCompleted`` so the
:class:`GovernanceAgent` can run next.

Risk is a pure function of the decision's signals — purchase amount, supplier
delivery/quality history, quote carbon and policy thresholds — so the same
decision always yields the same risk record. There is no LLM and no autonomous
learning.
"""

from typing import Any

import structlog

from configs.settings import settings
from swarm.core.agent import BaseAgent
from swarm.core.capability import Capability
from swarm.core.event import Event
from swarm.core.state import SwarmState
from swarm.domain.artifacts import (
    DECISION_ARTIFACT_NAME,
    REQUIREMENT_ARTIFACT_NAME,
    RiskAssessmentArtifact,
)
from swarm.domain.events import ProcurementEventType
from swarm.domain.governance import STANDARD_POLICY, GovernancePolicy
from swarm.domain.risk import RiskAssessment
from swarm.memory import SupplierMemoryStore

logger = structlog.get_logger(__name__)


class RiskAssessmentAgent(BaseAgent):
    """Assesses the deterministic procurement risk of a decision."""

    name = "risk_assessment_agent"
    description = "Computes a deterministic risk assessment from a procurement decision"
    capabilities = [
        Capability(
            name="risk.assess",
            description="Assesses procurement risk for a selected supplier",
        )
    ]

    def __init__(
        self,
        *,
        memory: SupplierMemoryStore | None = None,
        policy: GovernancePolicy | None = None,
        esg_baselines: dict[str, float] | None = None,
    ) -> None:
        super().__init__(memory=memory)
        self._policy = policy if policy is not None else STANDARD_POLICY
        self._esg_baselines = (
            dict(esg_baselines)
            if esg_baselines is not None
            else dict(settings.evaluation.esg_baselines)
        )
        self._correlation_id: str | None = None
        self._decision_artifact: str = DECISION_ARTIFACT_NAME
        self._selected_supplier: str = ""
        self._pending = False
        self._risk: RiskAssessment | None = None

    async def perceive(self, event: Event) -> None:
        if event.replayed:
            return
        if event.type == ProcurementEventType.DECISION_MADE:
            self._pending = True
            self._correlation_id = event.correlation_id
            self._decision_artifact = str(event.payload.get("artifact") or DECISION_ARTIFACT_NAME)
            self._selected_supplier = str(event.payload.get("selected_supplier") or "")
            self._risk = None

    async def reason(self, state: SwarmState) -> None:
        if not self._pending:
            return
        decision = state.get_artifact(self._decision_artifact)
        if decision is None:
            self._pending = False
            return

        supplier_id = self._selected_supplier or str(decision.data.get("selected_supplier") or "")
        decision_id = str(decision.id)

        purchase_amount, quantity = self._purchase_amount(state, supplier_id)
        carbon_per_unit = self._carbon_per_unit(state, supplier_id, quantity)
        evaluation_score = self._evaluation_score(state, supplier_id)
        performance = self._supplier_history(supplier_id)
        max_carbon_per_unit, material = self._carbon_constraints(state)
        baseline = float(self._esg_baselines.get(material, 10_000.0))

        self._risk = RiskAssessment.from_signals(
            supplier_id=supplier_id,
            decision_id=decision_id,
            purchase_amount=purchase_amount,
            max_purchase_amount=float(self._policy.max_purchase_amount),
            performance=performance,
            evaluation_score=evaluation_score,
            carbon_per_unit=carbon_per_unit,
            max_carbon_per_unit=max_carbon_per_unit,
            esg_baseline=baseline,
        )
        logger.info(
            "agent_executing",
            agent=self.name,
            phase="risk_assessed",
            supplier_id=supplier_id,
            risk_level=self._risk.risk_level.value,
            overall_risk_score=self._risk.overall_risk_score,
            correlation_id=self._correlation_id,
        )

    async def act(self, state: SwarmState) -> None:
        if not self._pending or self._risk is None:
            return
        risk = self._risk
        artifact = RiskAssessmentArtifact(
            data={
                "decision_id": risk.decision_id,
                "supplier_id": risk.supplier_id,
                "risk_id": risk.risk_id,
                "purchase_amount": risk.purchase_amount,
                "risk_scores": risk.risk_scores,
                "risk_level": risk.risk_level.value,
                "policy_name": self._policy.name,
                "created_at": risk.created_at,
            },
            parent_ids=[risk.decision_id],
            tags={"supplier": risk.supplier_id, "decision": risk.decision_id},
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
                type=ProcurementEventType.RISK_ASSESSMENT_COMPLETED,
                source=self.name,
                payload={
                    "artifact": artifact.name,
                    "decision_id": risk.decision_id,
                    "supplier_id": risk.supplier_id,
                    "risk_level": risk.risk_level.value,
                    "overall_risk_score": risk.overall_risk_score,
                },
                correlation_id=self._correlation_id,
            )
        )
        self._pending = False
        self._risk = None

    def _memory(self) -> SupplierMemoryStore | None:
        memory = getattr(self, "memory", None)
        if isinstance(memory, SupplierMemoryStore):
            return memory
        return None

    @staticmethod
    def _purchase_amount(state: SwarmState, supplier_id: str) -> tuple[float, int]:
        """Total purchase amount (unit price * quantity) for the selected supplier."""
        decision = state.get_artifact(DECISION_ARTIFACT_NAME)
        if decision is None:
            return 0.0, 1
        quantity = 1
        requirement = state.get_artifact(REQUIREMENT_ARTIFACT_NAME)
        if requirement is not None:
            constraints = requirement.data.get("constraints", {}) or {}
            quantity = int(constraints.get("quantity") or 1)
        price = 0.0
        for entry in decision.data.get("reasoning", {}).get("ranked", []) or []:
            if str(entry.get("supplier_id")) == supplier_id:
                price = float(entry.get("price") or 0.0)
                break
        return round(price * quantity, 2), quantity

    @staticmethod
    def _carbon_per_unit(state: SwarmState, supplier_id: str, quantity: int) -> float | None:
        """Per-unit carbon (kg CO2e) for the selected supplier's quote, if known."""
        quote = state.get_artifact(f"quote_{supplier_id}")
        if quote is None:
            return None
        footprint = float(quote.data.get("metadata", {}).get("carbon_footprint_kg") or 0.0)
        if quantity <= 0:
            return None
        return round(footprint / quantity, 4)

    @staticmethod
    def _evaluation_score(state: SwarmState, supplier_id: str) -> float | None:
        """The composite evaluation score for the supplier, if available."""
        evaluation = state.get_artifact(f"evaluation_{supplier_id}")
        if evaluation is None:
            return None
        return float(evaluation.data.get("score") or 0.0)

    def _supplier_history(self, supplier_id: str) -> Any:
        """The supplier's cumulative performance record, if any history exists."""
        memory = self._memory()
        if memory is None:
            return None
        return memory.get_supplier_performance(supplier_id)

    @staticmethod
    def _carbon_constraints(state: SwarmState) -> tuple[float | None, str]:
        """The carbon constraint (if any) and material from the requirement."""
        max_carbon_per_unit = None
        material = "steel"
        requirement = state.get_artifact(REQUIREMENT_ARTIFACT_NAME)
        if requirement is not None:
            constraints = requirement.data.get("constraints", {}) or {}
            material = str(constraints.get("material") or "steel")
            carbon = constraints.get("max_carbon_per_unit")
            if carbon is not None:
                max_carbon_per_unit = float(carbon)
        return max_carbon_per_unit, material
