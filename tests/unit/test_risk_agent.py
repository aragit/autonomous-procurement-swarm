"""Unit tests for the Phase 6 RiskAssessmentAgent."""


import pytest

from swarm import Event, SwarmState
from swarm.core.artifact import Artifact
from swarm.domain.agents import RiskAssessmentAgent
from swarm.domain.risk import RiskLevel
from swarm.domain.strategy import DEFAULT_STRATEGIES
from tests.unit.procurement_helpers import drive

DECISION_ID = "deadbeef-risk-decision"


def seed_decision(state: SwarmState, *, supplier: str = "MinerCorp_A") -> Artifact:
    return state.put_artifact(
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
                            "score": 0.858,
                            "price": 984.0,
                            "policy_passed": True,
                        }
                    ],
                },
            },
            created_by="decision_agent",
            correlation_id="REQ-RSK-01",
        )
    )


def seed_requirement(state: SwarmState, *, quantity: int = 1000) -> None:
    state.put_artifact(
        Artifact(
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
            correlation_id="REQ-RSK-01",
        )
    )


def seed_quote(state: SwarmState, supplier: str = "MinerCorp_A") -> None:
    state.put_artifact(
        Artifact(
            kind="quote",
            name=f"quote_{supplier}",
            data={
                "supplier_id": supplier,
                "price": 984.0,
                "terms": "net_30",
                "metadata": {
                    "quantity": 1000,
                    "lead_time_days": 28,
                    "carbon_footprint_kg": 1_800_000.0,
                    "reliability_score": 0.85,
                },
            },
            created_by="negotiation_agent",
            correlation_id="REQ-RSK-01",
        )
    )


def seed_evaluation(state: SwarmState, supplier: str = "MinerCorp_A") -> None:
    state.put_artifact(
        Artifact(
            kind="evaluation",
            name=f"evaluation_{supplier}",
            data={
                "supplier_id": supplier,
                "score": 0.858,
                "breakdown": {"price": 0.9, "lead_time": 0.8, "esg": 0.95, "reliability": 0.85},
                "strategy": {
                    "strategy_name": "balanced",
                    "weights": DEFAULT_STRATEGIES["balanced"].as_weights(),
                },
                "history": {
                    "applied": False,
                    "adjustment": 0.0,
                    "reliability": None,
                    "total_orders": 0,
                },
                "bid": {},
            },
            created_by="evaluation_agent",
            correlation_id="REQ-RSK-01",
        )
    )


def decision_made_event() -> Event:
    return Event(
        type="DecisionMade",
        source="decision_agent",
        payload={"artifact": "decision", "selected_supplier": "MinerCorp_A"},
        correlation_id="REQ-RSK-01",
    )


@pytest.mark.asyncio
async def test_risk_agent_creates_low_risk_assessment() -> None:
    agent = RiskAssessmentAgent()
    state = SwarmState()
    seed_decision(state)
    seed_requirement(state)
    seed_quote(state)
    seed_evaluation(state)
    bus = await drive(agent, state, decision_made_event())

    risks = state.find_artifacts(kind="risk_assessment")
    assert len(risks) == 1
    risk = risks[0]
    assert risk.parent_ids == [DECISION_ID]
    assert risk.data["supplier_id"] == "MinerCorp_A"
    assert risk.data["risk_level"] == RiskLevel.LOW.value
    assert risk.data["purchase_amount"] == 984000.0
    assert risk.data["risk_scores"]["overall_risk_score"] == 0.1178

    completed = [e for e in bus.event_log() if e.type == "RiskAssessmentCompleted"]
    assert len(completed) == 1
    assert completed[0].payload["risk_level"] == RiskLevel.LOW.value


@pytest.mark.asyncio
async def test_risk_agent_degrades_gracefully_without_quote_or_requirement() -> None:
    agent = RiskAssessmentAgent()
    state = SwarmState()
    seed_decision(state)
    await drive(agent, state, decision_made_event())

    risk = state.get_artifact("risk_assessment")
    assert risk is not None
    assert risk.parent_ids == [DECISION_ID]
    # No requirement -> quantity defaults to 1, so purchase = unit price (984.0).
    assert risk.data["purchase_amount"] == 984.0
    assert risk.data["risk_level"] == RiskLevel.LOW.value


@pytest.mark.asyncio
async def test_risk_agent_ignores_replayed_events() -> None:
    agent = RiskAssessmentAgent()
    state = SwarmState()
    seed_decision(state)
    seed_requirement(state)
    seed_quote(state)
    agent.state = state
    await agent.step(decision_made_event().model_copy(update={"replayed": True}))

    assert state.find_artifacts(kind="risk_assessment") == []
