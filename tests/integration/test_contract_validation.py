"""Tests for the Phase 8 ContractValidationAgent and Contract model."""

from datetime import UTC, datetime, timedelta

import pytest

from swarm import Event, SwarmState
from swarm.core.artifact import Artifact
from swarm.domain.agents import ContractValidationAgent
from swarm.domain.contracts import Contract, ContractStatus
from swarm.domain.events import ProcurementEventType
from tests.unit.procurement_helpers import drive

DECISION_ID = "contract-decision-1"


def _decision_artifact(state: SwarmState, supplier: str = "MinerCorp_A") -> None:
    state.put_artifact(
        Artifact(
            id=DECISION_ID,
            kind="decision",
            name="decision",
            data={
                "selected_supplier": supplier,
                "reasoning": {"ranked": [{"supplier_id": supplier, "score": 0.9}]},
            },
            correlation_id="REQ-CTR-01",
        )
    )


def _requirement_artifact(state: SwarmState, material: str = "aluminum") -> None:
    state.put_artifact(
        Artifact(
            kind="requirement",
            name="requirement",
            data={
                "constraints": {
                    "material": material,
                    "quantity": 100,
                    "budget": 500_000.0,
                    "target_lead_time_days": 30,
                    "max_carbon_per_unit": None,
                }
            },
            correlation_id="REQ-CTR-01",
        )
    )


def _quote_artifact(state: SwarmState, supplier: str = "MinerCorp_A", price: float = 984.0) -> None:
    state.put_artifact(
        Artifact(
            kind="quote",
            name=f"quote_{supplier}",
            data={
                "supplier_id": supplier,
                "price": price,
                "metadata": {"quantity": 100, "carbon_footprint_kg": 100.0},
            },
            correlation_id="REQ-CTR-01",
        )
    )


def _decision_made(supplier: str = "MinerCorp_A") -> Event:
    return Event(
        type=ProcurementEventType.DECISION_MADE,
        source="decision_agent",
        payload={"artifact": "decision", "selected_supplier": supplier},
        correlation_id="REQ-CTR-01",
    )


def _valid_contract(supplier: str = "MinerCorp_A", material: str = "aluminum") -> Contract:
    return Contract(
        contract_id="C-001",
        supplier_id=supplier,
        allowed_items=[material],
        pricing_rules=[],
        expiry_date=None,
        compliance_flags={"active"},
        status=ContractStatus.ACTIVE,
    )


@pytest.mark.asyncio
async def test_valid_contract_publishes_contract_validated() -> None:
    agent = ContractValidationAgent(contracts={"MinerCorp_A": _valid_contract()})
    state = SwarmState()
    _decision_artifact(state, "MinerCorp_A")
    _requirement_artifact(state, "aluminum")
    _quote_artifact(state, "MinerCorp_A")
    bus = await drive(agent, state, _decision_made("MinerCorp_A"))

    events = [e.type for e in bus.event_log()]
    assert ProcurementEventType.CONTRACT_VALIDATED in events
    assert ProcurementEventType.CONTRACT_REJECTED not in events
    cv = state.get_artifact("contract_validation")
    assert cv is not None
    assert cv.data["valid"] is True
    assert cv.data["contract_id"] == "C-001"


@pytest.mark.asyncio
async def test_invalid_supplier_contract_publishes_contract_rejected() -> None:
    contract = Contract(
        contract_id="C-002",
        supplier_id="MinerCorp_A",
        allowed_items=["steel"],
        pricing_rules=[],
        compliance_flags={"active"},
        status=ContractStatus.ACTIVE,
    )
    agent = ContractValidationAgent(contracts={"MinerCorp_A": contract})
    state = SwarmState()
    _decision_artifact(state, "MinerCorp_A")
    _requirement_artifact(state, "aluminum")
    _quote_artifact(state, "MinerCorp_A")
    await drive(agent, state, _decision_made("MinerCorp_A"))

    cv = state.get_artifact("contract_validation")
    assert cv is not None
    assert cv.data["valid"] is False
    assert "aluminum" in cv.data["reason"]
    assert state.find_artifacts(kind="contract_validation")[0].data["reason"]


@pytest.mark.asyncio
async def test_expired_contract_is_rejected() -> None:
    expired = datetime.now(UTC) - timedelta(days=1)
    contract = Contract(
        contract_id="C-003",
        supplier_id="MinerCorp_A",
        allowed_items=["aluminum"],
        pricing_rules=[],
        expiry_date=expired.isoformat(),
        compliance_flags={"active"},
        status=ContractStatus.ACTIVE,
    )
    agent = ContractValidationAgent(contracts={"MinerCorp_A": contract})
    state = SwarmState()
    _decision_artifact(state, "MinerCorp_A")
    _requirement_artifact(state, "aluminum")
    _quote_artifact(state, "MinerCorp_A")
    await drive(agent, state, _decision_made("MinerCorp_A"))
    cv = state.find_artifacts(kind="contract_validation")[0]
    assert cv.data["valid"] is False
    assert "not active" in cv.data["reason"]


@pytest.mark.asyncio
async def test_no_contract_passes_when_require_contract_false() -> None:
    agent = ContractValidationAgent(contracts={}, require_contract=False)
    state = SwarmState()
    _decision_artifact(state, "MinerCorp_A")
    _requirement_artifact(state, "aluminum")
    _quote_artifact(state, "MinerCorp_A")
    bus = await drive(agent, state, _decision_made("MinerCorp_A"))
    assert ProcurementEventType.CONTRACT_VALIDATED in [e.type for e in bus.event_log()]
    assert state.get_artifact("contract_validation").data["valid"] is True


@pytest.mark.asyncio
async def test_no_contract_rejected_when_require_contract_true() -> None:
    agent = ContractValidationAgent(contracts={}, require_contract=True)
    state = SwarmState()
    _decision_artifact(state, "MinerCorp_A")
    _requirement_artifact(state, "aluminum")
    _quote_artifact(state, "MinerCorp_A")
    bus = await drive(agent, state, _decision_made("MinerCorp_A"))
    types = [e.type for e in bus.event_log()]
    assert ProcurementEventType.CONTRACT_REJECTED in types
    cv = state.find_artifacts(kind="contract_validation")[0]
    assert cv.data["valid"] is False
    assert "No contract" in cv.data["reason"]


@pytest.mark.asyncio
async def test_contract_validation_agent_ignores_replayed_events() -> None:
    agent = ContractValidationAgent(contracts={"MinerCorp_A": _valid_contract()})
    state = SwarmState()
    _decision_artifact(state, "MinerCorp_A")
    _requirement_artifact(state, "aluminum")
    _quote_artifact(state, "MinerCorp_A")
    agent.state = state
    await agent.step(_decision_made("MinerCorp_A").model_copy(update={"replayed": True}))
    assert state.find_artifacts(kind="contract_validation") == []
