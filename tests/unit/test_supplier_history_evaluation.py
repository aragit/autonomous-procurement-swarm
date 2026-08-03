"""Unit tests for Phase 5 evaluation-time supplier-history adjustment.

Confirms the ``balanced`` strategy still produces the exact Phase 4 scores when
no supplier memory is attached, and that a strong reliability record shifts the
composite deterministically while an unknown supplier is unaffected.
"""

import pytest

from swarm import Event, SwarmState
from swarm.domain import ProcurementEventType, SupplierDiscoveryAgent
from swarm.domain.agents import EvaluationAgent
from swarm.domain.artifacts import RequirementArtifact, StrategyArtifact
from swarm.domain.strategy import DEFAULT_STRATEGIES
from swarm.memory import SupplierMemoryStore
from tests.unit.procurement_helpers import drive

SUPPLIER_IDS = [
    "MinerCorp_A",
    "DistribCorp_B",
    "RecycleCorp_C",
    "TraderCorp_D",
    "PremiumSteel_E",
]


def seed_requirement(state: SwarmState) -> None:
    state.put_artifact(
        RequirementArtifact(
            data={
                "text": "buy aluminum",
                "constraints": {
                    "material": "aluminum",
                    "quantity": 1000,
                    "budget": 2_000_000.0,
                    "max_unit_price": 2640.0,
                    "target_lead_time_days": 30,
                },
                "metadata": {},
            },
            created_by="requirement_agent",
            correlation_id="REQ-501",
        )
    )


def seed_strategy(state: SwarmState, strategy_name: str = "balanced") -> None:
    strategy = DEFAULT_STRATEGIES[strategy_name]
    state.put_artifact(
        StrategyArtifact(
            data={
                "strategy_name": strategy.name,
                "description": strategy.description,
                "weights": strategy.as_weights(),
            },
            parent_ids=["requirement"],
            created_by="strategy_agent",
            correlation_id="REQ-501",
        )
    )


async def seed_pool(state: SwarmState) -> None:
    seed_requirement(state)
    seed_strategy(state, "balanced")
    discovery = SupplierDiscoveryAgent()
    await drive(
        discovery,
        state,
        Event(
            type=ProcurementEventType.REQUIREMENT_CREATED,
            source="requirement_agent",
            payload={"artifact": "requirement"},
            correlation_id="REQ-501",
        ),
    )


async def evaluate_all(state: SwarmState, agent: EvaluationAgent) -> dict[str, dict]:
    for supplier_id in SUPPLIER_IDS:
        await drive(
            agent,
            state,
            Event(
                type=ProcurementEventType.SUPPLIER_DISCOVERED,
                source="supplier_discovery_agent",
                payload={
                    "supplier_id": supplier_id,
                    "material": "aluminum",
                    "artifact": "suppliers",
                },
                correlation_id="REQ-501",
            ),
        )
    return {
        artifact.data["supplier_id"]: artifact.data
        for artifact in state.find_artifacts(kind="evaluation")
    }


@pytest.mark.asyncio
async def test_no_history_preserves_phase4_balanced_scores() -> None:
    state = SwarmState()
    await seed_pool(state)
    agent = EvaluationAgent()
    results = await evaluate_all(state, agent)

    scores = {sid: data["score"] for sid, data in results.items()}
    assert scores == {
        "MinerCorp_A": 0.858,
        "PremiumSteel_E": 0.803,
        "DistribCorp_B": 0.795,
        "RecycleCorp_C": 0.7687,
        "TraderCorp_D": 0.7263,
    }
    for data in results.values():
        assert data["history"]["applied"] is False
        assert data["history"]["adjustment"] == 0.0
        assert data["history"]["total_orders"] == 0


@pytest.mark.asyncio
async def test_history_adjustment_shifts_composite_for_recorded_supplier() -> None:
    store = SupplierMemoryStore()
    perf = store.update_from_outcome(
        {
            "supplier_id": "MinerCorp_A",
            "delivered_on_time": True,
            "quality_score": 0.92,
            "actual_price": 984.0,
            "carbon_score": 1800.0,
        },
        reference_price=984.0,
    )
    assert store.get_supplier_performance("MinerCorp_A") is perf

    state = SwarmState()
    await seed_pool(state)
    agent = EvaluationAgent(memory=store)
    results = await evaluate_all(state, agent)

    # MinerCorp_A had reliability 1.0 -> +5% on top of 0.858 -> 0.908
    assert results["MinerCorp_A"]["score"] == 0.908
    assert results["MinerCorp_A"]["history"]["applied"] is True
    assert results["MinerCorp_A"]["history"]["adjustment"] == 0.05
    assert results["MinerCorp_A"]["history"]["reliability"] == 1.0

    # Other suppliers have no history -> unchanged Phase 4 scores
    assert results["DistribCorp_B"]["score"] == 0.795
    assert results["DistribCorp_B"]["history"]["applied"] is False
    assert results["TraderCorp_D"]["score"] == 0.7263


@pytest.mark.asyncio
async def test_poor_history_penalizes_composite() -> None:
    store = SupplierMemoryStore()
    for _ in range(8):
        store.update_from_outcome(
            {
                "supplier_id": "MinerCorp_A",
                "delivered_on_time": False,
                "quality_score": 0.7,
                "actual_price": 984.0,
                "carbon_score": 1800.0,
            },
            reference_price=984.0,
        )
    assert store.get_supplier_performance("MinerCorp_A").delivery_reliability == 0.0

    state = SwarmState()
    await seed_pool(state)
    agent = EvaluationAgent(memory=store)
    results = await evaluate_all(state, agent)

    # 0.858 - 0.05 -> 0.808, clamped into range
    assert results["MinerCorp_A"]["score"] == 0.808
    assert results["MinerCorp_A"]["history"]["adjustment"] == -0.05
