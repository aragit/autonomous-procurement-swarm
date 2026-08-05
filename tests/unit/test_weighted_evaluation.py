"""Unit tests for Phase 4 strategy-weighted evaluation scoring.

Confirms the ``balanced`` strategy reproduces the exact Phase 3 composite
scores and that the other strategies shift the composite toward their priority
(price or carbon) without any randomness.
"""

import pytest

from swarm import Event, SwarmState
from swarm.domain import ProcurementEventType, SupplierDiscoveryAgent
from swarm.domain.agents import EvaluationAgent
from swarm.domain.artifacts import RequirementArtifact, StrategyArtifact
from swarm.domain.strategy import DEFAULT_STRATEGIES
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
            correlation_id="REQ-CONV",
        )
    )


def seed_strategy(state: SwarmState, strategy_name: str) -> None:
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
            correlation_id="REQ-CONV",
        )
    )


async def seed_pool(state: SwarmState) -> None:
    seed_requirement(state)
    discovery = SupplierDiscoveryAgent()
    await drive(
        discovery,
        state,
        Event(
            type=ProcurementEventType.REQUIREMENT_CREATED,
            source="requirement_agent",
            payload={"artifact": "requirement"},
            correlation_id="REQ-CONV",
        ),
    )


async def scores_under(strategy_name: str) -> dict[str, float]:
    state = SwarmState()
    await seed_pool(state)
    seed_strategy(state, strategy_name)
    agent = EvaluationAgent()
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
                correlation_id="REQ-CONV",
            ),
        )
    return {e.data["supplier_id"]: e.data["score"] for e in state.find_artifacts(kind="evaluation")}


@pytest.mark.asyncio
async def test_balanced_strategy_reproduces_phase3_scores():
    scores = await scores_under("balanced")
    assert scores == {
        "MinerCorp_A": 0.858,
        "PremiumSteel_E": 0.803,
        "DistribCorp_B": 0.795,
        "RecycleCorp_C": 0.7687,
        "TraderCorp_D": 0.7263,
    }


@pytest.mark.asyncio
async def test_strategies_produce_distinct_composites():
    balanced = await scores_under("balanced")
    low_carbon = await scores_under("low_carbon")
    cost = await scores_under("cost_optimized")

    assert balanced != low_carbon
    assert balanced != cost
    assert low_carbon != cost


@pytest.mark.asyncio
async def test_low_carbon_favors_low_footprint_supplier():
    scores = await scores_under("low_carbon")

    assert scores["RecycleCorp_C"] > scores["MinerCorp_A"]
    assert scores["RecycleCorp_C"] > scores["DistribCorp_B"]
    assert scores["RecycleCorp_C"] > scores["PremiumSteel_E"]


@pytest.mark.asyncio
async def test_evaluation_records_strategy_used():
    scores = await scores_under("low_carbon")
    assert len(scores) == 5

    state = SwarmState()
    await seed_pool(state)
    seed_strategy(state, "low_carbon")
    agent = EvaluationAgent()
    await drive(
        agent,
        state,
        Event(
            type=ProcurementEventType.SUPPLIER_DISCOVERED,
            source="supplier_discovery_agent",
            payload={
                "supplier_id": "MinerCorp_A",
                "material": "aluminum",
                "artifact": "suppliers",
            },
            correlation_id="REQ-CONV",
        ),
    )

    miner = state.get_artifact("evaluation_MinerCorp_A")
    assert miner is not None
    assert miner.data["strategy"]["strategy_name"] == "low_carbon"
    assert miner.data["strategy"]["weights"] == DEFAULT_STRATEGIES["low_carbon"].as_weights()
