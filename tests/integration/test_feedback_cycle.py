"""Integration test: the deterministic feedback cycle (Phase 5).

Demonstrates the supplier-intelligence arc end to end with a single shared
in-memory store:

    Requirement -> Strategy -> Decision -> Outcome -> SupplierPerformanceMemory
    -> Future Evaluation (history-adjusted)
"""

from typing import Any

import pytest

from swarm.domain import CREATE_REQUIREMENT_INTENT, RECORD_OUTCOME_INTENT
from swarm.domain.artifacts import DECISION_ARTIFACT_NAME
from swarm.domain.wiring import build_procurement_swarm
from swarm.memory import SupplierMemoryStore


async def run_procurement(swarm) -> dict[str, Any]:
    await swarm.start()
    await swarm.send_message(
        CREATE_REQUIREMENT_INTENT,
        {
            "text": "Source 1000 units of aluminum at market price",
            "material": "aluminum",
            "quantity": 1000,
            "budget": 2_000_000.0,
            "target_lead_time_days": 30,
        },
        sender="user",
        correlation_id=f"{swarm.state.request_id}-CONV",
    )
    await swarm.shutdown()
    decision = swarm.state.get_artifact(DECISION_ARTIFACT_NAME)
    assert decision is not None
    return {
        "decision": decision,
        "winner": decision.data["selected_supplier"],
        "ranked": decision.data["reasoning"]["ranked"],
    }


@pytest.mark.asyncio
async def test_full_feedback_cycle_then_history_adjusted_evaluation() -> None:
    shared_store = SupplierMemoryStore()

    # Phase 1: initial procurement -> decision
    swarm1 = build_procurement_swarm(
        request_id="REQ-FB-01", goal="initial procurement", supplier_memory=shared_store
    )
    result = await run_procurement(swarm1)
    assert result["winner"] == "MinerCorp_A"

    winner = result["winner"]
    assert shared_store.get_supplier_performance(winner) is None

    # Phase 2: record outcome -> supplier performance updated (shares memory store)
    outcome_swarm = build_procurement_swarm(
        request_id="REQ-FB-02", goal="record outcome", supplier_memory=shared_store
    )
    await outcome_swarm.start()
    await outcome_swarm.send_message(
        RECORD_OUTCOME_INTENT,
        {
            "decision_id": result["decision"].id,
            "supplier_id": winner,
            "delivered_on_time": True,
            "quality_score": 0.92,
            "actual_price": 984.0,
            "carbon_score": 1800.0,
        },
        sender="user",
        correlation_id="REQ-FB-02-CONV",
    )
    await outcome_swarm.shutdown()

    perf = shared_store.get_supplier_performance(winner)
    assert perf is not None
    assert perf.total_orders == 1
    assert perf.delivery_reliability == 1.0
    assert perf.average_quality_score == 0.92

    perf_artifact = outcome_swarm.state.get_artifact("performance_MinerCorp_A")
    assert perf_artifact is not None
    outcomes = outcome_swarm.state.find_artifacts(
        kind="procurement_outcome", correlation_id="REQ-FB-02-CONV"
    )
    assert len(outcomes) == 1
    assert perf_artifact.parent_ids == [outcomes[0].id]
    assert outcomes[0].parent_ids == [result["decision"].id]
    assert outcomes[0].kind == "procurement_outcome"
    assert outcomes[0].data["supplier_id"] == winner

    # Phase 3: a future procurement evaluation is now history-adjusted
    swarm2 = build_procurement_swarm(
        request_id="REQ-FB-03", goal="future procurement", supplier_memory=shared_store
    )
    await run_procurement(swarm2)
    evaluations = swarm2.state.find_artifacts(kind="evaluation")
    by_id = {a.data["supplier_id"]: a.data for a in evaluations}

    assert by_id[winner]["history"]["applied"] is True
    assert by_id[winner]["history"]["reliability"] == 1.0
    assert by_id[winner]["history"]["adjustment"] == 0.05

    # Phase 4 baseline (no history) score for MinerCorp_A was 0.858; +5% -> 0.908
    assert by_id[winner]["score"] == 0.908
    # a supplier with no history keeps the exact Phase 4 balanced score
    assert by_id["DistribCorp_B"]["score"] == 0.795
    assert by_id["DistribCorp_B"]["history"]["applied"] is False

    assert swarm2.state.get_artifact(DECISION_ARTIFACT_NAME).data["selected_supplier"] == winner
