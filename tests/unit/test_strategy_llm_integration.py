"""Unit tests for StrategyAgent LLM advisory integration (v0.9 Step 3).

Verifies the critical invariant: LLM context is attached as advisory only
and does NOT influence strategy selection, scoring weights, or any
authoritative path. The strategy outcome is identical with or without LLM
artifacts present.
"""

import pytest

from swarm import Event, EventBus, SwarmState
from swarm.core.timeline import build_timeline
from swarm.domain import ProcurementEventType, StrategyAgent
from swarm.domain.artifacts import (
    REQUIREMENT_ARTIFACT_NAME,
    RequirementArtifact,
)
from swarm.domain.strategy import DEFAULT_STRATEGIES
from swarm.utils.llm_hash import record_llm_artifact
from swarm.utils.llm_reader import get_latest_llm_completion
from tests.unit.procurement_helpers import drive

CORRELATION_ID = "REQ-STRAT-LLM-CONV"


def _requirement_event() -> Event:
    return Event(
        type=ProcurementEventType.REQUIREMENT_CREATED,
        source="requirement_agent",
        payload={"artifact": REQUIREMENT_ARTIFACT_NAME},
        correlation_id=CORRELATION_ID,
    )


def _seed_low_carbon_requirement(state: SwarmState) -> None:
    state.put_artifact(
        RequirementArtifact(
            data={
                "text": "buy aluminum with strict carbon constraints",
                "constraints": {
                    "material": "aluminum",
                    "quantity": 1000,
                    "budget": 2_000_000.0,
                    "max_unit_price": 2640.0,
                    "target_lead_time_days": 30,
                    "max_carbon_per_unit": 800.0,
                },
                "metadata": {},
            },
            created_by="requirement_agent",
            correlation_id=CORRELATION_ID,
        )
    )


def _seed_balanced_requirement(state: SwarmState) -> None:
    state.put_artifact(
        RequirementArtifact(
            data={
                "text": "buy aluminum, relaxed constraints",
                "constraints": {
                    "material": "aluminum",
                    "quantity": 1000,
                    "budget": 2_000_000.0,
                    "max_unit_price": 2640.0,
                    "target_lead_time_days": 30,
                    "max_carbon_per_unit": None,
                },
                "metadata": {},
            },
            created_by="requirement_agent",
            correlation_id=CORRELATION_ID,
        )
    )


def _seed_cost_optimized_requirement(state: SwarmState) -> None:
    state.put_artifact(
        RequirementArtifact(
            data={
                "text": "buy aluminum on a tight budget",
                "constraints": {
                    "material": "aluminum",
                    "quantity": 1000,
                    "budget": 500_000.0,
                    "max_unit_price": 2640.0,
                    "target_lead_time_days": 30,
                    "max_carbon_per_unit": None,
                },
                "metadata": {},
            },
            created_by="requirement_agent",
            correlation_id=CORRELATION_ID,
        )
    )


def _seed_llm_completion(state: SwarmState) -> None:
    """Pre-seed an LLM completion artifact in state (simulating prior analysis)."""
    record_llm_artifact(
        state,
        model="stub",
        prompt="Analyze suppliers",
        parameters={"payload": {}},
        output={
            "summary": "Risk assessment complete",
            "risks": [
                {"supplier_id": "MinerCorp_A", "risk_factors": ["price_variance"]},
                {"supplier_id": "DistribCorp_B", "risk_factors": ["delivery_delay"]},
                {"supplier_id": "RecycleCorp_C", "risk_factors": ["quality_concern"]},
                {"supplier_id": "TraderCorp_D", "risk_factors": ["capacity_constraint"]},
            ],
            "tradeoffs": [
                "Lower price may imply higher delivery risk",
                "Higher reliability suppliers charge premium pricing",
                "Carbon-optimized suppliers have limited capacity",
            ],
        },
        kind="llm_completion",
        correlation_id=CORRELATION_ID,
        by="supplier_analysis_llm_agent",
    )


# --- LLM reader tests ---


def test_get_latest_llm_completion_returns_none_when_absent() -> None:
    state = SwarmState(request_id="REQ-READER-1", goal="llm")
    assert get_latest_llm_completion(state) is None
    assert get_latest_llm_completion(state, correlation_id="anything") is None


def test_get_latest_llm_completion_returns_output_when_present() -> None:
    state = SwarmState(request_id="REQ-READER-2", goal="llm")
    record_llm_artifact(
        state,
        model="stub",
        prompt="test",
        parameters={"payload": {}},
        output={"summary": "ok", "risks": [], "tradeoffs": []},
        kind="llm_completion",
        correlation_id="CONV-1",
        by="test",
    )
    result = get_latest_llm_completion(state, correlation_id="CONV-1")
    assert result is not None
    assert result["summary"] == "ok"


def test_get_latest_llm_completion_filters_by_correlation_id() -> None:
    state = SwarmState(request_id="REQ-READER-3", goal="llm")
    record_llm_artifact(
        state,
        model="stub",
        prompt="test_a",
        parameters={"payload": {}},
        output={"summary": "conv1", "risks": [], "tradeoffs": []},
        kind="llm_completion",
        correlation_id="CONV-1",
        by="test",
    )
    record_llm_artifact(
        state,
        model="stub",
        prompt="test_b",
        parameters={"payload": {}},
        output={"summary": "conv2", "risks": [], "tradeoffs": []},
        kind="llm_completion",
        correlation_id="CONV-2",
        by="test",
    )
    result1 = get_latest_llm_completion(state, correlation_id="CONV-1")
    result2 = get_latest_llm_completion(state, correlation_id="CONV-2")
    result_all = get_latest_llm_completion(state)
    assert result1 is not None
    assert result2 is not None
    assert result_all is not None
    assert result1["summary"] == "conv1"
    assert result2["summary"] == "conv2"
    assert result_all["summary"] == "conv2"


def test_get_latest_llm_completion_ignores_prompt_artifacts() -> None:
    state = SwarmState(request_id="REQ-READER-4", goal="llm")
    record_llm_artifact(
        state,
        model="stub",
        prompt="test",
        parameters={"payload": {}},
        kind="llm_prompt",
        correlation_id="CONV-1",
        by="test",
    )
    assert get_latest_llm_completion(state, correlation_id="CONV-1") is None


def test_get_latest_llm_completion_returns_latest_when_multiple() -> None:
    state = SwarmState(request_id="REQ-READER-5", goal="llm")
    record_llm_artifact(
        state,
        model="stub",
        prompt="first",
        parameters={"payload": {}},
        output={"summary": "first", "risks": [], "tradeoffs": []},
        kind="llm_completion",
        correlation_id="CONV-1",
        by="test",
    )
    record_llm_artifact(
        state,
        model="stub",
        prompt="second",
        parameters={"payload": {}},
        output={"summary": "second", "risks": [], "tradeoffs": []},
        kind="llm_completion",
        correlation_id="CONV-1",
        by="test",
    )
    result = get_latest_llm_completion(state, correlation_id="CONV-1")
    assert result is not None
    assert result["summary"] == "second"


# --- StrategyAgent: LLM absence produces unchanged behavior ---


@pytest.mark.asyncio
async def test_strategy_without_llm_has_llm_context_used_false() -> None:
    state = SwarmState(request_id="REQ-STRAT-1", goal="strategy")
    _seed_low_carbon_requirement(state)
    agent = StrategyAgent()
    await drive(agent, state, _requirement_event())

    strategy = state.get_artifact("strategy")
    assert strategy is not None
    assert strategy.data["llm_context"]["used"] is False
    assert strategy.data["strategy_name"] == "low_carbon"


@pytest.mark.asyncio
async def test_strategy_without_llm_selects_low_carbon() -> None:
    state = SwarmState(request_id="REQ-STRAT-2", goal="strategy")
    _seed_low_carbon_requirement(state)
    agent = StrategyAgent()
    await drive(agent, state, _requirement_event())

    strategy = state.get_artifact("strategy")
    assert strategy is not None
    assert strategy.data["strategy_name"] == "low_carbon"
    assert strategy.data["weights"] == DEFAULT_STRATEGIES["low_carbon"].as_weights()


# --- StrategyAgent: LLM presence does NOT change strategy outcome ---


@pytest.mark.asyncio
async def test_strategy_with_llm_same_outcome_low_carbon() -> None:
    """LLM presence must not change the strategy selection."""
    # Without LLM
    state_no_llm = SwarmState(request_id="REQ-STRAT-NO-LLM", goal="strategy")
    _seed_low_carbon_requirement(state_no_llm)
    agent1 = StrategyAgent()
    await drive(agent1, state_no_llm, _requirement_event())
    strat_no_llm = state_no_llm.get_artifact("strategy")

    # With LLM (pre-seeded at same correlation_id)
    state_with_llm = SwarmState(request_id="REQ-STRAT-WITH-LLM", goal="strategy")
    _seed_low_carbon_requirement(state_with_llm)
    _seed_llm_completion(state_with_llm)
    agent2 = StrategyAgent()
    await drive(agent2, state_with_llm, _requirement_event())
    strat_with_llm = state_with_llm.get_artifact("strategy")

    assert strat_no_llm is not None
    assert strat_with_llm is not None

    # Critical invariant: strategy_name and weights are identical
    assert strat_no_llm.data["strategy_name"] == strat_with_llm.data["strategy_name"]
    assert strat_no_llm.data["weights"] == strat_with_llm.data["weights"]
    assert strat_with_llm.data["strategy_name"] == "low_carbon"

    # LLM context is attached as advisory
    assert strat_with_llm.data["llm_context"]["used"] is True
    assert len(strat_with_llm.data["llm_context"]["risk_hints"]) == 3
    assert len(strat_with_llm.data["llm_context"]["tradeoff_hints"]) == 3

    # Without LLM, no advisory context
    assert strat_no_llm.data["llm_context"]["used"] is False


@pytest.mark.asyncio
async def test_strategy_with_llm_same_outcome_balanced() -> None:
    """Balanced strategy must not change with LLM present."""
    # Without LLM
    state_no_llm = SwarmState(request_id="REQ-STRAT-NO-LLM-B", goal="strategy")
    _seed_balanced_requirement(state_no_llm)
    agent1 = StrategyAgent()
    await drive(agent1, state_no_llm, _requirement_event())
    strat_no_llm = state_no_llm.get_artifact("strategy")

    # With LLM
    state_with_llm = SwarmState(request_id="REQ-STRAT-WITH-LLM-B", goal="strategy")
    _seed_balanced_requirement(state_with_llm)
    _seed_llm_completion(state_with_llm)
    agent2 = StrategyAgent()
    await drive(agent2, state_with_llm, _requirement_event())
    strat_with_llm = state_with_llm.get_artifact("strategy")

    assert strat_no_llm is not None
    assert strat_with_llm is not None

    assert strat_no_llm.data["strategy_name"] == strat_with_llm.data["strategy_name"]
    assert strat_no_llm.data["weights"] == strat_with_llm.data["weights"]
    assert strat_with_llm.data["strategy_name"] == "balanced"
    assert strat_with_llm.data["llm_context"]["used"] is True


@pytest.mark.asyncio
async def test_strategy_with_llm_same_outcome_cost_optimized() -> None:
    """Cost-optimized strategy must not change with LLM present."""
    # Without LLM
    state_no_llm = SwarmState(request_id="REQ-STRAT-NO-LLM-CO", goal="strategy")
    _seed_cost_optimized_requirement(state_no_llm)
    agent1 = StrategyAgent()
    await drive(agent1, state_no_llm, _requirement_event())
    strat_no_llm = state_no_llm.get_artifact("strategy")

    # With LLM
    state_with_llm = SwarmState(request_id="REQ-STRAT-WITH-LLM-CO", goal="strategy")
    _seed_cost_optimized_requirement(state_with_llm)
    _seed_llm_completion(state_with_llm)
    agent2 = StrategyAgent()
    await drive(agent2, state_with_llm, _requirement_event())
    strat_with_llm = state_with_llm.get_artifact("strategy")

    assert strat_no_llm is not None
    assert strat_with_llm is not None

    assert strat_no_llm.data["strategy_name"] == strat_with_llm.data["strategy_name"]
    assert strat_no_llm.data["weights"] == strat_with_llm.data["weights"]
    assert strat_with_llm.data["strategy_name"] == "cost_optimized"


# --- StrategyAgent: malformed LLM output is safely ignored ---


@pytest.mark.asyncio
async def test_strategy_with_malformed_llm_output_is_safe() -> None:
    """Malformed LLM output must not crash the agent or change strategy."""
    state = SwarmState(request_id="REQ-STRAT-MALFORMED", goal="strategy")
    _seed_low_carbon_requirement(state)
    record_llm_artifact(
        state,
        model="stub",
        prompt="bad output",
        parameters={"payload": {}},
        output={"summary": "broken", "risks": "not_a_list", "tradeoffs": None},
        kind="llm_completion",
        correlation_id=CORRELATION_ID,
        by="test",
    )
    agent = StrategyAgent()
    await drive(agent, state, _requirement_event())

    strategy = state.get_artifact("strategy")
    assert strategy is not None
    assert strategy.data["strategy_name"] == "low_carbon"
    assert strategy.data["llm_context"]["used"] is True
    assert strategy.data["llm_context"]["risk_hints"] == []
    assert strategy.data["llm_context"]["tradeoff_hints"] == []


@pytest.mark.asyncio
async def test_strategy_with_non_dict_output_is_safe() -> None:
    """Non-dict LLM output must be safely ignored by get_latest_llm_completion."""
    state = SwarmState(request_id="REQ-STRAT-NONDICT", goal="strategy")
    _seed_low_carbon_requirement(state)
    record_llm_artifact(
        state,
        model="stub",
        prompt="string output",
        parameters={"payload": {}},
        output="just a string",
        kind="llm_completion",
        correlation_id=CORRELATION_ID,
        by="test",
    )
    agent = StrategyAgent()
    await drive(agent, state, _requirement_event())

    strategy = state.get_artifact("strategy")
    assert strategy is not None
    assert strategy.data["strategy_name"] == "low_carbon"
    assert strategy.data["llm_context"]["used"] is False


# --- LLM does not trigger external calls ---


@pytest.mark.asyncio
async def test_strategy_with_llm_does_not_call_connectors() -> None:
    state = SwarmState(request_id="REQ-STRAT-NO-CONN", goal="strategy")
    _seed_low_carbon_requirement(state)
    _seed_llm_completion(state)
    agent = StrategyAgent()
    await drive(agent, state, _requirement_event())

    external_calls = state.find_artifacts(kind="external_call")
    assert len(external_calls) == 0


# --- LLM does not publish events ---


@pytest.mark.asyncio
async def test_strategy_with_llm_publishes_only_strategy_selected() -> None:
    state = SwarmState(request_id="REQ-STRAT-EVENTS", goal="strategy")
    _seed_low_carbon_requirement(state)
    _seed_llm_completion(state)

    bus = EventBus()
    published: list[Event] = []

    async def capture(event: Event) -> None:
        published.append(event)

    bus.subscribe(ProcurementEventType.STRATEGY_SELECTED, capture)

    agent = StrategyAgent()
    await drive(agent, state, _requirement_event(), bus=bus)

    assert len(published) == 1
    assert published[0].type == ProcurementEventType.STRATEGY_SELECTED


# --- Timeline: LLM artifacts appear under cognitive phase ---


@pytest.mark.asyncio
async def test_strategy_agent_timeline_shows_cognitive_then_decision() -> None:
    state = SwarmState(request_id="REQ-STRAT-TIMELINE", goal="strategy")
    _seed_low_carbon_requirement(state)
    _seed_llm_completion(state)
    agent = StrategyAgent()
    await drive(agent, state, _requirement_event())

    timeline = build_timeline(state)

    subtypes = {item.subtype for item in timeline.timeline}
    assert "llm" in subtypes
    assert "strategy" in subtypes

    for item in timeline.timeline:
        if item.subtype == "llm":
            assert item.phase == "cognitive"
        if item.subtype == "strategy":
            assert item.phase == "discovery"


# --- Replay handling ---


@pytest.mark.asyncio
async def test_strategy_agent_ignores_replayed_events() -> None:
    state = SwarmState(request_id="REQ-STRAT-REPLAY", goal="strategy")
    _seed_low_carbon_requirement(state)
    agent = StrategyAgent()
    event = _requirement_event().model_copy(update={"replayed": True})
    agent.state = state
    await agent.step(event)

    assert state.get_artifact("strategy") is None
