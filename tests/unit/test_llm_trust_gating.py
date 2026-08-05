"""Unit tests for v0.9 Step 6: Trust-Gated LLM Influence.

Tests the StrategyAgent's temporal stability + trust scoring integration:
- Stable history → high trust → adjustments applied
- Divergent history → low trust → adjustments rejected
- Single datapoint → zero stability → no influence
- Replay produces identical trust scores
"""

from typing import Any

import pytest

from swarm import Event, EventBus, SwarmState
from swarm.core.timeline import build_timeline
from swarm.domain import ProcurementEventType, StrategyAgent
from swarm.domain.artifacts import (
    REQUIREMENT_ARTIFACT_NAME,
    RequirementArtifact,
)
from swarm.utils.llm_hash import record_llm_artifact
from swarm.utils.llm_memory import get_llm_consensus_history, record_llm_consensus
from swarm.utils.llm_stability import TRUST_THRESHOLD
from tests.unit.procurement_helpers import drive

CORRELATION_ID = "REQ-TRUST-GATING-CONV"


def _requirement_event() -> Event:
    return Event(
        type=ProcurementEventType.REQUIREMENT_CREATED,
        source="requirement_agent",
        payload={"artifact": REQUIREMENT_ARTIFACT_NAME},
        correlation_id=CORRELATION_ID,
    )


def _quotes_completed_event() -> Event:
    return Event(
        type=ProcurementEventType.QUOTES_COMPLETED,
        source="completion_tracker",
        payload={},
        correlation_id=CORRELATION_ID,
    )


def _seed_balanced_requirement(state: SwarmState) -> None:
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
                    "max_carbon_per_unit": None,
                },
                "metadata": {},
            },
            created_by="requirement_agent",
            correlation_id=CORRELATION_ID,
        )
    )


def _seed_completion_with_output(
    state: SwarmState,
    output: dict[str, Any],
    variant: int = 0,
) -> None:
    record_llm_artifact(
        state,
        model="stub",
        prompt="Analyze suppliers",
        parameters={"payload": {}},
        output=output,
        kind="llm_completion",
        variant=variant,
        correlation_id=CORRELATION_ID,
        by="supplier_analysis_llm_agent",
    )


def _completion_output(
    price_delta: float,
    delivery_delta: float,
    summary: str = "ok",
) -> dict[str, Any]:
    return {
        "summary": summary,
        "risks": [],
        "tradeoffs": [],
        "suggested_adjustments": {
            "price_weight_delta": price_delta,
            "delivery_weight_delta": delivery_delta,
        },
    }


def _seed_consensus_history(
    state: SwarmState,
    *,
    rounds: int = 2,
    price_delta: float = -0.05,
    delivery_delta: float = 0.05,
    correlation_id: str = CORRELATION_ID,
) -> None:
    """Pre-seed consensus history artifacts to simulate prior rounds."""
    for round_num in range(1, rounds + 1):
        consensus = {
            "confidence": 0.95,
            "agreement_score": 0.99,
            "completeness": 1.0,
            "num_completions": 3,
            "aggregated_adjustments": {
                "price_weight_delta": price_delta,
                "delivery_weight_delta": delivery_delta,
            },
        }
        record_llm_consensus(
            state,
            correlation_id=correlation_id,
            consensus=consensus,
            round_number=round_num,
            by="test",
        )


# --- Trust gating: high stability ---


@pytest.mark.asyncio
async def test_stable_history_enables_trust_and_adjustments() -> None:
    """With stable prior history + high-confidence current consensus → trust ≥ 0.7."""
    state = SwarmState(request_id="REQ-STABLE-TRUST", goal="strat")
    _seed_balanced_requirement(state)

    # Pre-seed 2 stable rounds (same adjustments)
    _seed_consensus_history(state, rounds=2, price_delta=-0.05, delivery_delta=0.05)

    # Current completions also stable
    for variant in range(3):
        _seed_completion_with_output(state, _completion_output(-0.05, 0.05), variant=variant)

    await drive(StrategyAgent(), state, _requirement_event())

    strategy = state.get_artifact("strategy")
    assert strategy is not None

    influence = strategy.data["llm_influence"]
    trust = strategy.data["llm_trust"]

    assert trust["stability"] == 1.0
    assert trust["trust_score"] >= TRUST_THRESHOLD
    assert influence["adjustments_applied"] is True


# --- Trust gating: low stability ---


@pytest.mark.asyncio
async def test_divergent_history_blocks_trust_despite_high_confidence() -> None:
    """High confidence consensus but divergent history → trust < 0.7 → no adjustments."""
    state = SwarmState(request_id="REQ-DIVERGENT-TRUST", goal="strat")
    _seed_balanced_requirement(state)

    # Pre-seed divergent rounds: round 1 has price -0.1, round 2 has price +0.1
    _seed_consensus_history(state, rounds=1, price_delta=-0.1, delivery_delta=0.1)
    record_llm_consensus(
        state,
        correlation_id=CORRELATION_ID,
        consensus={
            "confidence": 0.95,
            "agreement_score": 0.99,
            "completeness": 1.0,
            "num_completions": 3,
            "aggregated_adjustments": {
                "price_weight_delta": 0.10,
                "delivery_weight_delta": -0.10,
            },
        },
        round_number=2,
        by="test",
    )

    # Current completions are high-confidence but differ from history
    for variant in range(3):
        _seed_completion_with_output(state, _completion_output(-0.05, 0.05), variant=variant)

    await drive(StrategyAgent(), state, _requirement_event())

    strategy = state.get_artifact("strategy")
    assert strategy is not None

    trust = strategy.data["llm_trust"]

    # Drift: round 1 (-0.1) → round 2 (+0.1) → round 3 (-0.05)
    # Each transition has significant drift → low stability
    assert trust["stability"] < 0.5
    assert trust["trust_score"] < TRUST_THRESHOLD
    influence = strategy.data["llm_influence"]
    assert influence["adjustments_applied"] is False
    assert strategy.data["adjusted_weights"] == strategy.data["weights"]


# --- Trust gating: single datapoint ---


@pytest.mark.asyncio
async def test_single_consensus_record_has_zero_stability() -> None:
    """First round has no history → stability=0 → trust=0 → no adjustments."""
    state = SwarmState(request_id="REQ-SINGLE-TRUST", goal="strat")
    _seed_balanced_requirement(state)

    # Only completions, no pre-seeded history
    for variant in range(3):
        _seed_completion_with_output(state, _completion_output(-0.05, 0.05), variant=variant)

    await drive(StrategyAgent(), state, _requirement_event())

    strategy = state.get_artifact("strategy")
    assert strategy is not None

    trust = strategy.data["llm_trust"]
    assert trust["stability"] == 0.0
    assert trust["trust_score"] == 0.0
    assert trust["history_depth"] >= 1
    influence = strategy.data["llm_influence"]
    assert influence["adjustments_applied"] is False


# --- Trust gating: no completions ---


@pytest.mark.asyncio
async def test_no_completions_zero_trust() -> None:
    """No LLM completions → confidence=0, stability=0, trust=0."""
    state = SwarmState(request_id="REQ-NOCOMP-TRUST", goal="strat")
    _seed_balanced_requirement(state)

    await drive(StrategyAgent(), state, _requirement_event())

    strategy = state.get_artifact("strategy")
    assert strategy is not None

    trust = strategy.data["llm_trust"]
    assert trust["confidence"] == 0.0
    assert trust["stability"] == 0.0
    assert trust["trust_score"] == 0.0
    assert trust["history_depth"] == 0


# --- Temporal re-evaluation on QuotesCompleted ---


@pytest.mark.asyncio
async def test_re_evaluation_on_quotes_completed_updates_trust() -> None:
    """StrategyAgent re-evaluates on QuotesCompleted and updates trust score."""
    state = SwarmState(request_id="REQ-REEVAL-TRUST", goal="strat")
    _seed_balanced_requirement(state)

    # Pre-seed 2 stable rounds
    _seed_consensus_history(state, rounds=2)

    # Seed completions before the first run (simulating prior LLM analysis)
    for variant in range(3):
        _seed_completion_with_output(state, _completion_output(-0.05, 0.05), variant=variant)

    # Initial: RequirementCreated
    await drive(StrategyAgent(), state, _requirement_event())

    strategy1 = state.get_artifact("strategy")
    assert strategy1 is not None
    # After recording current consensus, history_depth should be 3 (2 pre-seeded + 1 current)
    assert strategy1.data["llm_trust"]["history_depth"] == 3
    assert strategy1.data["llm_trust"]["trust_score"] >= TRUST_THRESHOLD


@pytest.mark.asyncio
async def test_re_evaluation_does_not_publish_strategy_selected() -> None:
    """Re-evaluation on QuotesCompleted should NOT emit StrategySelected."""
    state = SwarmState(request_id="REQ-NOPROTECT-TRUST", goal="strat")
    _seed_balanced_requirement(state)
    _seed_consensus_history(state, rounds=2)

    await drive(StrategyAgent(), state, _requirement_event())

    # Capture events on the bus
    bus = EventBus()
    published: list[Event] = []

    async def capture(event: Event) -> None:
        published.append(event)

    bus.subscribe(ProcurementEventType.STRATEGY_SELECTED, capture)

    for variant in range(3):
        _seed_completion_with_output(state, _completion_output(-0.05, 0.05), variant=variant)

    # Drive the SAME agent instance on QuotesCompleted
    agent = StrategyAgent()
    agent.bus = bus
    agent.state = state
    await agent.step(_quotes_completed_event())

    # Only the initial StrategySelected from the first drive
    strat_events = [e for e in published if e.type == ProcurementEventType.STRATEGY_SELECTED]
    assert len(strat_events) == 0  # re-eval doesn't publish


# --- Replay safety ---


@pytest.mark.asyncio
async def test_replay_produces_identical_trust_scores() -> None:
    """Replaying the same sequence must produce identical trust records."""

    def _build_state() -> SwarmState:
        state = SwarmState(request_id="REQ-REPLAY-TRUST", goal="strat")
        _seed_balanced_requirement(state)
        _seed_consensus_history(state, rounds=2)
        for variant in range(3):
            _seed_completion_with_output(state, _completion_output(-0.05, 0.05), variant=variant)
        return state

    state1 = _build_state()
    await drive(StrategyAgent(), state1, _requirement_event())

    state2 = _build_state()
    await drive(StrategyAgent(), state2, _requirement_event())

    s1 = state1.get_artifact("strategy")
    s2 = state2.get_artifact("strategy")
    assert s1 is not None
    assert s2 is not None

    assert s1.data["llm_trust"] == s2.data["llm_trust"]
    assert s1.data["llm_influence"] == s2.data["llm_influence"]


# --- Timeline integration ---


@pytest.mark.asyncio
async def test_timeline_shows_consensus_history_artifacts() -> None:
    """Consensus history artifacts appear in the timeline under cognitive phase."""
    state = SwarmState(request_id="REQ-TL-TRUST", goal="strat")
    _seed_balanced_requirement(state)
    _seed_consensus_history(state, rounds=2)

    for variant in range(3):
        _seed_completion_with_output(state, _completion_output(-0.05, 0.05), variant=variant)

    await drive(StrategyAgent(), state, _requirement_event())

    timeline = build_timeline(state)

    cognitive_items = [i for i in timeline.timeline if i.phase == "cognitive"]
    # 3 LLM completions + 2 pre-seeded consensus + 1 new consensus = 6 cognitive artifacts
    assert len(cognitive_items) >= 5

    consensus_items = [i for i in timeline.timeline if i.subtype == "llm_consensus"]
    assert len(consensus_items) == 3

    for item in consensus_items:
        assert item.phase == "cognitive"


# --- Memory module tests ---


def test_record_and_retrieve_consensus_history() -> None:
    """record_llm_consensus + get_llm_consensus_history round-trip."""
    state = SwarmState(request_id="REQ-MEMORY-TEST", goal="strat")

    consensus1 = {
        "confidence": 0.9,
        "agreement_score": 0.95,
        "completeness": 1.0,
        "num_completions": 3,
        "aggregated_adjustments": {"price_weight_delta": -0.05},
    }
    consensus2 = {
        "confidence": 0.85,
        "agreement_score": 0.90,
        "completeness": 1.0,
        "num_completions": 3,
        "aggregated_adjustments": {"price_weight_delta": -0.04},
    }

    record_llm_consensus(state, correlation_id="cid", consensus=consensus1, round_number=1)
    record_llm_consensus(state, correlation_id="cid", consensus=consensus2, round_number=2)

    history = get_llm_consensus_history(state, correlation_id="cid")
    assert len(history) == 2
    assert history[0]["round"] == 1
    assert history[1]["round"] == 2
    assert history[0]["confidence"] == 0.9
    assert history[1]["aggregated_adjustments"]["price_weight_delta"] == -0.04


def test_get_consensus_history_limit() -> None:
    """get_llm_consensus_history respects the limit parameter."""
    state = SwarmState(request_id="REQ-LIMIT-TEST", goal="strat")

    for i in range(1, 8):
        record_llm_consensus(
            state,
            correlation_id="cid",
            consensus={"confidence": 0.9, "aggregated_adjustments": {}},
            round_number=i,
        )

    history = get_llm_consensus_history(state, correlation_id="cid", limit=5)
    assert len(history) == 5
    # Most recent 5 rounds: 3, 4, 5, 6, 7
    rounds = [h["round"] for h in history]
    assert rounds == [3, 4, 5, 6, 7]


def test_get_consensus_history_empty() -> None:
    """No history → empty list."""
    state = SwarmState(request_id="REQ-EMPTY-HIST", goal="strat")
    history = get_llm_consensus_history(state, correlation_id="nonexistent")
    assert history == []


def test_get_consensus_history_ignores_other_correlation_ids() -> None:
    """History is scoped to the correlation_id."""
    state = SwarmState(request_id="REQ-SCOPED", goal="strat")

    record_llm_consensus(
        state, correlation_id="cid1", consensus={"confidence": 0.9}, round_number=1
    )
    record_llm_consensus(
        state, correlation_id="cid2", consensus={"confidence": 0.8}, round_number=1
    )

    history1 = get_llm_consensus_history(state, correlation_id="cid1")
    history2 = get_llm_consensus_history(state, correlation_id="cid2")
    assert len(history1) == 1
    assert len(history2) == 1
    assert history1[0]["confidence"] == 0.9
    assert history2[0]["confidence"] == 0.8
