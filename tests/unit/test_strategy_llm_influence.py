"""Unit tests for v0.9 Step 5: Confidence-Gated LLM Influence.

Tests the consensus layer and the StrategyAgent's confidence-gated integration.
The critical invariant remains: strategy NAME and base weights never change
regardless of LLM input — influence is only applied to ``adjusted_weights``
and recorded for audit, never to the canonical decision path.
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
from swarm.domain.strategy import DEFAULT_STRATEGIES
from swarm.utils.llm_consensus import CONFIDENCE_THRESHOLD, compute_llm_consensus
from swarm.utils.llm_hash import record_llm_artifact
from swarm.utils.llm_memory import record_llm_consensus
from swarm.utils.llm_reader import get_all_llm_completions
from tests.unit.procurement_helpers import drive

CORRELATION_ID = "REQ-LLM-CONSENSUS-CONV"


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


def _requirement_event() -> Event:
    return Event(
        type=ProcurementEventType.REQUIREMENT_CREATED,
        source="requirement_agent",
        payload={"artifact": REQUIREMENT_ARTIFACT_NAME},
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


# --- Consensus layer tests ---


def test_consensus_identical_outputs_high_confidence() -> None:
    completions = [
        _completion_output(-0.05, 0.05),
        _completion_output(-0.05, 0.05),
        _completion_output(-0.05, 0.05),
    ]
    result = compute_llm_consensus(completions)
    assert result["num_completions"] == 3
    assert result["completeness"] == 1.0
    assert result["agreement_score"] >= 0.99
    assert result["confidence"] >= 0.99
    assert result["confidence"] >= CONFIDENCE_THRESHOLD
    assert "price_weight_delta" in result["aggregated_adjustments"]
    assert result["aggregated_adjustments"]["price_weight_delta"] == -0.05
    assert result["aggregated_adjustments"]["delivery_weight_delta"] == 0.05


def test_consensus_divergent_outputs_low_confidence() -> None:
    completions = [
        _completion_output(-0.1, 0.1),
        _completion_output(0.1, -0.1),
        _completion_output(0.0, 0.0),
    ]
    result = compute_llm_consensus(completions)
    assert result["agreement_score"] < 0.5
    assert result["confidence"] < CONFIDENCE_THRESHOLD
    assert result["aggregated_adjustments"] == {}


def test_consensus_missing_fields_reduces_confidence() -> None:
    """Completions without adjustments reduce completeness."""
    completions = [
        _completion_output(-0.05, 0.05),
        {"summary": "no adjustments here"},
        _completion_output(-0.05, 0.05),
    ]
    result = compute_llm_consensus(completions)
    assert result["completeness"] < 1.0
    assert result["completeness"] == pytest.approx(2 / 3, abs=0.01)


def test_consensus_malformed_outputs_ignored() -> None:
    completions: list[Any] = [
        _completion_output(-0.05, 0.05),
        "not a dict",
        None,
        {"suggested_adjustments": "not a dict"},
        {"suggested_adjustments": {"price_weight_delta": "fast"}},
    ]
    result = compute_llm_consensus(completions)
    assert result["num_completions"] == 5
    assert result["completeness"] < 0.5
    assert result["confidence"] < CONFIDENCE_THRESHOLD


def test_consensus_empty_list() -> None:
    result = compute_llm_consensus([])
    assert result["confidence"] == 0.0
    assert result["agreement_score"] == 0.0
    assert result["completeness"] == 0.0
    assert result["aggregated_adjustments"] == {}
    assert result["num_completions"] == 0


def test_consensus_single_completion_high_confidence() -> None:
    completions = [_completion_output(-0.05, 0.05)]
    result = compute_llm_consensus(completions)
    assert result["completeness"] == 1.0
    assert result["agreement_score"] == 1.0
    assert result["confidence"] >= CONFIDENCE_THRESHOLD


def test_consensus_aggregates_mean_of_values() -> None:
    completions = [
        _completion_output(-0.05, 0.05),
        _completion_output(-0.04, 0.04),
        _completion_output(-0.06, 0.06),
    ]
    result = compute_llm_consensus(completions)
    assert result["confidence"] >= CONFIDENCE_THRESHOLD
    # Mean of -0.05, -0.04, -0.06 = -0.05
    assert result["aggregated_adjustments"]["price_weight_delta"] == -0.05
    assert result["aggregated_adjustments"]["delivery_weight_delta"] == 0.05


def test_consensus_rejects_unknown_fields() -> None:
    completions = [
        {
            "suggested_adjustments": {
                "price_weight_delta": -0.05,
                "unknown_field": 0.1,
            }
        },
        {
            "suggested_adjustments": {
                "price_weight_delta": -0.05,
                "unknown_field": 0.2,
            }
        },
        {
            "suggested_adjustments": {
                "price_weight_delta": -0.05,
                "unknown_field": 0.05,
            }
        },
    ]
    result = compute_llm_consensus(completions)
    assert "unknown_field" not in result["aggregated_adjustments"]
    assert result["aggregated_adjustments"]["price_weight_delta"] == -0.05


def test_consensus_drops_non_numeric_values() -> None:
    completions: list[Any] = [
        {"suggested_adjustments": {"price_weight_delta": -0.05}},
        {"suggested_adjustments": {"price_weight_delta": "fast"}},
        {"suggested_adjustments": {"price_weight_delta": -0.05}},
    ]
    result = compute_llm_consensus(completions)
    # 2 of 3 have valid adjustments
    assert result["completeness"] == pytest.approx(2 / 3, abs=0.01)


# --- StrategyAgent with high-confidence consensus ---


@pytest.mark.asyncio
async def test_strategy_with_high_confidence_applies_adjustments() -> None:
    state = SwarmState(request_id="REQ-HIGH-CONF", goal="strat")
    _seed_balanced_requirement(state)

    # Pre-seed history: 2 stable prior rounds so temporal stability is high.
    _seed_consensus_history(state, rounds=2)

    # 3 completions with close (high-agreement) adjustments
    for variant, (p, d) in enumerate([(-0.05, 0.05), (-0.04, 0.04), (-0.06, 0.06)]):
        _seed_completion_with_output(state, _completion_output(p, d), variant=variant)

    await drive(StrategyAgent(), state, _requirement_event())

    strategy = state.get_artifact("strategy")
    assert strategy is not None

    # Strategy name unchanged
    assert strategy.data["strategy_name"] == "balanced"

    # Base weights unchanged (canonical)
    assert strategy.data["weights"] == DEFAULT_STRATEGIES["balanced"].as_weights()

    # Adjusted weights differ (bounded influence applied)
    adjusted = strategy.data["adjusted_weights"]
    assert adjusted != strategy.data["weights"]

    # Adjusted weights sum to 1.0
    assert abs(sum(adjusted.values()) - 1.0) < 1e-9
    assert all(w >= 0.0 for w in adjusted.values())
    assert all(w <= 1.0 for w in adjusted.values())

    # Consensus info recorded
    influence = strategy.data["llm_influence"]
    assert influence["used"] is True
    assert influence["adjustments_applied"] is True
    consensus = influence["llm_consensus"]
    assert consensus["confidence"] >= CONFIDENCE_THRESHOLD
    assert consensus["num_completions"] == 3


@pytest.mark.asyncio
async def test_strategy_with_low_confidence_ignores_adjustments() -> None:
    state = SwarmState(request_id="REQ-LOW-CONF", goal="strat")
    _seed_balanced_requirement(state)

    # 3 completions with divergent adjustments (low agreement)
    for variant, (p, d) in enumerate([(-0.1, 0.1), (0.1, -0.1), (0.0, 0.0)]):
        _seed_completion_with_output(state, _completion_output(p, d), variant=variant)

    await drive(StrategyAgent(), state, _requirement_event())

    strategy = state.get_artifact("strategy")
    assert strategy is not None

    # Base weights unchanged
    assert strategy.data["weights"] == DEFAULT_STRATEGIES["balanced"].as_weights()
    assert strategy.data["adjusted_weights"] == strategy.data["weights"]

    # No influence applied
    influence = strategy.data["llm_influence"]
    assert influence["used"] is False
    assert influence["adjustments_applied"] is False
    assert influence["validated_adjustments"] == {}

    consensus = influence["llm_consensus"]
    assert consensus["confidence"] < CONFIDENCE_THRESHOLD


@pytest.mark.asyncio
async def test_strategy_without_llm_has_zero_confidence() -> None:
    state = SwarmState(request_id="REQ-NO-LLM-CONF", goal="strat")
    _seed_balanced_requirement(state)
    await drive(StrategyAgent(), state, _requirement_event())

    strategy = state.get_artifact("strategy")
    assert strategy is not None
    influence = strategy.data["llm_influence"]
    assert influence["used"] is False
    assert influence["validated_adjustments"] == {}
    assert influence["llm_consensus"]["num_completions"] == 0


# --- Strategy name invariance ---


@pytest.mark.asyncio
async def test_strategy_name_invariant_under_high_confidence_influence() -> None:
    """Even with accepted adjustments, strategy name must not change."""
    for strat_name, constraints in [
        (
            "balanced",
            {
                "material": "aluminum",
                "quantity": 1000,
                "budget": 2_000_000.0,
                "max_unit_price": 2640.0,
                "max_carbon_per_unit": None,
            },
        ),
        (
            "low_carbon",
            {
                "material": "aluminum",
                "quantity": 1000,
                "budget": 2_000_000.0,
                "max_unit_price": 2640.0,
                "max_carbon_per_unit": 800.0,
            },
        ),
        (
            "cost_optimized",
            {
                "material": "aluminum",
                "quantity": 1000,
                "budget": 500_000.0,
                "max_unit_price": 2640.0,
                "max_carbon_per_unit": None,
            },
        ),
    ]:
        state = SwarmState(request_id=f"REQ-{strat_name}-INV", goal="strat")
        state.put_artifact(
            RequirementArtifact(
                data={
                    "text": "buy aluminum",
                    "constraints": constraints,
                    "metadata": {},
                },
                created_by="requirement_agent",
                correlation_id=CORRELATION_ID,
            )
        )
        for variant, (p, d) in enumerate([(-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05)]):
            _seed_completion_with_output(state, _completion_output(p, d), variant=variant)

        await drive(StrategyAgent(), state, _requirement_event())

        strategy = state.get_artifact("strategy")
        assert strategy is not None
        assert strategy.data["strategy_name"] == strat_name


# --- Deterministic replay ---


@pytest.mark.asyncio
async def test_replay_produces_identical_llm_influence() -> None:
    """Replaying the same run must produce identical influence records."""

    def _build_state() -> SwarmState:
        state = SwarmState(request_id="REQ-REPLAY-CONF", goal="strat")
        _seed_balanced_requirement(state)
        for variant, (p, d) in enumerate([(-0.05, 0.05), (-0.04, 0.04), (-0.06, 0.06)]):
            _seed_completion_with_output(state, _completion_output(p, d), variant=variant)
        return state

    state1 = _build_state()
    await drive(StrategyAgent(), state1, _requirement_event())

    state2 = _build_state()
    await drive(StrategyAgent(), state2, _requirement_event())

    s1 = state1.get_artifact("strategy")
    s2 = state2.get_artifact("strategy")
    assert s1 is not None
    assert s2 is not None

    # All influence fields must be identical (timestamps excluded by design)
    assert s1.data["strategy_name"] == s2.data["strategy_name"]
    assert s1.data["weights"] == s2.data["weights"]
    assert s1.data["adjusted_weights"] == s2.data["adjusted_weights"]
    assert s1.data["llm_influence"] == s2.data["llm_influence"]


# --- No authority leakage ---


@pytest.mark.asyncio
async def test_confidence_gating_does_not_trigger_connectors() -> None:
    state = SwarmState(request_id="REQ-NOCONN-CONF", goal="strat")
    _seed_balanced_requirement(state)
    for variant, (p, d) in enumerate([(-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05)]):
        _seed_completion_with_output(state, _completion_output(p, d), variant=variant)

    await drive(StrategyAgent(), state, _requirement_event())

    assert len(state.find_artifacts(kind="external_call")) == 0


@pytest.mark.asyncio
async def test_confidence_gating_publishes_only_strategy_selected() -> None:
    state = SwarmState(request_id="REQ-EVENTS-CONF", goal="strat")
    _seed_balanced_requirement(state)
    for variant, (p, d) in enumerate([(-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05)]):
        _seed_completion_with_output(state, _completion_output(p, d), variant=variant)

    bus = EventBus()
    published: list[Event] = []

    async def capture(event: Event) -> None:
        published.append(event)

    bus.subscribe(ProcurementEventType.STRATEGY_SELECTED, capture)
    await drive(StrategyAgent(), state, _requirement_event(), bus=bus)

    assert len(published) == 1
    assert published[0].type == ProcurementEventType.STRATEGY_SELECTED


# --- Timeline ---


@pytest.mark.asyncio
async def test_timeline_shows_consensus_and_influence() -> None:
    state = SwarmState(request_id="REQ-TL-CONF", goal="strat")
    _seed_balanced_requirement(state)
    for variant, (p, d) in enumerate([(-0.05, 0.05), (-0.04, 0.04), (-0.06, 0.06)]):
        _seed_completion_with_output(state, _completion_output(p, d), variant=variant)

    await drive(StrategyAgent(), state, _requirement_event())

    timeline = build_timeline(state)

    llm_items = [i for i in timeline.timeline if i.subtype == "llm"]
    strat_items = [i for i in timeline.timeline if i.subtype == "strategy"]
    assert len(llm_items) == 3  # 3 completions
    assert len(strat_items) == 1

    for item in llm_items:
        assert item.phase == "cognitive"
    assert strat_items[0].phase == "discovery"


# --- get_all_llm_completions tests ---


def test_get_all_llm_completions_returns_all_variants() -> None:
    state = SwarmState(request_id="REQ-GETALL-1", goal="strat")
    for variant, (p, d) in enumerate([(-0.05, 0.05), (-0.04, 0.04), (-0.06, 0.06)]):
        _seed_completion_with_output(state, _completion_output(p, d), variant=variant)

    all_completions = get_all_llm_completions(state, correlation_id=CORRELATION_ID)
    assert len(all_completions) == 3


def test_get_all_llm_completions_empty_when_no_llm() -> None:
    state = SwarmState(request_id="REQ-GETALL-2", goal="strat")
    assert get_all_llm_completions(state) == []


def test_get_all_llm_completions_ignores_prompts() -> None:
    state = SwarmState(request_id="REQ-GETALL-3", goal="strat")
    _seed_completion_with_output(state, _completion_output(-0.05, 0.05), variant=0)
    record_llm_artifact(
        state,
        model="stub",
        prompt="test",
        parameters={"payload": {}},
        kind="llm_prompt",
        correlation_id=CORRELATION_ID,
        by="test",
    )

    completions = get_all_llm_completions(state, correlation_id=CORRELATION_ID)
    assert len(completions) == 1  # only the completion, not the prompt


# --- Integration with SupplierAnalysisLLMAgent ---


@pytest.mark.asyncio
async def test_strategy_reads_multi_variant_completions() -> None:
    """The StrategyAgent can read and aggregate multi-variant completions."""
    state = SwarmState(request_id="REQ-MULTI-VARIANT", goal="strat")
    _seed_balanced_requirement(state)

    # Pre-seed stable history so trust can exceed threshold.
    _seed_consensus_history(state, rounds=2)

    for variant, (p, d) in enumerate([(-0.05, 0.05), (-0.04, 0.04), (-0.06, 0.06)]):
        _seed_completion_with_output(state, _completion_output(p, d), variant=variant)

    await drive(StrategyAgent(), state, _requirement_event())

    strategy = state.get_artifact("strategy")
    assert strategy is not None
    assert strategy.data["strategy_name"] == "balanced"
    influence = strategy.data["llm_influence"]
    assert influence["used"] is True
    assert influence["llm_consensus"]["num_completions"] == 3
    assert influence["llm_consensus"]["confidence"] >= CONFIDENCE_THRESHOLD
    assert influence["adjustments_applied"] is True
