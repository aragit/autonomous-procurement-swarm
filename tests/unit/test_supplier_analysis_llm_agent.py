"""Unit tests for SupplierAnalysisLLMAgent (v0.9 Step 2).

Verifies that the agent is read-only: it records LLMArtifacts without
side effects, is deterministic, and leaves the execution/governance flow
untouched.
"""

import pytest

from swarm import Event, EventBus, SwarmState
from swarm.core.timeline import build_timeline
from swarm.domain import (
    ProcurementEventType,
    SupplierDiscoveryAgent,
)
from swarm.domain.agents import (
    EvaluationAgent,
    NegotiationAgent,
    SupplierAnalysisLLMAgent,
)
from swarm.domain.artifacts import (
    REQUIREMENT_ARTIFACT_NAME,
    RequirementArtifact,
)
from tests.unit.procurement_helpers import drive

SUPPLIER_IDS = [
    "MinerCorp_A",
    "DistribCorp_B",
    "RecycleCorp_C",
    "TraderCorp_D",
    "PremiumSteel_E",
]

CORRELATION_ID = "REQ-LLM-AGENT-CONV"


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
            correlation_id=CORRELATION_ID,
        )
    )


async def seed_quotes(state: SwarmState) -> list[str]:
    """Seed requirement + discovery + evaluation + negotiation so quotes exist."""
    seed_requirement(state)
    discovery = SupplierDiscoveryAgent()
    await drive(
        discovery,
        state,
        Event(
            type=ProcurementEventType.REQUIREMENT_CREATED,
            source="requirement_agent",
            payload={"artifact": REQUIREMENT_ARTIFACT_NAME},
            correlation_id=CORRELATION_ID,
        ),
    )
    evaluation = EvaluationAgent()
    for supplier_id in SUPPLIER_IDS:
        await drive(
            evaluation,
            state,
            Event(
                type=ProcurementEventType.SUPPLIER_DISCOVERED,
                source="supplier_discovery_agent",
                payload={
                    "supplier_id": supplier_id,
                    "material": "aluminum",
                    "artifact": "suppliers",
                },
                correlation_id=CORRELATION_ID,
            ),
        )
    negotiation = NegotiationAgent()
    for supplier_id in SUPPLIER_IDS:
        await drive(
            negotiation,
            state,
            Event(
                type=ProcurementEventType.SUPPLIER_EVALUATED,
                source="evaluation_agent",
                payload={
                    "supplier_id": supplier_id,
                    "artifact": f"evaluation_{supplier_id}",
                },
                correlation_id=CORRELATION_ID,
            ),
        )
    return [f"quote_{supplier_id}" for supplier_id in SUPPLIER_IDS]


def quotes_completed_event() -> Event:
    return Event(
        type=ProcurementEventType.QUOTES_COMPLETED,
        source="completion_tracker",
        payload={"group": "quote", "count": len(SUPPLIER_IDS)},
        correlation_id=CORRELATION_ID,
    )


async def _seeded_state() -> SwarmState:
    state = SwarmState(request_id="REQ-LLM-AGENT", goal="llm_analysis")
    await seed_quotes(state)
    return state


# --- Deterministic input construction ---


@pytest.mark.asyncio
async def test_input_payload_is_sorted_and_deterministic() -> None:
    state = await _seeded_state()
    agent = SupplierAnalysisLLMAgent()
    await drive(agent, state, quotes_completed_event())

    prompt_artifacts = state.find_artifacts(kind="llm", tags={"kind": "llm_prompt"})
    assert prompt_artifacts
    payload = prompt_artifacts[0].data["parameters"]["payload"]
    supplier_ids_in_payload = [s["supplier_id"] for s in payload["suppliers"]]
    assert supplier_ids_in_payload == sorted(supplier_ids_in_payload)
    assert len(supplier_ids_in_payload) == len(SUPPLIER_IDS)


# --- Replay-safe dedup ---


@pytest.mark.asyncio
async def test_replay_produces_no_duplicate_artifacts() -> None:
    """Running the agent twice on the same inputs → only one set of artifacts."""
    state = await _seeded_state()
    agent = SupplierAnalysisLLMAgent()
    await drive(agent, state, quotes_completed_event())
    count_after_first = len(state.find_artifacts(kind="llm"))

    agent2 = SupplierAnalysisLLMAgent()
    await drive(agent2, state, quotes_completed_event())
    count_after_second = len(state.find_artifacts(kind="llm"))

    # 1 prompt + 3 completion variants = 4 total
    assert count_after_first == 4
    assert count_after_second == 4


@pytest.mark.asyncio
async def test_identical_inputs_yield_same_input_hash() -> None:
    state = await _seeded_state()
    agent = SupplierAnalysisLLMAgent()
    await drive(agent, state, quotes_completed_event())

    llm_artifacts = state.find_artifacts(kind="llm")
    assert len(llm_artifacts) == 4  # 1 prompt + 3 completions
    hashes = {a.data["input_hash"] for a in llm_artifacts}
    assert len(hashes) == 1  # all share the same input_hash


# --- Artifact creation ---


@pytest.mark.asyncio
async def test_records_prompt_and_completion_artifacts() -> None:
    state = await _seeded_state()
    agent = SupplierAnalysisLLMAgent()
    await drive(agent, state, quotes_completed_event())

    prompts = state.find_artifacts(kind="llm", tags={"kind": "llm_prompt"})
    completions = state.find_artifacts(kind="llm", tags={"kind": "llm_completion"})
    assert len(prompts) == 1
    assert len(completions) == 3  # 3 variants for consensus

    prompt = prompts[0]
    completion = completions[0]
    assert prompt.data["model"] == "stub"
    assert completion.data["model"] == "stub"
    assert completion.parent_ids == [prompt.id]

    output = completion.data["output"]
    assert "summary" in output
    assert "risks" in output
    assert "tradeoffs" in output


# --- Timeline visibility ---


@pytest.mark.asyncio
async def test_llm_artifacts_appear_in_timeline_under_cognitive() -> None:
    state = await _seeded_state()
    agent = SupplierAnalysisLLMAgent()
    await drive(agent, state, quotes_completed_event())

    timeline = build_timeline(state)
    cognitive_items = [item for item in timeline.timeline if item.subtype == "llm"]
    assert len(cognitive_items) == 4  # 1 prompt + 3 completions
    for item in cognitive_items:
        assert item.phase == "cognitive"


# --- Hard constraints: no side effects ---


@pytest.mark.asyncio
async def test_agent_does_not_publish_events() -> None:
    """The agent must not publish any domain events (read-only)."""
    state = await _seeded_state()

    bus = EventBus()
    published: list[Event] = []

    async def capture(event: Event) -> None:
        published.append(event)

    bus.subscribe(ProcurementEventType.QUOTES_COMPLETED, capture)

    agent = SupplierAnalysisLLMAgent()
    await drive(agent, state, quotes_completed_event(), bus=bus)

    assert len(published) == 0


@pytest.mark.asyncio
async def test_agent_does_not_trigger_execution_or_governance() -> None:
    state = await _seeded_state()

    assert state.get_artifact("decision") is None
    assert state.get_artifact("governance_decision") is None
    assert state.get_artifact("contract_validation") is None
    assert state.get_artifact("execution_authorization") is None
    assert state.get_artifact("purchase_order") is None
    assert state.get_artifact("execution_status") is None
    assert state.get_artifact("procurement_outcome") is None

    agent = SupplierAnalysisLLMAgent()
    await drive(agent, state, quotes_completed_event())

    assert state.get_artifact("decision") is None
    assert state.get_artifact("governance_decision") is None
    assert state.get_artifact("contract_validation") is None
    assert state.get_artifact("execution_authorization") is None
    assert state.get_artifact("purchase_order") is None
    assert state.get_artifact("execution_status") is None
    assert state.get_artifact("procurement_outcome") is None


@pytest.mark.asyncio
async def test_agent_does_not_call_connectors() -> None:
    state = await _seeded_state()
    agent = SupplierAnalysisLLMAgent()
    await drive(agent, state, quotes_completed_event())

    external_calls = state.find_artifacts(kind="external_call")
    assert len(external_calls) == 0


# --- Replay handling ---


@pytest.mark.asyncio
async def test_agent_ignores_replayed_events() -> None:
    state = await _seeded_state()
    agent = SupplierAnalysisLLMAgent()
    event = quotes_completed_event().model_copy(update={"replayed": True})
    agent.state = state
    await agent.step(event)

    llm_artifacts = state.find_artifacts(kind="llm")
    assert len(llm_artifacts) == 0


@pytest.mark.asyncio
async def test_agent_does_not_reprocess_same_correlation_id() -> None:
    state = await _seeded_state()
    agent = SupplierAnalysisLLMAgent()
    await drive(agent, state, quotes_completed_event())

    agent2 = SupplierAnalysisLLMAgent()
    await drive(agent2, state, quotes_completed_event())

    llm_artifacts = state.find_artifacts(kind="llm")
    assert len(llm_artifacts) == 4  # 1 prompt + 3 completions


# --- Deterministic stub output ---


@pytest.mark.asyncio
async def test_stub_output_is_deterministic() -> None:
    state1 = await _seeded_state()
    agent1 = SupplierAnalysisLLMAgent()
    await drive(agent1, state1, quotes_completed_event())

    state2 = await _seeded_state()
    agent2 = SupplierAnalysisLLMAgent()
    await drive(agent2, state2, quotes_completed_event())

    llm1 = state1.find_artifacts(kind="llm")
    llm2 = state2.find_artifacts(kind="llm")

    # Compare deterministic output fields only (timestamps differ by design)
    sorted1 = sorted(llm1, key=lambda x: x.data["kind"])
    sorted2 = sorted(llm2, key=lambda x: x.data["kind"])
    for a, b in zip(sorted1, sorted2, strict=True):
        assert a.data["input_hash"] == b.data["input_hash"]
        assert a.data["kind"] == b.data["kind"]
        assert a.data["model"] == b.data["model"]
        assert a.data["prompt"] == b.data["prompt"]
        assert a.data["parameters"] == b.data["parameters"]
        assert a.data["output"] == b.data["output"]


# --- Capability declaration ---


def test_agent_declares_supplier_analyze_capability() -> None:
    agent = SupplierAnalysisLLMAgent()
    assert "supplier.analyze" in agent.capability_names


def test_agent_name_is_unique() -> None:
    agent = SupplierAnalysisLLMAgent()
    assert agent.name == "supplier_analysis_llm_agent"
