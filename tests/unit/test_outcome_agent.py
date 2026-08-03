"""Unit tests for the Phase 5 OutcomeAgent."""

from typing import Any

import pytest

from swarm import Event, EventBus, SwarmState
from swarm.core.artifact import Artifact
from swarm.core.event import SwarmEventType
from swarm.core.message import Message
from swarm.domain import RECORD_OUTCOME_INTENT
from swarm.domain.agents import OutcomeAgent
from swarm.domain.artifacts import DECISION_ARTIFACT_NAME
from tests.unit.procurement_helpers import drive

DECISION_ID = "deadbeef-decision-id"


def seed_decision(state: SwarmState, *, supplier_id: str = "MinerCorp_A") -> None:
    state.put_artifact(
        Artifact(
            id=DECISION_ID,
            kind="decision",
            name=DECISION_ARTIFACT_NAME,
            data={
                "selected_supplier": supplier_id,
                "reasoning": {
                    "criteria": "score",
                    "ranked": [
                        {"supplier_id": supplier_id, "price": 984.0, "policy_passed": True}
                    ],
                },
            },
            created_by="decision_agent",
            correlation_id="REQ-OUT-01",
        )
    )


def outcome_payload(**overrides: Any) -> dict:
    base = {
        "decision_id": DECISION_ID,
        "supplier_id": "MinerCorp_A",
        "delivered_on_time": True,
        "quality_score": 0.92,
        "actual_price": 984.0,
        "carbon_score": 1800.0,
    }
    base.update(overrides)
    return base


def outcome_message(cid: str = "REQ-OUT-02", **overrides: Any) -> Event:
    payload = outcome_payload(**overrides)
    return Event(
        type=SwarmEventType.MESSAGE,
        source="user",
        payload={"intent": RECORD_OUTCOME_INTENT, "receiver": None},
        correlation_id=cid,
        message=Message(
            sender="user",
            intent=RECORD_OUTCOME_INTENT,
            payload=payload,
            correlation_id=cid,
        ),
    )


def test_outcome_agent_requires_decision_field() -> None:
    # sanity: the seed helper places a decision artifact named "decision"
    state = SwarmState()
    seed_decision(state)
    decision = state.get_artifact(DECISION_ARTIFACT_NAME)
    assert decision is not None
    assert decision.id == DECISION_ID


@pytest.mark.asyncio
async def test_outcome_agent_creates_outcome_artifact() -> None:
    agent = OutcomeAgent()
    state = SwarmState()
    seed_decision(state)
    await drive(agent, state, outcome_message())

    outcomes = state.find_artifacts(kind="procurement_outcome")
    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.kind == "procurement_outcome"
    assert outcome.data["supplier_id"] == "MinerCorp_A"
    assert outcome.data["delivered_on_time"] is True
    assert outcome.data["quality_score"] == 0.92
    assert outcome.parent_ids == [DECISION_ID]
    assert outcome.correlation_id == "REQ-OUT-02"


@pytest.mark.asyncio
async def test_outcome_agent_publishes_outcome_recorded() -> None:
    agent = OutcomeAgent()
    bus = EventBus()
    agent.bus = bus
    seen: list[Event] = []

    async def record(event: Event) -> None:
        seen.append(event)

    bus.subscribe("OutcomeRecorded", record)
    state = SwarmState()
    seed_decision(state)
    await drive(agent, state, outcome_message())

    assert len(seen) == 1
    assert seen[0].correlation_id == "REQ-OUT-02"
    assert seen[0].payload["supplier_id"] == "MinerCorp_A"


@pytest.mark.asyncio
async def test_outcome_agent_rejects_missing_fields() -> None:
    agent = OutcomeAgent()
    state = SwarmState()
    seed_decision(state)
    bad = outcome_message(decision_id=DECISION_ID, supplier_id="MinerCorp_A")
    # Strip the required numeric fields by overwriting the message payload.
    bad = bad.model_copy(
        update={
            "message": Message(
                sender="user",
                intent=RECORD_OUTCOME_INTENT,
                payload={"supplier_id": "MinerCorp_A"},
                correlation_id="REQ-OUT-02",
            )
        }
    )
    await drive(agent, state, bad)

    assert state.find_artifacts(kind="procurement_outcome") == []


@pytest.mark.asyncio
async def test_outcome_agent_ignores_replayed_messages() -> None:
    agent = OutcomeAgent()
    state = SwarmState()
    seed_decision(state)
    event = outcome_message().model_copy(update={"replayed": True})
    agent.state = state
    await agent.step(event)

    assert state.find_artifacts(kind="procurement_outcome") == []


@pytest.mark.asyncio
async def test_outcome_agent_records_with_wrong_decision_id_as_lineage() -> None:
    agent = OutcomeAgent()
    state = SwarmState()
    seed_decision(state)
    await drive(agent, state, outcome_message(decision_id="wrong-id", cid="REQ-OUT-03"))

    outcomes = state.find_artifacts(kind="procurement_outcome")
    assert len(outcomes) == 1
    assert outcomes[0].parent_ids == ["wrong-id"]
