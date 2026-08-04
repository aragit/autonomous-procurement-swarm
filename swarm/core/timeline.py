"""Deterministic timeline projection for a procurement swarm run.

``GET /swarm/timeline/{request_id}`` returns the *single*, causally ordered story
of one run: every event and artifact, merged by time, normalized into a stable
shape, with the contract/risk/governance/execution phases marked and any
sensitive fields masked.

This is a **read-only projection**: it copies stored events and artifacts only —
it never re-runs agent logic, never mutates state, and never calls an external
system. Because events are recorded in publish order and artifacts in creation
order, the merged stream is fully reproducible: the same state always yields the
same timeline ordering (timestamps are the primary key, with a stable per-stream
stream rank + index tie-breaker so microsecond collisions never reorder the
output nondeterministically).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from swarm.core.artifact import Artifact
from swarm.core.event import Event
from swarm.core.state import SwarmState

TimelineItemType = Literal["event", "artifact"]

#: Sensitive payload keys are redacted (exact, case-insensitive match). The
#: integration layer never stores secrets in artifacts/events by design; this is
#: a defensive guarantee that masking is never surprising on identifiers such as
#: ``idempotency_key``/``authorization_id``/``decision_id``.
_SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "password",
        "secret",
        "token",
        "api_key",
        "apikey",
        "client_secret",
        "private_key",
        "access_token",
        "refresh_token",
    }
)

#: Primary timestamp source per item type.
_TIMESTAMP_ATTRS = {"event": "timestamp", "artifact": "created_at"}

#: Stream tie-breaker rank: artifacts (created) sort before events (published)
#: when timestamps collide — causally correct since an agent writes the artifact
#: before publishing the event it produced.
_STREAM_RANK: dict[str, int] = {"artifact": 0, "event": 1}

_EVENT_PHASE: dict[str, str] = {
    "RequirementCreated": "discovery",
    "StrategySelected": "discovery",
    "SupplierDiscovered": "discovery",
    "SupplierEvaluated": "evaluation",
    "QuoteGenerated": "evaluation",
    "EvaluationCompleted": "evaluation",
    "QuotesCompleted": "evaluation",
    "DecisionMade": "decision",
    "ContractValidated": "contract",
    "ContractRejected": "contract",
    "RiskAssessmentCompleted": "risk",
    "GovernanceDecisionMade": "governance",
    "ApprovalGranted": "governance",
    "ApprovalRequired": "governance",
    "ApprovalRejected": "governance",
    "PurchaseOrderCreated": "execution",
    "ExecutionStatusUpdated": "execution",
    "ExternalCallRecorded": "execution",
    "OutcomeRecorded": "outcome",
    "SupplierPerformanceUpdated": "outcome",
}

_ARTIFACT_PHASE: dict[str, str] = {
    "requirement": "discovery",
    "strategy": "discovery",
    "supplier_list": "discovery",
    "evaluation": "evaluation",
    "quote": "evaluation",
    "decision": "decision",
    "decision_explanation": "decision",
    "contract_validation": "contract",
    "risk_assessment": "risk",
    "governance_decision": "governance",
    "execution_authorization": "governance",
    "purchase_order": "execution",
    "execution_status": "execution",
    "external_call": "execution",
    "procurement_outcome": "outcome",
    "supplier_performance": "outcome",
}

#: Event sources that are runtime bookkeeping, not procurement agency.
_RUNTIME_SOURCES = {"swarm", "coordinator", "completion_tracker"}


def _mask_secrets(value: Any, key: str | None = None) -> Any:
    """Recursively redact values whose key names secrets fields."""
    if isinstance(value, dict):
        return {name: _mask_secrets(val, name) for name, val in value.items()}
    if isinstance(value, list):
        return [_mask_secrets(item) for item in value]
    if key is not None and key.lower() in _SENSITIVE_KEYS:
        return "***REDACTED***"
    return value


def _phase_for_event(event: Event) -> str:
    if event.source in _RUNTIME_SOURCES:
        return "runtime"
    return _EVENT_PHASE.get(event.type, "unknown")


def _phase_for_artifact(artifact: Artifact) -> str:
    return _ARTIFACT_PHASE.get(artifact.kind, "unknown")


def _agent_of(artifact: Artifact) -> str | None:
    return artifact.created_by or None


class TimelineItem(BaseModel):
    """A single normalized, causally-positioned node in a run's timeline."""

    id: str
    type: TimelineItemType
    subtype: str
    phase: str
    agent: str | None = None
    timestamp: str
    correlation_id: str | None = None
    parent_ids: list[str] = Field(default_factory=list)
    replayed: bool | None = None
    order: int
    payload: dict[str, Any] = Field(default_factory=dict)


class TimelineSummary(BaseModel):
    """Aggregate counts for a timeline projection."""

    total_events: int
    total_artifacts: int
    external_calls: int


class TimelineResponse(BaseModel):
    """Read-only, merged timeline for one procurement swarm run."""

    request_id: str
    status: str
    timeline: list[TimelineItem]
    summary: TimelineSummary


def _terminal_status(state: SwarmState) -> str:
    """Best-effort terminal status: execution → governance outcome → incomplete."""
    exec_status = state.get_artifact("execution_status")
    if exec_status is not None:
        return str(exec_status.data.get("status", "unknown"))
    governance = state.get_artifact("governance_decision")
    if governance is not None:
        return str(governance.data.get("status", "unknown"))
    contract = state.get_artifact("contract_validation")
    if contract is not None and contract.data.get("valid") is False:
        return "REJECTED"
    return "incomplete"


def _build_event_item(event: Event, order: int) -> TimelineItem:
    return TimelineItem(
        id=event.id,
        type="event",
        subtype=event.type,
        phase=_phase_for_event(event),
        agent=event.source or None,
        timestamp=event.timestamp,
        correlation_id=event.correlation_id,
        parent_ids=[],
        replayed=event.replayed or None,
        order=order,
        payload=_mask_secrets(dict(event.payload)),
    )


def _build_artifact_item(artifact: Artifact, order: int) -> TimelineItem:
    return TimelineItem(
        id=artifact.id,
        type="artifact",
        subtype=artifact.kind,
        phase=_phase_for_artifact(artifact),
        agent=_agent_of(artifact),
        timestamp=artifact.created_at,
        correlation_id=artifact.correlation_id,
        parent_ids=list(artifact.parent_ids),
        replayed=None,
        order=order,
        payload=_mask_secrets(dict(artifact.data)),
    )


def build_timeline(state: SwarmState) -> TimelineResponse:
    """Project a run's events + artifacts into a single deterministic timeline.

    Pure read: iterates ``state.events`` and ``state.artifacts`` (never mutates),
    normalizes each into a :class:`TimelineItem`, and merges them by timestamp
    with a stable tie-break (artifact before event on collision, then per-stream
    creation/publish index) so the ordering is reproducible across replays.
    """
    raw: list[tuple[str, int, int, Any]] = []
    for idx, artifact in enumerate(state.artifacts):
        raw.append((artifact.created_at, _STREAM_RANK["artifact"], idx, artifact))
    for idx, event in enumerate(state.events):
        raw.append((event.timestamp, _STREAM_RANK["event"], idx, event))

    # Stable sort: (timestamp, stream_rank, stream_index) — deterministic and
    # independent of dict/set iteration.
    raw.sort(key=lambda item: (str(item[0]), item[1], item[2]))

    items: list[TimelineItem] = []
    for order, (_, _, _, node) in enumerate(raw):
        if isinstance(node, Artifact):
            items.append(_build_artifact_item(node, order))
        elif isinstance(node, Event):
            items.append(_build_event_item(node, order))

    external_calls = len(state.find_artifacts(kind="external_call"))
    return TimelineResponse(
        request_id=str(state.request_id),
        status=_terminal_status(state),
        timeline=items,
        summary=TimelineSummary(
            total_events=len(state.events),
            total_artifacts=len(state.artifacts),
            external_calls=external_calls,
        ),
    )
