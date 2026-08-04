"""Unit tests for the deterministic timeline projection (Phase 8.1)."""

from swarm import Event, SwarmState
from swarm.core.artifact import Artifact
from swarm.core.timeline import build_timeline
from swarm.domain.artifacts import ExternalCallArtifact


def _state_with_run() -> SwarmState:
    """A synthetic, causally consistent run.

    Each artifact is created before the event it produced, in realistic order:
    decision -> governance -> execution (external call).
    """
    state = SwarmState(request_id="REQ-TL-01", goal="timeline")
    # decision: artifact at T0, then its DecisionMade event at T0+1
    state.put_artifact(
        Artifact(
            id="d1",
            kind="decision",
            name="decision",
            data={"selected_supplier": "MinerCorp_A", "reasoning": {"ranked": []}},
            parent_ids=[],
            correlation_id="REQ-TL-01-CONV",
            created_at="2026-01-01T00:00:00.000000+00:00",
            created_by="decision_agent",
        )
    )
    state.events.append(
        Event(
            id="e1",
            type="DecisionMade",
            source="decision_agent",
            payload={"selected_supplier": "MinerCorp_A"},
            timestamp="2026-01-01T00:00:00.000001+00:00",
            correlation_id="REQ-TL-01-CONV",
        )
    )
    # governance: artifact at T2, then GovernanceDecisionMade at T3
    state.put_artifact(
        Artifact(
            id="g1",
            kind="governance_decision",
            name="governance_decision",
            data={"status": "APPROVED", "decision_id": "d1"},
            parent_ids=["d1"],
            correlation_id="REQ-TL-01-CONV",
            created_at="2026-01-01T00:00:00.000002+00:00",
            created_by="governance_agent",
        )
    )
    state.events.append(
        Event(
            id="e2",
            type="GovernanceDecisionMade",
            source="governance_agent",
            payload={"status": "APPROVED", "decision_id": "d1"},
            timestamp="2026-01-01T00:00:00.000003+00:00",
            correlation_id="REQ-TL-01-CONV",
        )
    )
    # execution: external call carries a (never-stored) secret + safe identifiers
    state.put_artifact(
        ExternalCallArtifact(
            id="ext1",
            data={
                "system": "coupa",
                "action": "submit_order",
                "request_payload": {"order_id": "PO-d1", "credentials": {"password": "sekret"}},
                "response_payload": {"client_secret": "shh", "token": "tok", "ref": "ok"},
                "status": "success",
                "idempotency_key": "d1:submit_order",
                "order_id": "PO-d1",
                "decision_id": "d1",
                "timestamp": "2026-01-01T00:00:00.000004+00:00",
            },
            correlation_id="REQ-TL-01-CONV",
            parent_ids=["d1"],
            created_by="purchase_order_agent",
            created_at="2026-01-01T00:00:00.000004+00:00",
        )
    )
    return state


def test_timeline_has_normalized_shape_and_summary() -> None:
    response = build_timeline(_state_with_run())
    assert response.request_id == "REQ-TL-01"
    assert response.status == "APPROVED"  # governance decision, no execution status
    assert response.summary.total_events == 2
    assert response.summary.total_artifacts == 3
    assert response.summary.external_calls == 1


def test_timeline_is_sorted_by_order_with_consecutive_indices() -> None:
    timeline = build_timeline(_state_with_run()).timeline
    assert [item.order for item in timeline] == list(range(len(timeline)))
    timestamps = [item.timestamp for item in timeline]
    assert timestamps == sorted(timestamps)
    # causal: each artifact precedes the event it produced -> art/ev/art/ev/art
    types = [item.type for item in timeline]
    assert types == ["artifact", "event", "artifact", "event", "artifact"]


def test_timeline_phase_markers() -> None:
    by_subtype = {item.subtype: item.phase for item in build_timeline(_state_with_run()).timeline}
    assert by_subtype["decision"] == "decision"
    assert by_subtype["governance_decision"] == "governance"
    assert by_subtype["external_call"] == "execution"
    assert by_subtype["DecisionMade"] == "decision"
    assert by_subtype["GovernanceDecisionMade"] == "governance"


def test_timeline_masks_secrets_but_preserves_identifiers() -> None:
    ext = next(
        item
        for item in build_timeline(_state_with_run()).timeline
        if item.type == "artifact" and item.subtype == "external_call"
    )
    payload = ext.payload
    # sensitive values redacted...
    assert payload["response_payload"]["client_secret"] == "***REDACTED***"
    assert payload["response_payload"]["token"] == "***REDACTED***"
    assert payload["request_payload"]["credentials"]["password"] == "***REDACTED***"
    # ...while non-sensitive identifiers are preserved for audit
    assert payload["idempotency_key"] == "d1:submit_order"
    assert payload["order_id"] == "PO-d1"
    assert payload["decision_id"] == "d1"
    assert payload["response_payload"]["ref"] == "ok"


def test_timeline_is_deterministic_across_calls() -> None:
    state = _state_with_run()
    first = build_timeline(state).model_dump_json()
    second = build_timeline(state).model_dump_json()
    assert first == second


def test_timeline_terminal_status_uses_execution_status_when_present() -> None:
    state = _state_with_run()
    state.put_artifact(
        Artifact(
            id="es1",
            kind="execution_status",
            name="execution_status",
            data={"status": "DELIVERED", "lifecycle": ["SUBMITTED", "DELIVERED"]},
            correlation_id="REQ-TL-01-CONV",
            created_at="2026-01-01T00:00:00.000005+00:00",
            created_by="execution_tracking_agent",
        )
    )
    assert build_timeline(state).status == "DELIVERED"


def test_timeline_rejected_contract_yields_rejected_status() -> None:
    state = SwarmState(request_id="REQ-TL-02", goal="rejected")
    state.put_artifact(
        Artifact(
            id="cv1",
            kind="contract_validation",
            name="contract_validation",
            data={"valid": False, "reason": "expired"},
            correlation_id="REQ-TL-02-CONV",
            created_at="2026-01-01T00:00:00.000000+00:00",
        )
    )
    timeline = build_timeline(state)
    assert timeline.status == "REJECTED"
    assert timeline.timeline[0].phase == "contract"
