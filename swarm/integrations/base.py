"""BaseConnector — the deterministic integration port for external systems.

This is the *Adapter* (ports & adapters) entry point the procurement swarm uses
to talk to ERP / supplier systems. It is a pure interface contract: every method
returns a deterministic, serializable response object, holds no mutable state of
its own, and never raises on a missing credential (it simulates the response
deterministically instead). Real connector credentials are injected as data, so
the swarm trace stays reproducible even when no live system is configured.

Concretely, an adapter translating the external schema:

    PurchaseOrder ──submit_order──> ExternalResponse
    order_id   ──get_order_status──> ExternalStatus
    supplier_id ──validate_supplier──> bool

The three methods are the minimal surface agents need. Replay safety lives in
the runtime (``step`` skips ``reason``/``act`` for replayed events) and in the
:mod:`swarm.utils.idempotency` layer — connectors are never invoked during a
replay, and every live call is deduplicated by ``decision_id + action`` before
it leaves the swarm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from swarm.domain.artifacts import ExternalCallArtifact
from swarm.domain.order import PurchaseOrder


@dataclass(frozen=True)
class ExternalResponse:
    """Normalized result of submitting an order to an external system."""

    success: bool
    order_id: str
    status: str
    reference_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExternalStatus:
    """Normalized order status reported by an external system.

    Note: the *status values* are a pure, deterministic function of the order —
    no live timestamp is carried here. The observing agent stamps the
    :class:`ExternalCallArtifact` / :class:`ExecutionStatusArtifact` with the
    audit time, keeping the connector's contract fully reproducible.
    """

    order_id: str
    status: str
    lifecycle: list[str] = field(default_factory=list)
    reference_id: str = ""


class BaseConnector(Protocol):
    """Minimal, deterministic port every ERP/supplier adapter must satisfy.

    - ``submit_order`` hands a :class:`PurchaseOrder` to an external system and
      returns a normalized :class:`ExternalResponse`.
    - ``get_order_status`` queries the realized status of a submitted order by
      its external id and returns a normalized :class:`ExternalStatus`.
    - ``validate_supplier`` checks whether a supplier is contractually / system-
      wise usable prior to ordering.

    Implementations MUST be pure functions of their inputs (determinism), MUST
    NOT perform blocking side effects that escape idempotency, and MUST simulate
    a response when no live credentials are configured so the swarm remains
    replay-safe and auditable.
    """

    def submit_order(self, order: PurchaseOrder) -> ExternalResponse: ...

    def get_order_status(self, order_id: str) -> ExternalStatus: ...

    def validate_supplier(self, supplier_id: str) -> bool: ...


def record_external_call(
    state: Any,
    *,
    system: str,
    action: str,
    request_payload: dict[str, Any] | None = None,
    response_payload: dict[str, Any] | None = None,
    status: str = "success",
    order_id: str | None = None,
    decision_id: str | None = None,
    idempotency_key: str | None = None,
    correlation_id: str | None = None,
    created_by: str | None = None,
) -> ExternalCallArtifact:
    """Record one outbound call as an :class:`ExternalCallArtifact`.

    Every connector invocation the agents perform must produce exactly one of
    these so the artifact graph carries a full audit trail of external side
    effects — independent of which adapter answered.
    """
    data: dict[str, Any] = {
        "system": system,
        "action": action,
        "request_payload": request_payload or {},
        "response_payload": response_payload or {},
        "status": status,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    if order_id is not None:
        data["order_id"] = order_id
    if decision_id is not None:
        data["decision_id"] = decision_id
    if idempotency_key is not None:
        data["idempotency_key"] = idempotency_key
    artifact = ExternalCallArtifact(
        data=data,
        parent_ids=[decision_id] if decision_id else [],
        tags={
            "system": system,
            "action": action,
            "order_id": order_id or "",
        },
        created_by=created_by or "connector",
        correlation_id=correlation_id,
    )
    state.put_artifact(artifact)
    return artifact
