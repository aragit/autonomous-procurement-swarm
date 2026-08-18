"""Mesh runtime abstraction for the v2 API.

Decouples the FastAPI routes from Ray: the routes depend only on the
:class:`MeshRuntime` protocol, never importing Ray at module load.  The real
:class:`RayMeshRuntime` performs its ``ray`` / ``mesh`` imports lazily inside
methods, so the router is importable and unit-testable without Ray installed.

Tests (and alternative runtimes) inject an implementation via
:func:`set_runtime`.
"""

from __future__ import annotations

import uuid
from typing import Any, Protocol, runtime_checkable

import structlog

from api.v2.models import (
    ProcurementRunResponse,
    ProcurementStatusResponse,
)
from mesh.neuro import LLMConfig

logger = structlog.get_logger(__name__)


@runtime_checkable
class MeshRuntime(Protocol):
    """Contract every v2 runtime implementation must satisfy."""

    async def run_procurement(
        self, trace_id: str, requirement: dict[str, Any]
    ) -> ProcurementRunResponse:
        """Submit a requirement, await the DECISION, and return the outcome."""
        ...

    async def get_status(self, trace_id: str) -> ProcurementStatusResponse | None:
        """Pull the current blackboard snapshot/stats for ``trace_id``."""
        ...

    async def shutdown(self) -> None:
        """Release any long-lived resources (cluster, actors, ...)."""
        ...


# ─── Default runtime (lazily initialised, injectable in tests) ────────────────

_runtime: MeshRuntime | None = None


def set_runtime(runtime: MeshRuntime | None) -> None:
    """Bind a runtime (used by tests and application bootstrap)."""
    global _runtime
    _runtime = runtime


def _build_default_runtime() -> MeshRuntime:
    """Construct the production Ray-backed runtime.

    Ray is imported lazily so the v2 package remains importable in environments
    where the optional ``ray`` extra is not installed (e.g. unit-test sandboxes).
    When Ray is unavailable this raises a clear, actionable error so a host
    application can inject an alternative runtime via :func:`set_runtime`.
    """
    try:
        return RayMeshRuntime()
    except Exception as exc:  # pragma: no cover - exercised only without ray
        raise RuntimeError(
            "Mesh runtime unavailable: 'ray' is not installed. Call "
            "api.v2.runtime.set_runtime(<MeshRuntime>) to inject a runtime."
        ) from exc


def get_runtime() -> MeshRuntime:
    """Resolve the active runtime, lazily building a Ray-backed one if needed."""
    global _runtime
    if _runtime is None:
        _runtime = _build_default_runtime()
    return _runtime


def make_trace_id() -> str:
    """Generate a fresh v2 trace id."""
    return f"RUN-{uuid.uuid4().hex[:12].upper()}"


def _neuro_config_from_requirement(requirement: dict[str, Any]) -> LLMConfig | None:
    """Build an LLMConfig from neuro fields on the requirement, if neuro enabled."""
    if not requirement.get("enable_neuro"):
        return None
    base_url = requirement.get("neuro_llm_base_url") or "http://localhost:8000/v1"
    model = requirement.get("neuro_llm_model") or "gemma-2b-it"
    return LLMConfig(base_url=base_url, model_name=model)


class _RunRecord:
    """In-memory record of an in-flight or completed procurement trace."""

    def __init__(
        self,
        correlation_id: str,
        blackboard: Any,
        cluster: Any,
        decision: dict[str, Any] | None,
        snapshot: Any,
        stats: dict[str, Any] | None,
        status: str,
    ) -> None:
        self.correlation_id = correlation_id
        self.blackboard = blackboard
        self.cluster = cluster
        self.decision = decision
        self.snapshot = snapshot
        self.stats = stats
        self.status = status


class RayMeshRuntime:
    """Production :class:`MeshRuntime` backed by the Ray ProcurementCluster.

    Each trace owns an isolated blackboard/kernel (unique Ray actor names) so
    concurrent runs never collide.  The cluster is kept alive after a run so
    :meth:`get_status` can serve a live blackboard snapshot; :meth:`shutdown`
    tears everything down.
    """

    def __init__(self) -> None:
        import ray

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, include_dashboard=False)
        self._ray = ray
        self._runs: dict[str, _RunRecord] = {}

    async def run_procurement(
        self, trace_id: str, requirement: dict[str, Any]
    ) -> ProcurementRunResponse:
        from mesh.cluster import MeshConfig, ProcurementCluster

        config = MeshConfig(
            blackboard_name=f"v2_bb_{trace_id}",
            kernel_name=f"v2_kernel_{trace_id}",
            neuro_llm_config=_neuro_config_from_requirement(requirement),
        )
        cluster = ProcurementCluster(config)
        cluster.initialize()
        await cluster.create_actors()

        outcome = await cluster.run_procurement(requirement)
        decision = outcome.get("decision")
        decision_data = decision.get("payload", {}).get("data") if decision else None

        # Keep the blackboard alive for status queries until shutdown.
        handles = cluster.handles
        assert handles is not None, "cluster handles should be initialized after run"
        blackboard = handles.blackboard
        try:
            snapshot = await blackboard.snapshot.remote()
            stats = await blackboard.stats.remote()
        except Exception as exc:  # pragma: no cover
            logger.warning("v2_snapshot_failed", trace_id=trace_id, error=str(exc))
            snapshot = None
            stats = None

        self._runs[trace_id] = _RunRecord(
            correlation_id=requirement.get("correlation_id", trace_id),
            blackboard=blackboard,
            cluster=cluster,
            decision=decision_data,
            snapshot=snapshot,
            stats=stats,
            status="completed",
        )
        logger.info("v2_procurement_completed", trace_id=trace_id)
        return ProcurementRunResponse(
            trace_id=trace_id,
            correlation_id=requirement.get("correlation_id", trace_id),
            status="completed",
            decision=decision_data,
            buyer_result=outcome.get("buyer_result"),
        )

    async def get_status(self, trace_id: str) -> ProcurementStatusResponse | None:
        record = self._runs.get(trace_id)
        if record is None:
            return None

        # Prefer a live snapshot if the blackboard actor is still alive.
        snapshot = record.snapshot
        stats = record.stats
        if record.blackboard is not None:
            try:
                snapshot = await record.blackboard.snapshot.remote()
                stats = await record.blackboard.stats.remote()
            except Exception:  # actor gone — fall back to the cached snapshot
                snapshot = record.snapshot
                stats = record.stats

        channels: dict[str, int] = {}
        if snapshot is not None:
            channels = {_channel_key(k): len(v) for k, v in snapshot.channels.items()}

        blackboard_stats: dict[str, Any] = dict(stats) if stats else {}
        return ProcurementStatusResponse(
            trace_id=trace_id,
            correlation_id=record.correlation_id,
            status=record.status,
            decision=record.decision,
            channels=channels,
            blackboard_stats=blackboard_stats,
        )

    async def shutdown(self) -> None:
        for record in self._runs.values():
            try:
                await record.cluster.shutdown()
            except Exception as exc:  # pragma: no cover
                logger.warning("v2_cluster_shutdown_failed", error=str(exc))
        self._runs.clear()


def _channel_key(channel: Any) -> str:
    """Normalise a channel key (ChannelType or str) to its string value."""
    return channel.value if hasattr(channel, "value") else str(channel)
