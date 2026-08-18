"""Ray cluster initialization and management for the Procurement Mesh.

This module provides bootstrap functions for initializing a Ray cluster,
managing actor pools, and handling graceful shutdown. It supports both
local development (ray.init) and connecting to an existing cluster
(ray.init(address="auto")).
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import ray
import structlog
from ray.actor import ActorHandle

if TYPE_CHECKING:
    from swarm.memory import SupplierMemoryStore

from mesh.actors import (
    BuyerActor,
    EvaluatorActor,
    NegotiatorActor,
    SafetyKernelActor,
    ScoutActor,
)
from mesh.blackboard import (
    create_blackboard,
    shutdown_blackboard,
)
from mesh.channels import ChannelType
from mesh.neuro import LLMConfig, save_bandit_state


@dataclass
class MeshConfig:
    """Configuration for the Procurement Mesh cluster."""

    # Cluster connection
    address: str | None = None  # None = local, "auto" = existing cluster
    namespace: str = "procurement-mesh"

    # Actor pool sizes (can be overridden by env vars)
    n_scouts: int = field(default_factory=lambda: int(os.getenv("MESH_N_SCOUTS", "3")))
    n_evaluators: int = field(default_factory=lambda: int(os.getenv("MESH_N_EVALUATORS", "3")))
    n_negotiators: int = field(default_factory=lambda: int(os.getenv("MESH_N_NEGOTIATORS", "2")))

    # Resource constraints
    num_cpus: int | None = None  # None = all available
    num_gpus: int = 0
    object_store_memory: int | None = None  # bytes, None = default

    # Actor options
    max_restarts: int = 3
    max_task_retries: int = 3

    # Blackboard
    blackboard_name: str = "distributed_blackboard"

    # Kernel (safety)
    kernel_name: str = "safety_kernel"

    # Governance policy
    governance_policy_name: str = "standard"

    # Neuro-Symbolic bridge (Phase 3).  When set, ScoutActor and NegotiatorActor
    # use schema-constrained LLM generation validated by the SafetyKernelActor
    # with an auto-correction retry loop; on exhaustion they fall back to the
    # deterministic path.  Defaults to None (deterministic only).
    neuro_llm_config: LLMConfig | None = None
    neuro_max_retries: int = 3

    # Contextual Bandit (Phase 5).  Persistence path for LinUCB state.
    bandit_state_path: str = "/app/data/bandit_state.json"


@dataclass
class ActorHandles:
    """Container for all mesh actor handles."""

    blackboard: ActorHandle[Any]
    scouts: list[ActorHandle[Any]] = field(default_factory=list)
    evaluators: list[ActorHandle[Any]] = field(default_factory=list)
    negotiators: list[ActorHandle[Any]] = field(default_factory=list)
    buyer: ActorHandle[Any] | None = None
    kernel: ActorHandle[Any] | None = None


class ProcurementCluster:
    """Manages the lifecycle of a Procurement Mesh Ray cluster."""

    def __init__(self, config: MeshConfig | None = None) -> None:
        self.config = config or MeshConfig()
        self._initialized = False
        self._handles: ActorHandles | None = None
        self._memory: SupplierMemoryStore | None = None  # SupplierMemoryStore for evaluators

    @property
    def handles(self) -> ActorHandles | None:
        return self._handles

    @property
    def is_initialized(self) -> bool:
        return self._initialized and ray.is_initialized()

    def initialize(self) -> None:
        """Initialize the Ray cluster connection."""
        if ray.is_initialized():
            ray.shutdown()

        init_kwargs = {
            "namespace": self.config.namespace,
            "ignore_reinit_error": True,
            "include_dashboard": False,
        }

        if self.config.address:
            init_kwargs["address"] = self.config.address
        else:
            if self.config.num_cpus:
                init_kwargs["num_cpus"] = self.config.num_cpus
            if self.config.num_gpus:
                init_kwargs["num_gpus"] = self.config.num_gpus
            if self.config.object_store_memory:
                init_kwargs["object_store_memory"] = self.config.object_store_memory

        ray.init(**init_kwargs)  # type: ignore[arg-type]
        self._initialized = True

    async def create_actors(self) -> ActorHandles:
        """Create all mesh actors and register them."""
        if not self._initialized:
            raise RuntimeError("Cluster not initialized. Call initialize() first.")

        # Set bandit state path for persistent bandit
        os.environ["BANDIT_STATE_PATH"] = self.config.bandit_state_path

        # Create blackboard first (other actors depend on it)
        blackboard = await create_blackboard(self.config.blackboard_name)

        # Create safety kernel (singleton)
        kernel = SafetyKernelActor.options(  # type: ignore[attr-defined]
            name=self.config.kernel_name,
            max_restarts=0,
            max_task_retries=0,
        ).remote()
        # Kernel handle is passed directly to actors; no global registration needed

        # Create supplier memory store for evaluators
        from swarm.memory import SupplierMemoryStore

        self._memory = SupplierMemoryStore()

        # Create Scout pool (elastic)
        scouts = [
            ScoutActor.options(  # type: ignore[attr-defined]
                name=f"scout_{i}",
                max_restarts=self.config.max_restarts,
                max_task_retries=self.config.max_task_retries,
            ).remote(
                f"scout_{i}",
                blackboard,
                kernel,
                llm_config=self.config.neuro_llm_config,
                neuro_max_retries=self.config.neuro_max_retries,
            )
            for i in range(self.config.n_scouts)
        ]

        # Create Evaluator pool (elastic)
        evaluators = [
            EvaluatorActor.options(  # type: ignore[attr-defined]
                name=f"evaluator_{i}",
                max_restarts=self.config.max_restarts,
                max_task_retries=self.config.max_task_retries,
            ).remote(f"evaluator_{i}", blackboard, kernel, self._memory)
            for i in range(self.config.n_evaluators)
        ]

        # Create Negotiator pool (elastic)
        negotiators = [
            NegotiatorActor.options(  # type: ignore[attr-defined]
                name=f"negotiator_{i}",
                max_restarts=self.config.max_restarts,
                max_task_retries=self.config.max_task_retries,
            ).remote(
                f"negotiator_{i}",
                blackboard,
                kernel,
                llm_config=self.config.neuro_llm_config,
                neuro_max_retries=self.config.neuro_max_retries,
            )
            for i in range(self.config.n_negotiators)
        ]

        # Create Buyer (SINGLETON)
        from swarm.domain.governance import STANDARD_POLICY, STRICT_POLICY

        policy = (
            STANDARD_POLICY if self.config.governance_policy_name == "standard" else STRICT_POLICY
        )
        buyer = BuyerActor.options(  # type: ignore[attr-defined]
            name="buyer_0",
            max_restarts=0,
            max_task_retries=0,
        ).remote("buyer_0", blackboard, kernel, policy)

        self._handles = ActorHandles(
            blackboard=blackboard,
            scouts=scouts,
            evaluators=evaluators,
            negotiators=negotiators,
            buyer=buyer,
            kernel=kernel,
        )

        return self._handles

    async def run_procurement(self, requirement: dict[str, Any]) -> dict[str, Any]:
        """Run a complete procurement cycle through the mesh.

        Phases:
        1. Write requirement to blackboard (kernel)
        2. Parallel scout deployment (discovery)
        3. Parallel evaluation (score + risk)
        4. Parallel negotiation (deal)
        5. Deterministic MCDA by buyer (decision)

        Args:
            requirement: The procurement requirement dict.

        Returns:
            The final decision dict.
        """
        if not self._handles:
            await self.create_actors()
        assert self._handles is not None, "Handles should be initialized after create_actors"

        bb = self._handles.blackboard

        # Phase 0: Write requirement (kernel writes to requirement channel)
        correlation_id = requirement.get("correlation_id", "")
        req_trace_id = await bb.write.remote(
            "kernel",
            ChannelType.REQUIREMENT,
            {"requirement": requirement, "correlation_id": correlation_id},
        )
        logger.info(
            "procurement_started",
            correlation_id=requirement.get("correlation_id", "unknown"),
            req_trace_id=req_trace_id,
        )

        # Phase 1: Deploy scouts in parallel
        scout_tasks = [s.step.remote() for s in self._handles.scouts]
        scout_results = await asyncio.gather(*scout_tasks, return_exceptions=True)
        for i, result in enumerate(scout_results):
            if isinstance(result, Exception):
                logger.error("scout_failed", scout=f"scout_{i}", error=str(result))
            else:
                logger.debug("scout_completed", scout=f"scout_{i}", result=result)

        # Phase 2: Parallel evaluation
        eval_tasks = [e.step.remote() for e in self._handles.evaluators]
        eval_results = await asyncio.gather(*eval_tasks, return_exceptions=True)
        for i, result in enumerate(eval_results):
            if isinstance(result, Exception):
                logger.error("evaluator_failed", evaluator=f"evaluator_{i}", error=str(result))
            else:
                logger.debug("evaluator_completed", evaluator=f"evaluator_{i}", result=result)

        # Phase 3: Parallel negotiation
        neg_tasks = [n.step.remote() for n in self._handles.negotiators]
        neg_results = await asyncio.gather(*neg_tasks, return_exceptions=True)
        for i, result in enumerate(neg_results):
            if isinstance(result, Exception):
                logger.error("negotiator_failed", negotiator=f"negotiator_{i}", error=str(result))
            else:
                logger.debug("negotiator_completed", negotiator=f"negotiator_{i}", result=result)

        # Phase 4: Buyer MCDA (singleton)
        assert self._handles.buyer is not None, "Buyer actor should be initialized"
        decision_result = await self._handles.buyer.step.remote()
        logger.info("buyer_completed", result=decision_result)

        # Read the decision from DECISION channel
        decisions = await bb.read.remote("kernel", ChannelType.DECISION, limit=5)
        final_decision = None
        for d in decisions:
            if d.get("correlation_id") == requirement.get("correlation_id"):
                final_decision = d
                break

        # Update bandits with reward from decision
        if final_decision:
            await self._update_bandits_from_decision(final_decision, requirement)

        return {
            "status": "completed",
            "decision": final_decision,
            "buyer_result": decision_result,
        }

    async def _update_bandits_from_decision(
        self, decision: dict[str, Any], requirement: dict[str, Any]
    ) -> None:
        """Update all negotiator bandits with reward from the final decision."""
        if self._handles is None:
            return
        try:
            # Get the selected supplier from decision
            decision_payload = decision.get("payload", {})
            data = decision_payload.get("data", {})
            selected_supplier = data.get("selected_supplier")
            if not selected_supplier:
                return

            # Read the DEAL channel to find the quote for the selected supplier
            deals = await self._handles.blackboard.read.remote(
                "kernel", ChannelType.DEAL, limit=20
            )

            for deal in deals:
                payload = deal.get("payload", {})
                if payload.get("type") != "quote":
                    continue
                deal_data = payload.get("data", {})
                if deal_data.get("supplier_id") == selected_supplier:
                    # Update bandit on each negotiator actor
                    for negotiator in self._handles.negotiators:
                        try:
                            await negotiator.update_bandit_from_decision.remote(
                                decision=data,
                                quote=deal_data,
                                requirement_data=requirement,
                            )
                        except Exception as e:
                            logger.warning(
                                "bandit_update_failed",
                                negotiator=str(negotiator),
                                error=str(e),
                            )
                    break
        except Exception as e:
            logger.warning("bandit_update_error", error=str(e))

    def get_cluster_info(self) -> dict[str, Any]:
        """Get information about the Ray cluster."""
        if not ray.is_initialized():
            return {"status": "not_initialized"}

        return {
            "status": "initialized",
            "address": ray.get_runtime_context().gcs_address,
            "resources": ray.cluster_resources(),  # type: ignore[no-untyped-call]
            "available_resources": ray.available_resources(),  # type: ignore[no-untyped-call]
            "nodes": len(ray.nodes()),  # type: ignore[no-untyped-call]
        }

    async def shutdown(self) -> None:
        """Gracefully shutdown all actors and the Ray cluster."""
        if self._handles:
            for scout in self._handles.scouts:
                ray.kill(scout)
            for evaluator in self._handles.evaluators:
                ray.kill(evaluator)
            for negotiator in self._handles.negotiators:
                ray.kill(negotiator)
            if self._handles.buyer:
                ray.kill(self._handles.buyer)
            if self._handles.kernel:
                ray.kill(self._handles.kernel)
            await shutdown_blackboard(self.config.blackboard_name)
            self._handles = None

        # Persist bandit state
        try:
            await save_bandit_state()
        except Exception as e:
            logger.warning("bandit_save_failed", error=str(e))

        if ray.is_initialized():
            ray.shutdown()
        self._initialized = False


logger = structlog.get_logger(__name__)


# Convenience functions for quick setup
_cluster_instance: ProcurementCluster | None = None


async def initialize_cluster(config: MeshConfig | None = None) -> ProcurementCluster:
    """Initialize the global procurement cluster."""
    global _cluster_instance
    _cluster_instance = ProcurementCluster(config)
    _cluster_instance.initialize()
    await _cluster_instance.create_actors()
    return _cluster_instance


def get_cluster() -> ProcurementCluster | None:
    """Get the global cluster instance."""
    return _cluster_instance


async def shutdown_cluster() -> None:
    """Shutdown the global cluster."""
    global _cluster_instance
    if _cluster_instance:
        await _cluster_instance.shutdown()
        _cluster_instance = None


async def run_procurement(requirement: dict[str, Any]) -> dict[str, Any]:
    """Run a procurement cycle on the global cluster."""
    cluster = get_cluster()
    if cluster is None:
        cluster = await initialize_cluster()
    return await cluster.run_procurement(requirement)


# Context manager for cluster lifecycle
class ClusterContext:
    """Async context manager for ProcurementCluster lifecycle."""

    def __init__(self, config: MeshConfig | None = None):
        self.config = config
        self.cluster: ProcurementCluster | None = None

    async def __aenter__(self) -> ProcurementCluster:
        self.cluster = ProcurementCluster(self.config)
        self.cluster.initialize()
        await self.cluster.create_actors()
        return self.cluster

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        if self.cluster:
            await self.cluster.shutdown()
            self.cluster = None
