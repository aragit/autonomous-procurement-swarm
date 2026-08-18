"""Ray Distributed Mesh for the Neuro-Symbolic Procurement System.

This package provides:
- mesh.channels: Channel definitions and capability-scoped ACLs (always importable)
- mesh.neuro: Neuro-Symbolic bridge (schemas, backend, retry loop) (always importable)
- mesh.actors: Ray actor archetypes (Scout, Evaluator, Negotiator, Buyer, Kernel)
- mesh.cluster: Ray cluster initialization and actor pool management
- mesh.blackboard: DistributedBlackboard Ray actor with typed channels

The Ray runtime (actors, blackboard, cluster) is optional at import time: the
Neuro-Symbolic bridge and channel ACLs can be imported and unit-tested without
Ray installed.  When ``ray`` is importable the full runtime is exposed.
"""

from mesh.channels import (
    ARTIFACT_KIND_TO_CHANNEL,
    CHANNEL_ACLS,
    CHANNEL_TO_ARTIFACT_KINDS,
    ChannelACL,
    ChannelSnapshot,
    ChannelTrace,
    ChannelType,
    artifact_kinds_for_channel,
    channel_for_artifact_kind,
    check_read_permission,
    check_write_permission,
    get_acl,
)
from mesh.neuro import (
    LLMConfig,
    MockNeuroBackend,
    NeuralProposal,
    NeuroResult,
    NeuroSymbolicBridge,
    OpenAICompatibleBackend,
    ProtocolViolation,
    ScoutProposal,
    SymbolicVerdict,
)

# Ray runtime components are optional — only exposed when Ray is installed so
# the Neuro-Symbolic bridge and channel ACLs can be used in pure-Python tests.
try:
    import ray  # noqa: F401

    _RAY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when ray is absent
    _RAY_AVAILABLE = False

if _RAY_AVAILABLE:
    from mesh.actors import (
        BuyerActor,
        EvaluatorActor,
        MeshActor,
        NegotiatorActor,
        SafetyKernelActor,
        ScoutActor,
    )
    from mesh.blackboard import (
        DistributedBlackboard,
        create_blackboard,
        get_blackboard,
        shutdown_blackboard,
    )
    from mesh.cluster import (
        ActorHandles,
        ClusterContext,
        MeshConfig,
        ProcurementCluster,
        get_cluster,
        initialize_cluster,
        run_procurement,
        shutdown_cluster,
    )

__all__ = [
    # Channels (always available)
    "ChannelType",
    "ChannelACL",
    "ChannelTrace",
    "ChannelSnapshot",
    "CHANNEL_ACLS",
    "get_acl",
    "check_read_permission",
    "check_write_permission",
    "ARTIFACT_KIND_TO_CHANNEL",
    "CHANNEL_TO_ARTIFACT_KINDS",
    "channel_for_artifact_kind",
    "artifact_kinds_for_channel",
    # Neuro-Symbolic bridge (always available)
    "NeuralProposal",
    "SymbolicVerdict",
    "NeuroResult",
    "ProtocolViolation",
    "NeuroSymbolicBridge",
    "LLMConfig",
    "OpenAICompatibleBackend",
    "MockNeuroBackend",
    "ScoutProposal",
    # Ray runtime (only present when ray is installed)
    "MeshActor",
    "SafetyKernelActor",
    "ScoutActor",
    "EvaluatorActor",
    "NegotiatorActor",
    "BuyerActor",
    "DistributedBlackboard",
    "create_blackboard",
    "get_blackboard",
    "shutdown_blackboard",
    "MeshConfig",
    "ActorHandles",
    "ProcurementCluster",
    "ClusterContext",
    "initialize_cluster",
    "get_cluster",
    "shutdown_cluster",
    "run_procurement",
]
