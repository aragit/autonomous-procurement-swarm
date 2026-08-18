"""Swarm runtime for autonomous multi-agent systems.

.. deprecated::
    This is the V1 asyncio runtime. New code should use the V2 mesh runtime
    in the ``mesh`` package with the ``api.v2`` FastAPI server instead. The V1
    runtime is preserved here for backward compatibility with existing tests
    and downstream consumers.

This package provides:
- swarm.core — agent primitives, event bus, and shared state (V1)
- swarm.orchestration — SwarmCoordinator (V1)
- swarm.domain — deterministic procurement agents (V1)
- swarm.integrations — external connectors (V1)
- swarm.learning — policy learning (V1)
- swarm.simulation — replay engine (V1)
- swarm.storage — event store (V1)
- swarm.utils — utility helpers (V1)
"""

import warnings

warnings.warn(
    "The 'swarm' package (V1 asyncio runtime) is deprecated. "
    "Use the 'mesh' package and 'api.v2' (V2 mesh runtime) for new development. "
    "For the legacy implementation, import from the 'legacy' package.",
    DeprecationWarning,
    stacklevel=2,
)

from swarm.core import (  # noqa: E402
    ANY_EVENT,
    AgentRegistry,
    AgentStatus,
    Artifact,
    BaseAgent,
    Capability,
    Event,
    EventBus,
    Message,
    Swarm,
    SwarmEventType,
    SwarmState,
    to_capability,
)
from swarm.core.completion import CompletionTracker  # noqa: E402
from swarm.orchestration.coordinator import SwarmCoordinator  # noqa: E402

__all__ = [
    "ANY_EVENT",
    "AgentRegistry",
    "AgentStatus",
    "Artifact",
    "BaseAgent",
    "Capability",
    "CompletionTracker",
    "Event",
    "EventBus",
    "Message",
    "Swarm",
    "SwarmCoordinator",
    "SwarmEventType",
    "SwarmState",
    "to_capability",
]
