"""Orchestration layer for the swarm runtime.

The coordinator is the central hub that registers agents, routes events and
maintains the shared swarm state. It performs no planning, invokes no LLM and
spawns no agents — it only moves events between agents and state.
"""

from swarm.orchestration.coordinator import SwarmCoordinator

__all__ = ["SwarmCoordinator"]
