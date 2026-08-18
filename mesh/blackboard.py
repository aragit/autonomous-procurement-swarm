"""Distributed Blackboard for the Neuro-Symbolic Procurement Mesh.

This module provides a Ray actor-based distributed blackboard that replaces
the in-process EventBus with capability-scoped channels. The blackboard
enforces ACLs on every read/write, provides atomic operations, and supports
deterministic replay through snapshots.

Key design decisions:
- Ray actors are single-threaded: method calls are serialized, providing
  inherent race-condition protection without explicit locks.
- All state mutations (writes) are append-only to channel traces.
- ACL checks happen atomically with the read/write operation.
- Snapshots provide point-in-time views for replay and audit.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import ray

from mesh.channels import (
    CHANNEL_ACLS,
    ChannelACL,
    ChannelSnapshot,
    ChannelTrace,
    ChannelType,
    check_read_permission,
)


@ray.remote
class DistributedBlackboard:
    """Ray actor implementing a distributed blackboard with capability-scoped channels.

    Unlike the legacy EventBus (global broadcast) or SwarmState (global dict),
    this enforces that agents only see what their capabilities allow.

    Race condition handling:
    - Ray actors process method calls sequentially (single-threaded by default).
    - Every write is an atomic append to a list.
    - Every read returns a consistent snapshot of the current state.
    - ACL checks are performed within the same serialized method call.
    - No explicit locks needed — Ray's actor model provides serialization.

    Usage:
        blackboard = DistributedBlackboard.remote()
        trace_id = await blackboard.write.remote("scout", ChannelType.DISCOVERY, payload)
        traces = await blackboard.read.remote("evaluator", ChannelType.DISCOVERY)
    """

    def __init__(self) -> None:
        # Initialize empty channels
        self._channels: dict[ChannelType, list[ChannelTrace]] = {ct: [] for ct in ChannelType}
        self._acls: dict[ChannelType, ChannelACL] = {acl.channel: acl for acl in CHANNEL_ACLS}
        self._write_count = 0
        self._read_count = 0

    def write(
        self,
        archetype: str,
        channel: ChannelType,
        payload: dict[str, Any],
        parent_ids: list[str] | None = None,
    ) -> str:
        """Write a trace to a channel. Enforced by ACL.

        Args:
            archetype: The agent archetype attempting the write (e.g., "scout").
            channel: The channel to write to.
            payload: The data payload to store.
            parent_ids: Optional list of parent trace IDs for lineage tracking.

        Returns:
            The trace ID of the created entry.

        Raises:
            PermissionError: If the archetype lacks write permission for the channel.
            ValueError: If the channel is invalid.
        """
        if channel not in self._channels:
            raise ValueError(f"Unknown channel: {channel}")

        acl = self._acls[channel]
        if archetype not in acl.write:
            raise PermissionError(
                f"Archetype '{archetype}' cannot write to '{channel.value}'. "
                f"Allowed writers: {acl.write}"
            )

        trace_id = uuid.uuid4().hex
        # Extract correlation_id from payload if present
        correlation_id = payload.get("correlation_id")
        trace = ChannelTrace(
            id=trace_id,
            timestamp=time.time(),
            archetype=archetype,
            payload=payload,
            parent_ids=parent_ids or [],
            correlation_id=correlation_id,
        )
        self._channels[channel].append(trace)
        self._write_count += 1
        return trace_id

    def read(
        self,
        archetype: str,
        channel: ChannelType,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Read traces from a channel. Enforced by ACL.

        Args:
            archetype: The agent archetype attempting the read.
            channel: The channel to read from.
            limit: Maximum number of traces to return (most recent first).

        Returns:
            List of trace dictionaries (most recent first, limited by limit).

        Raises:
            PermissionError: If the archetype lacks read permission for the channel.
            ValueError: If the channel is invalid.
        """
        if channel not in self._channels:
            raise ValueError(f"Unknown channel: {channel}")

        acl = self._acls[channel]
        if archetype not in acl.read:
            raise PermissionError(
                f"Archetype '{archetype}' cannot read from '{channel.value}'. "
                f"Allowed readers: {acl.read}"
            )

        traces = self._channels[channel]
        # Return most recent first, limited
        result = [t.model_dump() for t in traces[-limit:]]
        result.reverse()  # Most recent first
        self._read_count += 1
        return result

    def read_multi(
        self,
        archetype: str,
        channels: list[ChannelType],
        limit: int = 100,
    ) -> dict[ChannelType, list[dict[str, Any]]]:
        """Read from multiple channels atomically.

        Args:
            archetype: The agent archetype attempting the read.
            channels: List of channels to read from.
            limit: Maximum number of traces per channel.

        Returns:
            Dict mapping each channel to its list of traces.

        Raises:
            PermissionError: If the archetype lacks read permission for any channel.
            ValueError: If any channel is invalid.
        """
        result = {}
        for channel in channels:
            result[channel] = self.read(archetype, channel, limit)
        return result

    def read_all(
        self,
        archetype: str,
        limit: int = 100,
    ) -> dict[ChannelType, list[dict[str, Any]]]:
        """Read from all channels the archetype has read access to."""
        accessible = [ct for ct in ChannelType if check_read_permission(archetype, ct)]
        return self.read_multi(archetype, accessible, limit)

    def snapshot(self) -> ChannelSnapshot:
        """Get a full snapshot of all channels for replay/audit.

        Returns a point-in-time view of all channel traces.
        """
        channels_copy = {k: [t.model_copy() for t in v] for k, v in self._channels.items()}
        return ChannelSnapshot(channels=channels_copy, timestamp=time.time())

    def get_channel_traces(
        self,
        channel: ChannelType,
    ) -> list[ChannelTrace]:
        """Get all traces for a channel (for internal/replay use).

        This bypasses ACL checks — use with caution.
        """
        if channel not in self._channels:
            raise ValueError(f"Unknown channel: {channel}")
        return list(self._channels[channel])

    def clear_channel(self, channel: ChannelType) -> int:
        """Clear all traces from a channel. Returns count of removed traces.

        Intended for testing and reset scenarios only.
        """
        if channel not in self._channels:
            raise ValueError(f"Unknown channel: {channel}")
        count = len(self._channels[channel])
        self._channels[channel].clear()
        return count

    def clear_all(self) -> int:
        """Clear all channels. Returns total count of removed traces."""
        total = sum(len(v) for v in self._channels.values())
        for ch in self._channels.values():
            ch.clear()
        self._write_count = 0
        self._read_count = 0
        return total

    def stats(self) -> dict[str, Any]:
        """Get blackboard statistics."""
        return {
            "channels": {ct.value: len(traces) for ct, traces in self._channels.items()},
            "total_writes": self._write_count,
            "total_reads": self._read_count,
            "acls": {ct.value: acl.model_dump() for ct, acl in self._acls.items()},
        }

    def get_acl(self, channel: ChannelType) -> ChannelACL:
        """Get the ACL for a channel."""
        return self._acls[channel]


# Convenience functions for creating and accessing the blackboard
_BLACKBOARD_NAME = "distributed_blackboard"


async def create_blackboard(name: str = _BLACKBOARD_NAME) -> ray.actor.ActorHandle:
    """Create and register a named DistributedBlackboard actor."""
    return DistributedBlackboard.options(name=name).remote()  # type: ignore[attr-defined]


async def get_blackboard(name: str = _BLACKBOARD_NAME) -> ray.actor.ActorHandle:
    """Get a reference to an existing named DistributedBlackboard actor."""
    return ray.get_actor(name)


async def shutdown_blackboard(name: str = _BLACKBOARD_NAME) -> None:
    """Shutdown the named DistributedBlackboard actor."""
    try:
        actor = await get_blackboard(name)
        ray.kill(actor)
    except ValueError:
        # Actor doesn't exist
        pass
