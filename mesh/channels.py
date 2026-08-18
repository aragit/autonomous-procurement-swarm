"""Channel definitions and capability-scoped ACLs for the Ray Distributed Blackboard.

This module defines the typed channels that agents read from and write to,
and the Access Control Lists (ACLs) that enforce capability-based permissions.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ChannelType(StrEnum):
    """Typed channels for the distributed blackboard.

    Each channel represents a category of procurement information.
    Agents have scoped read/write access based on their archetype.
    """

    REQUIREMENT = "requirement"
    DISCOVERY = "discovery"
    SCORE = "score"
    RISK = "risk"
    DEAL = "deal"
    DECISION = "decision"


class ChannelACL(BaseModel):
    """Capability-based read/write permissions per channel."""

    channel: ChannelType
    read: list[str] = Field(default_factory=list)
    write: list[str] = Field(default_factory=list)


# ACL configuration matching the migration plan specification.
# Archetype names: "scout", "evaluator", "negotiator", "buyer", "kernel"
CHANNEL_ACLS: list[ChannelACL] = [
    ChannelACL(
        channel=ChannelType.REQUIREMENT,
        read=["scout", "evaluator", "negotiator", "buyer", "kernel"],
        write=["kernel"],
    ),
    ChannelACL(
        channel=ChannelType.DISCOVERY,
        read=["evaluator", "negotiator", "kernel"],
        write=["scout"],
    ),
    ChannelACL(
        channel=ChannelType.SCORE,
        read=["negotiator", "buyer", "kernel"],
        write=["evaluator"],
    ),
    ChannelACL(
        channel=ChannelType.RISK,
        read=["buyer", "kernel"],
        write=["evaluator"],
    ),
    ChannelACL(
        channel=ChannelType.DEAL,
        read=["buyer", "kernel"],
        write=["negotiator"],
    ),
    ChannelACL(
        channel=ChannelType.DECISION,
        read=["kernel"],
        write=["buyer"],
    ),
]


# Convenience: build a lookup dict for fast ACL checks
_ACL_LOOKUP: dict[ChannelType, ChannelACL] = {acl.channel: acl for acl in CHANNEL_ACLS}


def get_acl(channel: ChannelType) -> ChannelACL:
    """Get the ACL for a channel."""
    return _ACL_LOOKUP[channel]


def check_read_permission(archetype: str, channel: ChannelType) -> bool:
    """Check if an archetype can read from a channel."""
    acl = _ACL_LOOKUP[channel]
    return archetype in acl.read


def check_write_permission(archetype: str, channel: ChannelType) -> bool:
    """Check if an archetype can write to a channel."""
    acl = _ACL_LOOKUP[channel]
    return archetype in acl.write


class ChannelTrace(BaseModel):
    """A single trace entry in a channel.

    Every write to a channel creates an immutable trace entry with
    full lineage (parent_ids) and attribution (archetype, timestamp).
    """

    id: str
    timestamp: float
    archetype: str
    payload: dict[str, Any]
    parent_ids: list[str] = Field(default_factory=list)
    correlation_id: str | None = None


class ChannelSnapshot(BaseModel):
    """Full snapshot of all channels for replay/audit."""

    channels: dict[ChannelType, list[ChannelTrace]]
    timestamp: float


# Mapping from legacy artifact kinds to new channels for migration compatibility
ARTIFACT_KIND_TO_CHANNEL: dict[str, ChannelType] = {
    "requirement": ChannelType.REQUIREMENT,
    "supplier_list": ChannelType.DISCOVERY,
    "evaluation": ChannelType.SCORE,
    "risk_assessment": ChannelType.RISK,
    "quote": ChannelType.DEAL,
    "decision": ChannelType.DECISION,
    "governance_decision": ChannelType.DECISION,  # Kernel service, not a channel
    "execution_authorization": ChannelType.DECISION,  # Kernel service
}


# Mapping from channel to legacy artifact kinds for reverse compatibility
CHANNEL_TO_ARTIFACT_KINDS: dict[ChannelType, list[str]] = {
    ChannelType.REQUIREMENT: ["requirement"],
    ChannelType.DISCOVERY: ["supplier_list"],
    ChannelType.SCORE: ["evaluation"],
    ChannelType.RISK: ["risk_assessment"],
    ChannelType.DEAL: ["quote"],
    ChannelType.DECISION: ["decision"],
}


def channel_for_artifact_kind(kind: str) -> ChannelType | None:
    """Map a legacy artifact kind to its channel."""
    return ARTIFACT_KIND_TO_CHANNEL.get(kind)


def artifact_kinds_for_channel(channel: ChannelType) -> list[str]:
    """Map a channel to its legacy artifact kinds."""
    return CHANNEL_TO_ARTIFACT_KINDS.get(channel, [])
