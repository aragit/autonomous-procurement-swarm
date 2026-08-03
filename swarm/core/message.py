"""Agent-to-agent message model for the swarm runtime."""

from typing import Any

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A message exchanged between two agents.

    ``sender`` and ``intent`` are required. ``receiver`` is optional: when it
    is omitted the message is broadcast on its ``intent``, so agents never need
    a direct reference to one another. ``metadata`` carries routing/context
    data (priority, ttl, correlation, ...) separate from the business payload.
    ``correlation_id`` links every message and event that belong to one logical
    conversation (a single user request), so the full exchange can be traced.
    """

    sender: str
    receiver: str | None = None
    intent: str
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str | None = None
