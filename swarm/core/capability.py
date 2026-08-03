"""Declarative capability schema for swarm agents."""

from typing import Any

from pydantic import BaseModel, Field


class Capability(BaseModel):
    """A typed, declarative description of something an agent can do.

    ``name`` is the machine-readable identifier used for capability-based
    discovery and routing (e.g. ``"supplier.search"``). ``description``
    explains what the capability does and ``parameters`` documents the inputs
    it accepts, so coordinators can reason about an agent without inspecting
    its implementation.

    ``priority`` ranks agents that advertise the same capability: higher values
    are preferred when several agents match, letting specialization emerge
    (e.g. ``SupplierEUAgent`` with ``priority=10`` vs a general
    ``SupplierGlobalAgent`` with ``priority=1``).
    """

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    priority: int = 0


def to_capability(spec: str | Capability) -> Capability:
    """Normalize a capability declaration.

    Plain strings are the shorthand for ``Capability(name=...)``; passing a
    full :class:`Capability` is returned unchanged so agents can attach
    descriptions and parameters when they need them.
    """
    if isinstance(spec, Capability):
        return spec
    return Capability(name=spec)
