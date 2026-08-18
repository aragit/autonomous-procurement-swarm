"""Ray-free shared types for the Neuro-Symbolic mesh bridge.

These dataclasses are the contract between the "neuro" (LLM/SLM structured
generation) and the "symbolic" (SafetyKernelActor) halves of the mesh.  They are
intentionally kept free of any Ray import so they can be used in pure-Python
unit tests and by the API layer without launching a Ray cluster.

``mesh.actors.base`` re-exports these symbols so existing callers keep working.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NeuralProposal:
    """A proposal from a neural/cognitive component requiring kernel validation."""

    proposal_id: str
    archetype: str
    payload: dict[str, Any]
    confidence: float = 0.0
    structured: bool = True
    # Optional, free-form audit context attached by the neuro bridge (e.g. the
    # structured schema name and the retry attempt that produced this proposal).
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SymbolicVerdict:
    """Result of kernel symbolic validation."""

    approved: bool
    reason: str
    clamped_payload: dict[str, Any] | None = None
    audit_hash: str | None = None
    # Machine-readable rejection trace, e.g. ``"price_out_of_bounds"``.  Populated
    # by the SafetyKernelActor so the neuro bridge can feed it back into the LLM.
    violations: list[str] = field(default_factory=list)

    @staticmethod
    def rejected(
        reason: str,
        violations: list[str] | None = None,
        audit_hash: str | None = None,
    ) -> SymbolicVerdict:
        """Convenience factory for a rejected verdict."""
        return SymbolicVerdict(
            approved=False,
            reason=reason,
            clamped_payload=None,
            audit_hash=audit_hash or uuid.uuid4().hex,
            violations=violations or [],
        )

    @staticmethod
    def approved_verdict(
        clamped_payload: dict[str, Any] | None,
        reason: str = "VALIDATED",
    ) -> SymbolicVerdict:
        """Convenience factory for an approved verdict."""
        return SymbolicVerdict(
            approved=True,
            reason=reason,
            clamped_payload=clamped_payload,
            audit_hash=uuid.uuid4().hex,
        )
