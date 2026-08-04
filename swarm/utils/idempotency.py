"""Deterministic idempotency guards for external side effects (Phase 8).

The swarm source of truth is the artifact graph, never an external system.
When an external call *is* made, it must happen at most once per logical
operation: replaying an order submission or re-running an execution step must
not produce duplicate external side effects.

A :class:`IdempotencyGuard` deduplicates by a deterministic ``(decision_id,
action)`` key. The key is fully reproducible from the swarm state (it never
depends on wall-clock time or random ids), so the same key always resolves to
the same outcome — preserving replay safety. Guards are per-process / per-swarm
by default and reset on replay (the runtime skips ``act`` for replayed events
anyway, so the guard is a belt-and-braces defence for the API-driven execution
path that invokes agents directly).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IdempotencyKey:
    """A reproducible key for a single external operation on a decision."""

    decision_id: str
    action: str

    def __hash__(self) -> int:
        return hash((self.decision_id, self.action))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IdempotencyKey):
            return NotImplemented
        return (self.decision_id, self.action) == (other.decision_id, other.action)


@dataclass
class IdempotencyGuard:
    """Records which external operations have already been dispatched.

    ``action`` is the connector method name (e.g. ``"submit_order"``). A call to
    :meth:`seen` returns True if the operation was already performed, so the
    caller can return the cached result instead of re-invoking the connector.
    """

    _seen: dict[IdempotencyKey, bool] = field(default_factory=dict)

    def check(self, decision_id: str, action: str) -> bool:
        """Return True if ``(decision_id, action)`` was already performed."""
        return IdempotencyKey(decision_id, action) in self._seen

    def mark(self, decision_id: str, action: str) -> IdempotencyKey:
        """Record that ``(decision_id, action)`` was performed; return the key."""
        key = IdempotencyKey(decision_id=decision_id, action=action)
        self._seen[key] = True
        return key

    def reset(self) -> None:
        """Clear all recorded operations (e.g. when starting a fresh run)."""
        self._seen.clear()
