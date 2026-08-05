"""Shared, serializable swarm state."""

from typing import Any

from pydantic import BaseModel, Field

from swarm.core.artifact import Artifact
from swarm.core.event import Event


class SwarmState(BaseModel):
    """Shared context that every agent can read and write.

    Working data is stored as typed :class:`Artifact` objects rather than an
    arbitrary dict, so the state stays structured and auditable: every artifact
    records its kind, producer and version history. ``events`` records what has
    happened so far and ``results`` collects the final outputs produced by the
    swarm. Serialize with :meth:`to_dict` / :meth:`from_dict`. No database
    persistence is performed here — storage is a later phase.
    """

    request_id: str = ""
    goal: str = ""
    artifacts: list[Artifact] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    results: dict[str, Any] = Field(default_factory=dict)
    expectations: dict[str, dict[str, int]] = Field(default_factory=dict)
    completions: dict[str, list[str]] = Field(default_factory=dict)

    def put_artifact(self, artifact: Artifact) -> Artifact:
        """Record ``artifact`` in shared state and return it."""
        self.artifacts.append(artifact)
        return artifact

    def get_artifact(self, name: str) -> Artifact | None:
        """Return the latest version of the artifact named ``name``."""
        latest: Artifact | None = None
        for artifact in self.artifacts:
            if artifact.name == name and (latest is None or artifact.version > latest.version):
                latest = artifact
        return latest

    def artifacts_by_kind(self, kind: str) -> list[Artifact]:
        """Return every artifact of the given ``kind``, oldest first."""
        return [artifact for artifact in self.artifacts if artifact.kind == kind]

    def find_artifacts(
        self,
        *,
        name: str | None = None,
        kind: str | None = None,
        tags: dict[str, str] | None = None,
        correlation_id: str | None = None,
    ) -> list[Artifact]:
        """Return artifacts matching every supplied filter, oldest first.

        ``tags`` must match on all keys given (e.g. ``{"supplier": "abc"}``
        selects every quote for that supplier). ``correlation_id`` narrows the
        search to one logical conversation. All filters are optional and
        combine with AND.
        """
        matches = self.artifacts
        if name is not None:
            matches = [artifact for artifact in matches if artifact.name == name]
        if kind is not None:
            matches = [artifact for artifact in matches if artifact.kind == kind]
        if tags is not None:
            matches = [
                artifact
                for artifact in matches
                if all(artifact.tags.get(key) == value for key, value in tags.items())
            ]
        if correlation_id is not None:
            matches = [
                artifact for artifact in matches if artifact.correlation_id == correlation_id
            ]
        return matches

    def expect_artifact(
        self,
        kind: str,
        *,
        count: int = 1,
        correlation_id: str | None = None,
    ) -> None:
        """Declare that ``count`` artifacts of ``kind`` are expected for a request.

        Completion tracking uses this as the group size: once that many
        artifacts of ``kind`` exist for the request, the group is closed and
        its completion event fires. ``correlation_id`` defaults to the swarm's
        own request id.
        """
        cid = correlation_id or self.request_id
        if not cid:
            raise ValueError("expect_artifact requires a correlation_id or request_id")
        self.expectations.setdefault(cid, {})[kind] = count

    def expected_count(self, correlation_id: str, group: str) -> int | None:
        """The expected artifact count for ``group``, or None if undeclared."""
        return self.expectations.get(correlation_id, {}).get(group)

    def completed_artifact_count(self, correlation_id: str, group: str) -> int:
        """How many ``group`` artifacts exist so far for ``correlation_id``."""
        return len(self.find_artifacts(kind=group, correlation_id=correlation_id))

    def is_group_completed(self, correlation_id: str, group: str) -> bool:
        """Whether ``group`` has already been declared complete for a request."""
        return group in self.completions.get(correlation_id, [])

    def complete_artifact(self, kind: str, *, correlation_id: str | None = None) -> None:
        """Record that the ``kind`` group is complete for a request (idempotent)."""
        cid = correlation_id or self.request_id
        if not cid:
            raise ValueError("complete_artifact requires a correlation_id or request_id")
        if not self.is_group_completed(cid, kind):
            self.completions.setdefault(cid, []).append(kind)

    def get_execution_trace(self, correlation_id: str) -> dict[str, Any]:
        """Ordered events, artifacts and agent actions for one conversation.

        ``agent_actions`` merges artifact creation and event publication into a
        single chronological list (source events from the runtime itself are
        omitted), so the trace reads like an audit trail of what each agent
        did.
        """
        events = [event for event in self.events if event.correlation_id == correlation_id]
        artifacts = [
            artifact for artifact in self.artifacts if artifact.correlation_id == correlation_id
        ]

        agent_actions: list[dict[str, Any]] = []
        for artifact in artifacts:
            agent_actions.append(
                {
                    "agent": artifact.created_by,
                    "action": "artifact_created",
                    "kind": artifact.kind,
                    "name": artifact.name,
                    "parent_ids": list(artifact.parent_ids),
                    "tags": dict(artifact.tags),
                    "timestamp": artifact.created_at,
                }
            )
        for event in events:
            if event.source in ("swarm", "coordinator", "completion_tracker"):
                continue
            agent_actions.append(
                {
                    "agent": event.source,
                    "action": "event_published",
                    "event_type": event.type,
                    "payload": dict(event.payload),
                    "timestamp": event.timestamp,
                }
            )
        agent_actions.sort(key=lambda action: str(action["timestamp"]))

        return {
            "correlation_id": correlation_id,
            "events": [event.model_dump() for event in events],
            "artifacts": [artifact.model_dump() for artifact in artifacts],
            "agent_actions": agent_actions,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serializable dict representation of the state."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SwarmState":
        """Rebuild state from a serialized dict."""
        return cls.model_validate(raw)
