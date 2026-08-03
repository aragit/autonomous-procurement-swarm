"""Typed artifacts shared between swarm agents."""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Artifact(BaseModel):
    """A typed, versioned piece of working data produced by an agent.

    ``kind`` and ``name`` give the artifact a stable identity while ``tags``
    (e.g. ``{"supplier": "abc", "region": "EU"}``) enable the high-selectivity
    queries Phase 2 will need — "all quotes for supplier X", "latest evaluation
    for region EU" — without a database. ``parent_ids`` records the names of the
    source artifacts this artifact was derived from, so every piece of state
    carries its lineage. Versions are immutable — updating an artifact creates a
    new one, so history is preserved and auditable.
    """

    id: str = Field(default_factory=lambda: uuid4().hex)
    kind: str
    name: str
    data: dict[str, Any] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)
    parent_ids: list[str] = Field(default_factory=list)
    version: int = 1
    created_by: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_by: str | None = None
    updated_at: str | None = None
    correlation_id: str | None = None

    def update(self, data: dict[str, Any], *, by: str | None = None) -> "Artifact":
        """Return a new, next-versioned copy of this artifact with ``data``."""
        return self.model_copy(
            update={
                "data": data,
                "version": self.version + 1,
                "updated_by": by,
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )
