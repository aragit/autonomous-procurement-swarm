"""Read-only LLM artifact reader (v0.9 Step 3 & 5).

Provides a purely observational interface for agents to consume LLM analysis
output without any side effects or authority. This is the controlled bridge
between the cognitive layer (LLMArtifact) and the authoritative path
(StrategyAgent / DecisionAgent) — it only *reads*; it never creates, mutates,
or publishes.
"""

from __future__ import annotations

from typing import Any

from swarm.core.artifact import Artifact
from swarm.core.state import SwarmState


def _extract_output(artifact: Artifact) -> dict[str, Any] | None:
    """Return the ``output`` dict from an LLM artifact, or None."""
    output = artifact.data.get("output")
    if isinstance(output, dict):
        return output
    return None


def get_latest_llm_completion(
    state: SwarmState,
    *,
    correlation_id: str | None = None,
) -> dict[str, Any] | None:
    """Return the structured output of the latest ``llm_completion`` artifact.

    Searches ``state`` for LLM completion artifacts (kind ``"llm"``,
    tag ``kind="llm_completion"``), optionally scoped to one
    ``correlation_id`` (request). Returns the ``output`` dict of the most
    recently created matching artifact, or ``None`` if none exists.

    This is **read-only**: it inspects artifacts already in state and never
    creates or mutates them.
    """
    completions = _find_completion_artifacts(state, correlation_id)
    if not completions:
        return None
    return _extract_output(completions[-1])


def get_all_llm_completions(
    state: SwarmState,
    *,
    correlation_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return the structured output of every ``llm_completion`` artifact.

    Searches ``state`` for LLM completion artifacts (kind ``"llm"``,
    tag ``kind="llm_completion"``), optionally scoped to one
    ``correlation_id``. Returns the ``output`` dict of each matching artifact
    (oldest first). Non-dict outputs are silently skipped.

    This is **read-only**: it inspects artifacts already in state and never
    creates or mutates them.
    """
    completions = _find_completion_artifacts(state, correlation_id)
    results: list[dict[str, Any]] = []
    for artifact in completions:
        output = _extract_output(artifact)
        if output is not None:
            results.append(output)
    return results


def _find_completion_artifacts(
    state: SwarmState,
    correlation_id: str | None,
) -> list[Artifact]:
    """Find all llm_completion artifacts, optionally filtered by correlation_id."""
    return state.find_artifacts(
        kind="llm",
        tags={"kind": "llm_completion"},
        correlation_id=correlation_id,
    )
