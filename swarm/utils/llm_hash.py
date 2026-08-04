"""Deterministic LLM input hashing and artifact recording (v0.9 Step 1).

This module provides a replay-safe way to record LLM invocation inputs/output
pairs as :class:`LLMArtifact` objects. It never invokes an LLM itself — it
only computes a content hash of the canonicalized input and persists the
artifact so that identical prompts produce identical (deduplicated) artifacts.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from swarm.domain.artifacts import (
    LLM_KINDS,
    LLMArtifact,
    llm_artifact_name,
)


def compute_llm_input_hash(
    model: str,
    prompt: str | None,
    parameters: dict[str, Any] | None = None,
) -> str:
    """Compute a deterministic SHA256 hash of an LLM invocation's input.

    The hash is over the canonical (sorted-key) JSON of the model, prompt, and
    parameters. Two calls with identical inputs produce identical hashes,
    enabling replay-safe deduplication. The ``kind`` of the artifact
    (prompt/completion/embedding) is intentionally NOT part of the hash so
    that both the prompt and completion artifacts for the same LLM call
    share the same ``input_hash`` — the hash represents the LLM invocation,
    not the artifact type.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "parameters": parameters or {},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def record_llm_artifact(
    state: Any,
    *,
    model: str,
    prompt: str | None,
    parameters: dict[str, Any] | None = None,
    output: dict[str, Any] | str | None = None,
    kind: str = "llm_prompt",
    variant: int | None = None,
    correlation_id: str | None = None,
    parent_ids: list[str] | None = None,
    by: str = "llm_recorder",
) -> LLMArtifact:
    """Record an LLM invocation as a :class:`LLMArtifact`.

    Replay-safe: if an artifact with the same ``(kind, input_hash, variant)``
    already exists in ``state``, the existing artifact is returned without
    creating a duplicate. This guarantees deterministic, deduplicated traces
    across replays.

    ``variant`` differentiates multiple completion outputs for the same input
    (multi-LLM consensus). It is part of the artifact name but not the
    ``input_hash``, so all variants share one hash for grouping.

    ``kind`` must be one of :data:`LLM_KINDS`
    (``"llm_prompt"`` / ``"llm_completion"`` / ``"llm_embedding"``).
    """
    if kind not in LLM_KINDS:
        raise ValueError(f"Unknown LLM artifact kind: {kind!r}; expected one of {LLM_KINDS}")

    input_hash = compute_llm_input_hash(model, prompt, parameters)

    name = llm_artifact_name(kind, input_hash, variant)
    existing = state.find_artifacts(kind="llm", name=name)
    if existing:
        first = existing[0]
        if isinstance(first, LLMArtifact):
            return first

    data: dict[str, Any] = {
        "kind": kind,
        "model": model,
        "input_hash": input_hash,
        "prompt": prompt,
        "parameters": parameters or {},
        "variant": variant,
        "output": output,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    tags: dict[str, str] = {"model": model, "kind": kind}
    if variant is not None:
        tags["variant"] = str(variant)

    artifact = LLMArtifact(
        name=name,
        data=data,
        tags=tags,
        parent_ids=parent_ids or [],
        created_by=by,
        correlation_id=correlation_id,
    )
    state.put_artifact(artifact)
    return artifact
