"""Unit tests for LLMArtifact recording and hashing (v0.9 Step 1)."""

import pytest

from swarm import SwarmState
from swarm.core.timeline import build_timeline
from swarm.domain.artifacts import (
    LLM_ARTIFACT_NAME,
    LLM_KINDS,
    llm_artifact_name,
)
from swarm.utils.llm_hash import compute_llm_input_hash, record_llm_artifact


def test_compute_llm_input_hash_is_deterministic() -> None:
    h1 = compute_llm_input_hash("gpt-4o-mini", "hello", {"temperature": 0.0})
    h2 = compute_llm_input_hash("gpt-4o-mini", "hello", {"temperature": 0.0})
    assert h1 == h2
    assert len(h1) == 64  # SHA256 hex


def test_compute_llm_input_hash_changes_with_input() -> None:
    base = compute_llm_input_hash("gpt-4o-mini", "hello", {"temperature": 0.0})
    diff_prompt = compute_llm_input_hash("gpt-4o-mini", "world", {"temperature": 0.0})
    diff_model = compute_llm_input_hash("gpt-4", "hello", {"temperature": 0.0})
    diff_params = compute_llm_input_hash("gpt-4o-mini", "hello", {"temperature": 0.7})
    assert diff_prompt != base
    assert diff_model != base
    assert diff_params != base


def test_compute_llm_input_hash_is_order_independent() -> None:
    """Parameter key order must not affect the hash."""
    params_a = {"temperature": 0.0, "max_tokens": 100}
    params_b = {"max_tokens": 100, "temperature": 0.0}
    assert compute_llm_input_hash("gpt-4o-mini", "hello", params_a) == compute_llm_input_hash(
        "gpt-4o-mini", "hello", params_b
    )


def test_prompt_and_completion_share_input_hash() -> None:
    """Both artifacts for the same LLM call must share the same input_hash."""
    state = SwarmState(request_id="REQ-LLM-SHARE", goal="llm")
    prompt_artifact = record_llm_artifact(
        state,
        model="gpt-4o-mini",
        prompt="Analyze suppliers",
        parameters={"temperature": 0.0},
        kind="llm_prompt",
        correlation_id="REQ-LLM-SHARE-CONV",
    )
    completion_artifact = record_llm_artifact(
        state,
        model="gpt-4o-mini",
        prompt="Analyze suppliers",
        parameters={"temperature": 0.0},
        output={"summary": "done"},
        kind="llm_completion",
        parent_ids=[prompt_artifact.id],
        correlation_id="REQ-LLM-SHARE-CONV",
    )
    assert prompt_artifact.data["input_hash"] == completion_artifact.data["input_hash"]
    assert len(state.find_artifacts(kind="llm")) == 2


def test_llm_artifact_fields() -> None:
    state = SwarmState(request_id="REQ-LLM-01", goal="llm")
    artifact = record_llm_artifact(
        state,
        model="gpt-4o-mini",
        prompt="What is procurement?",
        parameters={"temperature": 0.0},
        output={"answer": "Procurement is..."},
        kind="llm_prompt",
        correlation_id="REQ-LLM-01-CONV",
    )
    assert artifact.kind == "llm"
    assert artifact.name.startswith("llm_llm_prompt_")
    assert artifact.data["model"] == "gpt-4o-mini"
    assert artifact.data["prompt"] == "What is procurement?"
    assert artifact.data["input_hash"]
    assert artifact.data["output"] == {"answer": "Procurement is..."}
    assert artifact.tags == {"model": "gpt-4o-mini", "kind": "llm_prompt"}


def test_record_llm_artifact_persists_to_state() -> None:
    state = SwarmState(request_id="REQ-LLM-02", goal="llm")
    record_llm_artifact(state, model="gpt-4o-mini", prompt="test", kind="llm_prompt")
    llm_artifacts = state.find_artifacts(kind="llm")
    assert len(llm_artifacts) == 1


def test_record_llm_artifact_is_replay_safe_dedup() -> None:
    """Identical inputs must produce one artifact, not two."""
    state = SwarmState(request_id="REQ-LLM-03", goal="llm")
    first = record_llm_artifact(state, model="gpt-4o-mini", prompt="hello", kind="llm_prompt")
    second = record_llm_artifact(state, model="gpt-4o-mini", prompt="hello", kind="llm_prompt")
    assert first.id == second.id
    assert len(state.find_artifacts(kind="llm")) == 1


def test_record_llm_artifact_different_prompts_create_distinct_artifacts() -> None:
    state = SwarmState(request_id="REQ-LLM-04", goal="llm")
    record_llm_artifact(state, model="gpt-4o-mini", prompt="hello", kind="llm_prompt")
    record_llm_artifact(state, model="gpt-4o-mini", prompt="world", kind="llm_prompt")
    assert len(state.find_artifacts(kind="llm")) == 2


def test_record_llm_artifact_rejects_unknown_kind() -> None:
    state = SwarmState(request_id="REQ-LLM-05", goal="llm")
    try:
        record_llm_artifact(state, model="gpt-4o-mini", prompt="hello", kind="llm_unknown")
    except ValueError as exc:
        assert "llm_unknown" in str(exc)
    else:
        pytest.fail("Should have raised ValueError")


def test_llm_artifact_appears_in_timeline() -> None:
    state = SwarmState(request_id="REQ-TL-LLM", goal="llm")
    record_llm_artifact(
        state,
        model="gpt-4o-mini",
        prompt="Plan the procurement",
        parameters={"temperature": 0.0},
        output={"plan": "step1, step2"},
        kind="llm_prompt",
        correlation_id="REQ-TL-LLM-CONV",
    )
    timeline = build_timeline(state)
    llm_item = next(item for item in timeline.timeline if item.subtype == "llm")
    assert llm_item.phase == "cognitive"
    assert llm_item.payload["model"] == "gpt-4o-mini"
    assert llm_item.payload["prompt"] == "Plan the procurement"
    assert "output" in llm_item.payload


def test_llm_artifact_masking_in_timeline() -> None:
    state = SwarmState(request_id="REQ-TL-LLM-SEC", goal="llm")
    record_llm_artifact(
        state,
        model="gpt-4o-mini",
        prompt="Do something",
        parameters={"api_key": "sk-secret"},
        output={"token": "leaked-token"},
        kind="llm_prompt",
        correlation_id="REQ-TL-LLM-SEC-CONV",
    )
    timeline = build_timeline(state)
    llm_item = next(item for item in timeline.timeline if item.subtype == "llm")
    assert llm_item.payload["parameters"]["api_key"] == "***REDACTED***"
    assert llm_item.payload["output"]["token"] == "***REDACTED***"


def test_llm_artifact_name_helper_is_stable() -> None:
    name = llm_artifact_name("llm_prompt", "abc123")
    assert name == "llm_llm_prompt_abc123"


def test_llm_kinds_contains_expected_values() -> None:
    assert frozenset({"llm_prompt", "llm_completion", "llm_embedding"}) == LLM_KINDS
    assert LLM_ARTIFACT_NAME == "llm"
