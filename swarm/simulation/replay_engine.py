"""Deterministic replay + simulation engine (v1.0 Step 21).

Design:

- **Read-only re-execution**. ``replay_trace`` reloads a full persisted trace,
  reconstructs the original requirement, re-runs the procurement swarm, and
  compares the replayed decision against the original.  All writes during the
  re-run are redirected to a throwaway database (see :func:`_isolate_storage`)
  so production state is never mutated.

- **Determinism**. Thresholds are pinned via :func:`override_thresholds` and
  the market simulator + LLM analysis are already deterministic, so the same
  trace replayed twice with the same thresholds produces an identical result.

- **Adaptive vs. static**. ``run_procurement(..., adaptive=...)`` controls
  whether adaptive (feedback-derived) thresholds or config-only thresholds are
  used.  Feeding different thresholds is the *only* sanctioned source of
  difference between an original run and its replay.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
from contextlib import suppress
from typing import Any

from swarm.api.procurement import (
    RequirementPayload,
    execute_procurement,
    generate_trace_id,
)
from swarm.config import SIMULATION_LIMIT
from swarm.learning.adaptive_policy import (
    get_adaptive_thresholds,
    get_config_thresholds,
    override_thresholds,
)
from swarm.storage.event_store import (
    load_full_trace,
    load_recent_trace_ids,
)


class TraceNotFoundError(Exception):
    """Raised when a trace id has no persisted events to replay."""


def extract_input(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Reconstruct the original procurement request from persisted events.

    Looks for an explicit ``procurement_request`` event (written by the
    ``/procurement/run`` endpoint).  Falls back to the ``RequirementCreated``
    domain event's normalized requirement if the original request event is
    absent (e.g. for traces persisted before this field existed).

    Args:
        events: The ``events`` list from :func:`load_full_trace`.

    Returns:
        A dict suitable for constructing a :class:`RequirementPayload`.
    """
    for event in events:
        if event.get("event_type") == "procurement_request":
            payload = event.get("payload")
            if isinstance(payload, dict):
                return dict(payload)
    for event in events:
        if event.get("event_type") == "RequirementCreated":
            requirement = event.get("payload", {}).get("requirement", {})
            constraints = requirement.get("constraints", {})
            return {
                "material": constraints.get("material", "steel"),
                "quantity": constraints.get("quantity", 1000),
                "budget": constraints.get("budget", 500_000.0),
                "target_lead_time_days": constraints.get("target_lead_time_days", 30),
                "max_carbon_per_unit": constraints.get("max_carbon_per_unit"),
                "goal": requirement.get("text"),
            }
    return {}


def _decision_from_result(result: dict[str, Any]) -> dict[str, Any]:
    """Extract the comparable decision fields from a ``run_procurement`` result."""
    return {
        "selected_supplier": result.get("selected_supplier"),
        "score": result.get("score", 0.0),
    }


def _decision_from_artifacts(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract the comparable decision fields from persisted artifact rows.

    The procurement endpoint stores the decision artifact under the type
    ``"result"``; its ``data`` contains ``selected_supplier`` and a ranked
    list whose first entry carries the winning ``score``.
    """
    for artifact in artifacts:
        if artifact.get("artifact_type") == "result":
            data = artifact.get("data", {})
            ranked = data.get("reasoning", {}).get("ranked", [])
            score = ranked[0].get("score", 0.0) if ranked else 0.0
            return {
                "selected_supplier": data.get("selected_supplier"),
                "score": round(score, 4),
            }
    return {"selected_supplier": None, "score": 0.0}


@contextlib.contextmanager
def _isolate_storage() -> Any:
    """Redirect all event-store writes to a throwaway database.

    The replay/simulation path must never mutate production traces.  Swapping
    the module-level ``_DB_PATH`` guarantees that ``store_llm_record`` (the only
    write triggered during ``run_swarm``) lands in a temporary database that is
    deleted on exit.
    """
    import swarm.storage.event_store as es

    orig = es._DB_PATH
    fd, tmp_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    es._DB_PATH = tmp_path
    try:
        es.init_db(tmp_path)
        yield
    finally:
        es._DB_PATH = orig
        with suppress(OSError):
            os.unlink(tmp_path)


def run_procurement(
    original_input: dict[str, Any],
    *,
    adaptive: bool = True,
) -> dict[str, Any]:
    """Re-execute a procurement from a reconstructed input dict.

    This is the simulation entry point: it runs the swarm in an isolated
    database (no production writes) and pins the thresholds so the run is
    fully deterministic.

    Args:
        original_input: A dict for constructing :class:`RequirementPayload`.
        adaptive: When True, use feedback-derived (adaptive) thresholds read
            from the production store *before* isolation.  When False, use
            config-only thresholds.

    Returns:
        A result dict with ``selected_supplier``, ``score``, ``strategy``,
        ``llm`` (observability incl. the thresholds actually used) and
        ``trace_id``.
    """
    requirement = RequirementPayload(**original_input)
    trace_id = generate_trace_id(requirement)

    # Read thresholds from the production store BEFORE isolating writes so that
    # adaptive=True reflects accumulated feedback, while isolation still keeps
    # the re-run from mutating any production state.
    thresholds = get_adaptive_thresholds() if adaptive else get_config_thresholds()

    with _isolate_storage(), override_thresholds(thresholds):
        try:
            response, _state = asyncio.run(
                execute_procurement(
                    requirement,
                    trace_id=trace_id,
                    adaptive=adaptive,
                    thresholds=thresholds,
                )
            )
        except Exception as exc:
            return {
                "selected_supplier": None,
                "score": 0.0,
                "strategy": {},
                "llm": {
                    "used": False,
                    "reason": "no_llm_data",
                    "metrics": {},
                    "drift": {"detected": False, "reasons": []},
                    "explain": {},
                    "thresholds_used": thresholds,
                    "thresholds_source": "static" if not adaptive else "adaptive",
                    "error": str(exc),
                },
                "trace_id": trace_id,
            }

    return _format_replay_result(response, trace_id)


def _format_replay_result(response: dict[str, Any], trace_id: str) -> dict[str, Any]:
    """Project an :func:`execute_procurement` response into the replay result shape."""
    result = response.get("result", {})
    return {
        "selected_supplier": result.get("selected_supplier"),
        "score": result.get("score", 0.0),
        "strategy": response.get("strategy", {}),
        "llm": response.get("llm", {}),
        "trace_id": trace_id,
    }


def replay_trace(trace_id: str, use_adaptive: bool = True) -> dict[str, Any]:
    """Replay a single persisted trace and compare against the original.

    Args:
        trace_id: The trace id to look up in the event store.
        use_adaptive: Whether to replay with adaptive thresholds (True) or
            config-only thresholds (False).

    Returns:
        A dict with ``original`` (persisted artifacts), ``replayed`` (re-run
        result), ``comparison`` and ``trace_id``.

    Raises:
        TraceNotFoundError: If no events are persisted for ``trace_id``.
    """
    trace = load_full_trace(trace_id)
    if not trace.get("events") and not trace.get("artifacts") and not trace.get("llm_history"):
        raise TraceNotFoundError(trace_id)

    original_input = extract_input(trace["events"])
    new_result = run_procurement(original_input, adaptive=use_adaptive)

    original = _decision_from_artifacts(trace["artifacts"])
    replayed = _decision_from_result(new_result)
    comparison = compare_results(original, replayed)

    return {
        "original": trace["artifacts"],
        "replayed": new_result,
        "comparison": comparison,
        "trace_id": trace_id,
    }


def compare_results(original: dict[str, Any], replayed: dict[str, Any]) -> dict[str, Any]:
    """Compare a normalized original decision against a replayed decision.

    Both arguments are normalized dicts with ``selected_supplier`` and
    ``score`` keys (see :func:`_decision_from_artifacts` /
    :func:`_decision_from_result`).

    Returns:
        A dict with ``same_supplier`` (bool), ``score_delta`` (float, replayed
        minus original) and ``decision_changed`` (bool).
    """
    orig_supplier = original.get("selected_supplier")
    rep_supplier = replayed.get("selected_supplier")
    orig_score = float(original.get("score", 0.0))
    rep_score = float(replayed.get("score", 0.0))
    same_supplier = orig_supplier == rep_supplier
    return {
        "same_supplier": same_supplier,
        "score_delta": round(rep_score - orig_score, 4),
        "decision_changed": not same_supplier,
    }


def simulate_all_traces(limit: int = SIMULATION_LIMIT) -> dict[str, Any]:
    """Replay recent traces with adaptive thresholds and aggregate outcomes.

    Args:
        limit: Maximum number of recent traces to replay.

    Returns:
        A summary dict with ``total``, ``changed_decisions``,
        ``avg_score_delta`` and ``improvement_rate`` plus the per-trace
        ``results`` list.  Failed replays are skipped without aborting the
        batch.
    """
    trace_ids = load_recent_trace_ids(limit=limit)
    results: list[dict[str, Any]] = []
    for tid in trace_ids:
        try:
            res = replay_trace(tid, use_adaptive=True)
        except (TraceNotFoundError, Exception):
            continue
        results.append(res)

    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "changed_decisions": 0,
            "avg_score_delta": 0.0,
            "improvement_rate": 0.0,
            "results": [],
        }

    changed = sum(1 for r in results if r["comparison"]["decision_changed"])
    deltas = [r["comparison"]["score_delta"] for r in results]
    avg_delta = round(sum(deltas) / total, 4)
    return {
        "total": total,
        "changed_decisions": changed,
        "avg_score_delta": avg_delta,
        "improvement_rate": round(changed / total, 4),
        "results": results,
    }
