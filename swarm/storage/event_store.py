"""Persistent event store for execution state (v1.0 Step 18).

Provides an append-only SQLite-backed store for events, artifacts, and
LLM consensus history.  All writes are deterministic and the store
is never mutated after write — it is strictly append-only.

Tables::

    events      — every Event dispatched during a trace
    artifacts   — every Artifact created during a trace
    llm_history — per-round LLM consensus records for drift/metrics
    feedback    — outcome / user feedback linked to a trace
    policies    — learned candidate policies (v1.1 Step 22): thresholds + weights

The store is keyed by ``trace_id`` (the correlation identifier for
an entire procurement run).  Within a trace, ``correlation_id``
provides an additional layer of scoping (e.g. ``"TRACE-CONV"``).
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from typing import Any

from swarm.config import DB_PATH

_DB_PATH = DB_PATH


def _get_conn() -> sqlite3.Connection:
    return sqlite3.connect(_DB_PATH)


def _serialize(data: Any) -> str:
    """Serialize to a deterministic JSON string (sorted keys)."""
    return json.dumps(data, sort_keys=True)


def _deserialize(raw: str) -> Any:
    """Deserialize from a JSON string."""
    if not raw:
        return None
    return json.loads(raw)


def init_db(db_path: str | None = None) -> None:
    """Create tables if they do not already exist.

    Args:
        db_path: Optional override for the database file path.
            When provided, updates the module-level path used by all
            subsequent operations.
    """
    global _DB_PATH
    if db_path is not None:
        _DB_PATH = db_path

    conn = _get_conn()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS llm_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                round INTEGER NOT NULL,
                confidence REAL,
                stability REAL,
                trust REAL,
                decision_reason TEXT,
                payload TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                outcome_score REAL,
                success BOOLEAN,
                latency_ms REAL,
                user_feedback TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS policies (
                version TEXT PRIMARY KEY,
                signature TEXT NOT NULL,
                thresholds TEXT NOT NULL,
                weights TEXT NOT NULL,
                strategy TEXT NOT NULL,
                metric REAL,
                success_rate REAL,
                avg_score REAL,
                feedback_success_rate REAL,
                feedback_outcome_score REAL,
                decision_stability REAL,
                active INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_policies_active_only
                ON policies(active) WHERE active = 1;
            """
        )
        conn.commit()
    finally:
        conn.close()
def store_event(trace_id: str, event_type: str, payload: dict[str, Any]) -> None:
    """Append an event row for the given trace.

    Args:
        trace_id: The trace identifier.
        event_type: The event type string.
        payload: The event payload dict (serialized deterministically).
    """
    init_db()
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO events (trace_id, event_type, payload) VALUES (?, ?, ?)",
            (trace_id, event_type, _serialize(payload)),
        )
        conn.commit()
    finally:
        conn.close()


def store_artifact(trace_id: str, artifact_type: str, data: dict[str, Any]) -> None:
    """Append an artifact row for the given trace.

    Args:
        trace_id: The trace identifier.
        artifact_type: The type of the artifact (e.g. ``"strategy"``).
        data: The artifact data dict (serialized deterministically).
    """
    init_db()
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO artifacts (trace_id, artifact_type, data) VALUES (?, ?, ?)",
            (trace_id, artifact_type, _serialize(data)),
        )
        conn.commit()
    finally:
        conn.close()


def store_llm_record(trace_id: str, record: dict[str, Any]) -> None:
    """Append an LLM consensus history row.

    Args:
        trace_id: The trace identifier.
        record: A dict with keys ``round``, ``confidence``, ``stability``,
            ``trust``, ``decision_reason``, and ``payload``.
    """
    init_db()
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO llm_history "
            "(trace_id, round, confidence, stability, trust, decision_reason, payload) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                trace_id,
                record.get("round", 0),
                record.get("confidence"),
                record.get("stability"),
                record.get("trust"),
                record.get("decision_reason"),
                _serialize(record.get("payload", {})),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_state(trace_id: str) -> dict[str, Any]:
    """Reconstruct the persisted state for a given trace.

    Returns:
        A dict with keys ``events``, ``artifacts``, and ``llm_history``,
        each a list of dicts ordered by insertion (oldest first).
        If no data exists for the trace, all lists are empty.
    """
    init_db()
    conn = _get_conn()
    try:
        conn.row_factory = sqlite3.Row

        events_rows = conn.execute(
            "SELECT event_type, payload, created_at FROM events WHERE trace_id = ? ORDER BY id ASC",
            (trace_id,),
        ).fetchall()
        events: list[dict[str, Any]] = [
            {
                "event_type": row["event_type"],
                "payload": _deserialize(row["payload"]),
                "created_at": row["created_at"],
            }
            for row in events_rows
        ]

        artifact_rows = conn.execute(
            "SELECT artifact_type, data, created_at FROM artifacts "
            "WHERE trace_id = ? ORDER BY id ASC",
            (trace_id,),
        ).fetchall()
        artifacts: list[dict[str, Any]] = [
            {
                "artifact_type": row["artifact_type"],
                "data": _deserialize(row["data"]),
                "created_at": row["created_at"],
            }
            for row in artifact_rows
        ]

        history_rows = conn.execute(
            "SELECT round, confidence, stability, trust, decision_reason, payload "
            "FROM llm_history WHERE trace_id = ? ORDER BY id ASC",
            (trace_id,),
        ).fetchall()
        llm_history: list[dict[str, Any]] = [
            {
                "round": row["round"],
                "confidence": row["confidence"],
                "stability": row["stability"],
                "trust": row["trust"],
                "decision_reason": row["decision_reason"],
                "payload": _deserialize(row["payload"]),
            }
            for row in history_rows
        ]

        return {
            "events": events,
            "artifacts": artifacts,
            "llm_history": llm_history,
        }
    finally:
        conn.close()


def store_feedback(
    trace_id: str,
    outcome_score: float,
    success: bool,
    latency_ms: float,
    user_feedback: str | None = None,
) -> None:
    """Append a feedback row for the given trace.

    Feedback is append-only: each call creates a new row.  If feedback
    already exists for the trace, the latest row takes precedence
    when read back via :func:`load_feedback`.

    Args:
        trace_id: The trace identifier.
        outcome_score: Numerical outcome score (0.0–1.0).
        success: Whether the procurement was successful.
        latency_ms: Total execution latency in milliseconds.
        user_feedback: Optional free-text feedback from the user.
    """
    init_db()
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO feedback (trace_id, outcome_score, success, latency_ms, user_feedback) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                trace_id,
                outcome_score,
                1 if success else 0,
                latency_ms,
                user_feedback,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_feedback(trace_id: str) -> dict[str, Any] | None:
    """Load the most recent feedback for a trace.

    Returns ``None`` if no feedback exists.

    Returns:
        A dict with ``outcome_score``, ``success``, ``latency_ms``,
        and ``user_feedback`` keys, or ``None``.
    """
    init_db()
    conn = _get_conn()
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT outcome_score, success, latency_ms, user_feedback "
            "FROM feedback WHERE trace_id = ? ORDER BY id DESC LIMIT 1",
            (trace_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "outcome_score": row["outcome_score"],
            "success": bool(row["success"]),
            "latency_ms": row["latency_ms"],
            "user_feedback": row["user_feedback"],
        }
    finally:
        conn.close()


def load_full_trace(trace_id: str) -> dict[str, Any]:
    """Reconstruct the complete persisted state for a given trace.

    Combines events, artifacts, LLM history, and feedback into a single
    dict.  This is the canonical snapshot used by the learning layer.

    Returns:
        A dict with keys ``events``, ``artifacts``, ``llm_history``,
        and ``feedback``.  ``feedback`` is ``None`` when no feedback
        has been recorded.
    """
    base = load_state(trace_id)
    feedback = load_feedback(trace_id)
    return {
        "events": base["events"],
        "artifacts": base["artifacts"],
        "llm_history": base["llm_history"],
        "feedback": feedback,
    }


def load_all_feedback() -> list[dict[str, Any]]:
    """Load all feedback rows, sorted by creation time (oldest first).

    Used by the adaptive policy engine to compute threshold adjustments
    from the full feedback history.

    Returns:
        A list of dicts, each with ``trace_id``, ``outcome_score``,
        ``success``, ``latency_ms``, ``user_feedback``, and ``created_at``.
    """
    init_db()
    conn = _get_conn()
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT trace_id, outcome_score, success, latency_ms, user_feedback, created_at "
            "FROM feedback ORDER BY created_at ASC, id ASC"
        ).fetchall()
        return [
            {
                "trace_id": row["trace_id"],
                "outcome_score": row["outcome_score"],
                "success": bool(row["success"]),
                "latency_ms": row["latency_ms"],
                "user_feedback": row["user_feedback"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
    finally:
        conn.close()


def load_recent_trace_ids(limit: int = 100) -> list[str]:
    """Return the most recent ``limit`` trace IDs that have persisted events.

    Ordered by insertion (newest first) so the simulation engine can replay
    the latest executions. Traces with no persisted events are excluded.
    """
    init_db()
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT trace_id FROM events GROUP BY trace_id ORDER BY MAX(id) DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [row[0] for row in rows]
    finally:
        conn.close()


def load_policy_version_ids() -> list[str]:
    """Return all stored policy versions (newest first).

    The learner merges raw traces + raw feedback itself; this helper exposes
    only the opaque version identifiers for candidates already in the store.
    """
    init_db()
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT version FROM policies ORDER BY created_at DESC"
        ).fetchall()
        return [row[0] for row in rows]
    finally:
        conn.close()


def store_policy(
    *,
    version: str,
    signature: str,
    thresholds: dict[str, float],
    weights: dict[str, float],
    strategy: dict[str, Any] | None = None,
    metric: float | None = None,
    success_rate: float | None = None,
    avg_score: float | None = None,
    feedback_success_rate: float | None = None,
    feedback_outcome_score: float | None = None,
    decision_stability: float | None = None,
    active: bool = False,
) -> None:
    """Persist a policy row. Idempotent on the (version) primary key.

    If a row with the same ``version`` already exists it is left untouched
    (preserving the first writer's metric/activation). This keeps
    ``/policies/learn`` idempotent: the same candidate version is never
    double-counted.

    Args:
        version: Deterministic 12-char policy version (``sha256(signature)[:12]``).
        signature: The full canonical parameter signature used to derive version.
        thresholds: The ``confidence/stability/trust`` threshold dict.
        weights: The strategy ``price/score/carbon`` weight dict.
        strategy: The strategy dict (e.g. ``{"type": "balanced"}``).
        metric: Objective metric from candidate evaluation.
        success_rate: Decision stability measured during evaluation.
        avg_score: Average replayed score measured during evaluation.
        feedback_success_rate: Fraction of traces marked successful in feedback.
        feedback_outcome_score: Average outcome_score from feedback.
        decision_stability: Fraction of replays reproducing the original supplier.
        active: Whether this policy is currently active.
    """
    if strategy is None:
        from swarm.config import DEFAULT_STRATEGY_TYPE
        strategy = {"type": DEFAULT_STRATEGY_TYPE}
    init_db()
    conn = _get_conn()
    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO policies
                (version, signature, thresholds, weights, strategy, metric,
                 success_rate, avg_score, feedback_success_rate,
                 feedback_outcome_score, decision_stability, active, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                version,
                signature,
                _serialize(thresholds),
                _serialize(weights),
                _serialize(strategy),
                metric,
                success_rate,
                avg_score,
                feedback_success_rate,
                feedback_outcome_score,
                decision_stability,
                1 if active else 0,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_policy(version: str) -> dict[str, Any] | None:
    """Load a single policy row by version.

    Returns ``None`` when no policy with that version exists.
    """
    init_db()
    conn = _get_conn()
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT version, signature, thresholds, weights, strategy, "
            "metric, success_rate, avg_score, feedback_success_rate, "
            "feedback_outcome_score, decision_stability, active, created_at "
            "FROM policies WHERE version = ?",
            (version,),
        ).fetchone()
        if row is None:
            return None
        return {
            "version": row["version"],
            "signature": row["signature"],
            "thresholds": _deserialize(row["thresholds"]),
            "weights": _deserialize(row["weights"]),
            "strategy": _deserialize(row["strategy"]),
            "metric": row["metric"],
            "success_rate": row["success_rate"],
            "avg_score": row["avg_score"],
            "feedback_success_rate": row["feedback_success_rate"],
            "feedback_outcome_score": row["feedback_outcome_score"],
            "decision_stability": row["decision_stability"],
            "active": bool(row["active"]),
            "created_at": row["created_at"],
        }
    finally:
        conn.close()


def load_active_policy() -> dict[str, Any] | None:
    """Load the currently-active policy row, or ``None`` if none is active.

    The active policy is the most recently promoted candidate. When none has
    been promoted, the runtime falls back to config defaults via
    :func:`~swarm.learning.adaptive_policy.get_adaptive_thresholds`.
    """
    init_db()
    conn = _get_conn()
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT version, signature, thresholds, weights, strategy, "
            "metric, success_rate, avg_score, feedback_success_rate, "
            "feedback_outcome_score, decision_stability, active, created_at "
            "FROM policies WHERE active = 1 ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return {
            "version": row["version"],
            "signature": row["signature"],
            "thresholds": _deserialize(row["thresholds"]),
            "weights": _deserialize(row["weights"]),
            "strategy": _deserialize(row["strategy"]),
            "metric": row["metric"],
            "success_rate": row["success_rate"],
            "avg_score": row["avg_score"],
            "feedback_success_rate": row["feedback_success_rate"],
            "feedback_outcome_score": row["feedback_outcome_score"],
            "decision_stability": row["decision_stability"],
            "active": bool(row["active"]),
            "created_at": row["created_at"],
        }
    finally:
        conn.close()


def set_policy_active(version: str) -> bool:
    """Atomically promote a policy version to active.

    Uses a single transaction to (1) deactivate any currently-active policy and
    (2) mark ``version`` as active. The ``policies.active`` partial unique
    constraint (UNIQUE(active) WHERE active = 1) guarantees at most one active
    policy, so concurrent promote requests for the same or different versions
    are serialized safely.

    Returns:
        True if the policy was activated, False if it did not exist.
    """
    init_db()
    conn = _get_conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute("SELECT 1 FROM policies WHERE version = ?", (version,)).fetchone()
        if existing is None:
            conn.execute("ROLLBACK")
            return False
        conn.execute("UPDATE policies SET active = 0 WHERE active = 1")
        conn.execute("UPDATE policies SET active = 1 WHERE version = ?", (version,))
        conn.commit()
        return True
    except Exception:
        with contextlib.suppress(Exception):
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
