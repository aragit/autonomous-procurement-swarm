"""Adaptive threshold tuning from feedback signals (v1.0 Step 20).

Replaces static config thresholds with a runtime layer that adjusts
``confidence_threshold``, ``stability_threshold`, and ``trust_threshold``
based on accumulated outcome feedback.

Design principles:
- **Read-only**: Never mutates :mod:`swarm.config`.  Adaptation is a
  pure function of collected feedback.
- **Deterministic**: Sorted feedback → identical thresholds every call.
- **Safe**: Falls back to config defaults when learning is disabled,
  feedback is insufficient, or the DB is unavailable.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Any

from swarm.config import (
    ADAPTIVE_K,
    CONFIDENCE_THRESHOLD,
    ENABLE_LEARNING,
    MIN_FEEDBACK_SAMPLES,
    STABILITY_THRESHOLD,
    TRUST_THRESHOLD,
)
from swarm.storage.event_store import init_db, load_all_feedback

#: When set (e.g. by the replay engine), ``get_adaptive_thresholds`` returns
#: this value instead of reading the database. Enables deterministic
#: simulation: the same thresholds always reproduce the same replay.
_thresholds_override: ContextVar[dict[str, float] | None] = ContextVar(
    "swarm_thresholds_override", default=None
)


@contextlib.contextmanager
def override_thresholds(thresholds: dict[str, float]) -> Any:
    """Pin the active thresholds for the duration of the context.

    Used by the replay/simulation engine to guarantee deterministic re-runs
    that do not depend on live database reads.
    """
    token = _thresholds_override.set(thresholds)
    try:
        yield
    finally:
        _thresholds_override.reset(token)


def compute_adaptive_thresholds(feedback_list: list[dict[str, Any]]) -> dict[str, float]:
    """Compute threshold adjustments from a list of feedback records.

    Adaptation rules (deterministic, based on sorted feedback):

    1. **Confidence** — If high-confidence decisions frequently fail, lower
       the confidence threshold (be more conservative about LLM use).

    2. **Stability** — If low-stability rounds correlate with failures,
       raise the stability threshold (demand more stable consensus).

    3. **Trust** — If trust is high but outcomes are poor, lower the trust
       threshold (be more willing to override LLM when it's wrong).

    All values are clamped to ``[0.5, 0.95]``.

    Args:
        feedback_list: List of feedback dicts, each with ``outcome_score``
            (float 0-1), ``success`` (bool), and optionally ``trust`` (float).

    Returns:
        A dict with ``confidence_threshold``, ``stability_threshold``, and
        ``trust_threshold`` keys (all floats).
    """
    if not feedback_list or not ENABLE_LEARNING:
        return {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "stability_threshold": STABILITY_THRESHOLD,
            "trust_threshold": TRUST_THRESHOLD,
        }

    # Sort by timestamp for deterministic ordering
    sorted_feedback = sorted(
        feedback_list,
        key=lambda f: f.get("created_at", ""),
    )

    n = len(sorted_feedback)
    if n < MIN_FEEDBACK_SAMPLES:
        return _default_thresholds()

    failures = [f for f in sorted_feedback if not f.get("success", False)]
    failure_ratio = len(failures) / n if n > 0 else 0.0

    # --- Rule 1: Confidence calibration ---
    # High confidence + failure → overconfident, lower threshold
    # We look at failures with high outcome_score (they tried hard but failed)
    overconfident_failures = [
        f for f in failures
        if f.get("outcome_score", 0.0) < 0.5
    ]
    overconfident_ratio = len(overconfident_failures) / len(failures) if failures else 0.0

    # If overconfident failures are common, lower confidence threshold
    confidence_delta = ADAPTIVE_K * overconfident_ratio
    confidence_threshold = max(0.5, CONFIDENCE_THRESHOLD - confidence_delta)

    # --- Rule 2: Stability sensitivity ---
    # If failures are common, increase stability threshold (demand more stability)
    stability_delta = ADAPTIVE_K * failure_ratio
    stability_threshold = min(0.95, STABILITY_THRESHOLD + stability_delta)

    # --- Rule 3: Trust adjustment ---
    # If trust scores are high but outcomes are poor, lower trust threshold
    # (the system should be more skeptical of high-trust LLM outputs)
    trust_scores = [
        f.get("trust", 0.7) for f in sorted_feedback if "trust" in f
    ]
    if trust_scores and failure_ratio > 0.5:
        avg_trust = sum(trust_scores) / len(trust_scores)
        # High trust + high failure → distrust, lower threshold
        trust_delta = ADAPTIVE_K * failure_ratio * (avg_trust - 0.5)
        trust_threshold = max(0.5, TRUST_THRESHOLD - trust_delta)
    else:
        trust_threshold = TRUST_THRESHOLD

    # Clamp all thresholds
    confidence_threshold = min(max(confidence_threshold, 0.5), 0.95)
    stability_threshold = min(max(stability_threshold, 0.5), 0.95)
    trust_threshold = min(max(trust_threshold, 0.5), 0.95)

    return {
        "confidence_threshold": round(confidence_threshold, 4),
        "stability_threshold": round(stability_threshold, 4),
        "trust_threshold": round(trust_threshold, 4),
    }


def get_config_thresholds() -> dict[str, float]:
    """Return the static, config-defined thresholds (ignoring all feedback).

    This is the "static" threshold set used when learning is disabled or when a
    replay/simulation explicitly requests config-only thresholds.
    """
    return _default_thresholds()


def get_adaptive_thresholds() -> dict[str, float]:
    """Get current adaptive thresholds from accumulated feedback.

    Behavior:
    - If an explicit override is set (via :func:`override_thresholds`) →
      returns that value (deterministic, DB-free). Used by the replay engine.
    - If ``ENABLE_LEARNING`` is False → returns config defaults.
    - If feedback DB has fewer than ``MIN_FEEDBACK_SAMPLES`` rows →
      returns config defaults.
    - Otherwise → computes thresholds from all feedback.

    This is the main entry point called by runtime code.

    Returns:
        A dict with ``confidence_threshold``, ``stability_threshold``,
        and ``trust_threshold`` keys.
    """
    override = _thresholds_override.get()
    if override is not None:
        return dict(override)

    if not ENABLE_LEARNING:
        return _default_thresholds()

    try:
        init_db()
        feedback_list = load_all_feedback()
    except Exception:
        # If DB is unavailable, fall back to defaults
        return _default_thresholds()

    if len(feedback_list) < MIN_FEEDBACK_SAMPLES:
        return _default_thresholds()

    return compute_adaptive_thresholds(feedback_list)


def _default_thresholds() -> dict[str, float]:
    return {
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "stability_threshold": STABILITY_THRESHOLD,
        "trust_threshold": TRUST_THRESHOLD,
    }
