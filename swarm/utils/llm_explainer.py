"""Deterministic explainability layer for LLM influence decisions (v0.9 Step 7).

Produces a human-readable, fully deterministic explanation of whether LLM
adjustments were accepted or rejected — based only on the numeric trust score
and threshold. No LLM calls, no free text, no interpretation. The explanation
is purely observational: it does not affect any decision or branch in the
system's logic.
"""

from __future__ import annotations

from typing import Any


def build_llm_explanation(
    confidence: float,
    stability: float,
    trust: float,
    threshold: float,
    adjustments: dict[str, float] | None,
) -> dict[str, Any]:
    """Build a deterministic explanation dict for an LLM influence decision.

    Args:
        confidence: The consensus confidence score (0 → 1).
        stability: The temporal stability score (0 → 1).
        trust: The computed trust score = confidence × stability.
        threshold: The minimum trust required for adjustments to be applied.
        adjustments: The validated adjustments dict (may be empty).

    Returns:
        A dict with ``decision``, ``summary``, ``metrics``, and
        ``applied_adjustments``. The ``summary`` string follows a strict
        template — no free-form text — so output is deterministic and
        replay-safe.
    """
    if adjustments is None:
        adjustments = {}

    if trust >= threshold:
        decision = "accepted"
        summary = f"LLM adjustments applied (trust {trust:.2f} ≥ threshold {threshold:.1f})"
    else:
        decision = "rejected"
        summary = f"LLM adjustments rejected (trust {trust:.2f} < threshold {threshold:.1f})"

    return {
        "decision": decision,
        "summary": summary,
        "metrics": {
            "confidence": round(confidence, 4),
            "stability": round(stability, 4),
            "trust": round(trust, 4),
            "threshold": threshold,
        },
        "applied_adjustments": dict(adjustments),
    }
