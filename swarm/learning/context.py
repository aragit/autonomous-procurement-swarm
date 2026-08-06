"""Deterministic context extraction from procurement traces (v1.1 Step 24).

Context is derived purely from existing trace data — no LLM, no external state.
This keeps the routing decision fully reproducible and replay-safe.
"""

from __future__ import annotations

from typing import Any

#: Budget thresholds for classification (in currency units from trace payload).
BUDGET_HIGH_THRESHOLD: float = 2_000_000.0
BUDGET_LOW_THRESHOLD: float = 500_000.0

#: Urgency classification based on target lead time (in days).
URGENCY_LOW_DAYS: int = 14


def extract_context(trace_input: dict[str, Any]) -> dict[str, Any]:
    """Extract a deterministic context dict from a procurement trace input.

    Context fields::

        budget_level: "low" | "medium" | "high"
            Based on the ``budget`` field in the trace payload.
        urgency: "low" | "high"
            Based on ``target_lead_time_days`` — if <= 14 days, "high".
        supplier_count: int
            Number of suppliers in the pool.

    All values are derived deterministically from the trace input. No LLM or
    external state is involved.

    Args:
        trace_input: The reconstructed requirement input dict (same shape used
            by :func:`swarm.simulation.replay_engine.extract_input`).

    Returns:
        A dict with keys ``budget_level``, ``urgency``, and
        ``supplier_count``.
    """
    budget = float(trace_input.get("budget", 0.0) or 0.0)
    if budget >= BUDGET_HIGH_THRESHOLD:
        budget_level = "high"
    elif budget <= BUDGET_LOW_THRESHOLD:
        budget_level = "low"
    else:
        budget_level = "medium"

    target_lead = int(trace_input.get("target_lead_time_days", 30) or 30)
    urgency = "high" if target_lead <= URGENCY_LOW_DAYS else "low"

    supplier_count = int(trace_input.get("supplier_count", 3) or 3)

    return {
        "budget_level": budget_level,
        "urgency": urgency,
        "supplier_count": supplier_count,
    }
