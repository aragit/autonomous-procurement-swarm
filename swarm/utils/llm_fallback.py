"""Deterministic LLM fallback and failure-mode evaluation (v0.9 Step 9).

Guarantees safe, deterministic behavior when LLM signals are absent, weak,
or invalid. The evaluation is a strict ordered check — the first failing
condition wins, and the system always reverts to the canonical strategy.

Evaluation order:

1. **No LLM data** — no completions exist → ``no_llm_data``
2. **Low confidence** — consensus disagrees → ``low_confidence``
3. **Low stability** — LLM suggestions drift over time → ``low_stability``
4. **Low trust** — confidence × stability below threshold → ``low_trust``
5. **Valid** — all gates pass → ``accepted``
"""

from __future__ import annotations

from typing import Any


def evaluate_llm_usage(
    has_completions: bool,
    confidence: float,
    stability: float,
    trust: float,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate whether LLM influence should be used, with a deterministic reason.

    Checks conditions in a strict order — the first failing condition
    determines the ``reason``. Only when all conditions pass is
    ``use_llm`` set to ``True``.

    Args:
        has_completions: Whether any LLM completions exist for this round.
        confidence: The consensus confidence score (0 → 1).
        stability: The temporal stability score (0 → 1).
        trust: The computed trust score = confidence × stability.
        threshold: The minimum value for each gate.

    Returns:
        A dict with ``use_llm`` (bool) and ``reason`` (str).
    """
    if not has_completions:
        return {"use_llm": False, "reason": "no_llm_data"}

    if confidence < threshold:
        return {"use_llm": False, "reason": "low_confidence"}

    if stability < threshold:
        return {"use_llm": False, "reason": "low_stability"}

    if trust < threshold:
        return {"use_llm": False, "reason": "low_trust"}

    return {"use_llm": True, "reason": "accepted"}
