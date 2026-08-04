"""Unit tests for v0.9 Step 7: Deterministic LLM Explainability Layer.

Tests ``build_llm_explanation`` — the purely observational explanation builder
that explains accepted/rejected decisions without affecting behavior.
"""


from swarm.utils.llm_explainer import build_llm_explanation
from swarm.utils.llm_stability import TRUST_THRESHOLD

# --- Accepted case ---


def test_explanation_accepted_when_trust_above_threshold() -> None:
    explanation = build_llm_explanation(
        confidence=0.95,
        stability=0.85,
        trust=0.81,
        threshold=TRUST_THRESHOLD,
        adjustments={"price_weight_delta": -0.05},
    )
    assert explanation["decision"] == "accepted"
    assert "applied" in explanation["summary"]
    assert "0.81" in explanation["summary"]


def test_explanation_accepted_summary_format() -> None:
    explanation = build_llm_explanation(
        confidence=0.90,
        stability=0.90,
        trust=0.81,
        threshold=0.7,
        adjustments={"price_weight_delta": -0.03},
    )
    assert explanation["summary"] == (
        "LLM adjustments applied (trust 0.81 ≥ threshold 0.7)"
    )


# --- Rejected case ---


def test_explanation_rejected_when_trust_below_threshold() -> None:
    explanation = build_llm_explanation(
        confidence=0.95,
        stability=0.60,
        trust=0.57,
        threshold=TRUST_THRESHOLD,
        adjustments={},
    )
    assert explanation["decision"] == "rejected"
    assert "rejected" in explanation["summary"]
    assert "0.57" in explanation["summary"]


def test_explanation_rejected_summary_format() -> None:
    explanation = build_llm_explanation(
        confidence=0.95,
        stability=0.60,
        trust=0.57,
        threshold=0.7,
        adjustments={},
    )
    assert explanation["summary"] == (
        "LLM adjustments rejected (trust 0.57 < threshold 0.7)"
    )


# --- Boundary case ---


def test_explanation_boundary_trust_equals_threshold_is_accepted() -> None:
    explanation = build_llm_explanation(
        confidence=0.70,
        stability=1.0,
        trust=0.70,
        threshold=TRUST_THRESHOLD,
        adjustments={"price_weight_delta": -0.04},
    )
    assert explanation["decision"] == "accepted"


def test_explanation_just_below_threshold_is_rejected() -> None:
    explanation = build_llm_explanation(
        confidence=0.95,
        stability=0.73,
        trust=0.69,
        threshold=TRUST_THRESHOLD,
        adjustments={},
    )
    assert explanation["decision"] == "rejected"


# --- Determinism ---


def test_explanation_is_deterministic_same_inputs() -> None:
    e1 = build_llm_explanation(
        confidence=0.95,
        stability=0.85,
        trust=0.81,
        threshold=TRUST_THRESHOLD,
        adjustments={"price_weight_delta": -0.05, "delivery_weight_delta": 0.03},
    )
    e2 = build_llm_explanation(
        confidence=0.95,
        stability=0.85,
        trust=0.81,
        threshold=TRUST_THRESHOLD,
        adjustments={"price_weight_delta": -0.05, "delivery_weight_delta": 0.03},
    )
    assert e1 == e2


def test_explanation_deterministic_for_rejected() -> None:
    e1 = build_llm_explanation(
        confidence=0.50,
        stability=0.30,
        trust=0.15,
        threshold=TRUST_THRESHOLD,
        adjustments={},
    )
    e2 = build_llm_explanation(
        confidence=0.50,
        stability=0.30,
        trust=0.15,
        threshold=TRUST_THRESHOLD,
        adjustments={},
    )
    assert e1 == e2


# --- Metrics correctness ---


def test_explanation_metrics_preserved() -> None:
    explanation = build_llm_explanation(
        confidence=0.95,
        stability=0.85,
        trust=0.8075,
        threshold=0.7,
        adjustments={"price_weight_delta": -0.05},
    )
    metrics = explanation["metrics"]
    assert metrics["confidence"] == 0.95
    assert metrics["stability"] == 0.85
    assert metrics["trust"] == 0.8075
    assert metrics["threshold"] == 0.7


def test_explanation_metrics_rounded() -> None:
    explanation = build_llm_explanation(
        confidence=0.956789,
        stability=0.854321,
        trust=0.8154321,
        threshold=0.7,
        adjustments={},
    )
    metrics = explanation["metrics"]
    assert metrics["confidence"] == 0.9568
    assert metrics["stability"] == 0.8543
    assert metrics["trust"] == 0.8154


# --- Empty / None adjustments ---


def test_explanation_none_adjustments_normalized_to_empty() -> None:
    explanation = build_llm_explanation(
        confidence=0.95,
        stability=0.85,
        trust=0.81,
        threshold=TRUST_THRESHOLD,
        adjustments=None,
    )
    assert explanation["applied_adjustments"] == {}


def test_explanation_empty_adjustments_preserved() -> None:
    explanation = build_llm_explanation(
        confidence=0.95,
        stability=0.85,
        trust=0.81,
        threshold=TRUST_THRESHOLD,
        adjustments={},
    )
    assert explanation["applied_adjustments"] == {}


def test_explanation_adjustments_not_mutated() -> None:
    """The input adjustments dict should not be mutated."""
    original = {"price_weight_delta": -0.05}
    explanation = build_llm_explanation(
        confidence=0.95,
        stability=0.85,
        trust=0.81,
        threshold=TRUST_THRESHOLD,
        adjustments=original,
    )
    assert explanation["applied_adjustments"] == original
    assert original == {"price_weight_delta": -0.05}


# --- Structure ---


def test_explanation_has_all_required_keys() -> None:
    explanation = build_llm_explanation(
        confidence=0.95,
        stability=0.85,
        trust=0.81,
        threshold=TRUST_THRESHOLD,
        adjustments={"price_weight_delta": -0.05},
    )
    assert "decision" in explanation
    assert "summary" in explanation
    assert "metrics" in explanation
    assert "applied_adjustments" in explanation


def test_explanation_metrics_has_all_required_keys() -> None:
    explanation = build_llm_explanation(
        confidence=0.95,
        stability=0.85,
        trust=0.81,
        threshold=TRUST_THRESHOLD,
        adjustments={"price_weight_delta": -0.05},
    )
    metrics = explanation["metrics"]
    assert "confidence" in metrics
    assert "stability" in metrics
    assert "trust" in metrics
    assert "threshold" in metrics


def test_explanation_decision_only_two_values() -> None:
    accepted = build_llm_explanation(
        confidence=0.9, stability=0.9, trust=0.81, threshold=0.7, adjustments={}
    )
    rejected = build_llm_explanation(
        confidence=0.9, stability=0.5, trust=0.45, threshold=0.7, adjustments={}
    )
    assert accepted["decision"] == "accepted"
    assert rejected["decision"] == "rejected"
    assert accepted["decision"] != rejected["decision"]
