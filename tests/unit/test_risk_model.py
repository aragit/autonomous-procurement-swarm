"""Unit tests for the Phase 6 deterministic risk model."""

from swarm.domain.risk import (
    DEFAULT_NO_HISTORY_RISK,
    RISK_LEVEL_CRITICAL,
    RISK_LEVEL_HIGH,
    RISK_LEVEL_MEDIUM,
    RiskAssessment,
    RiskLevel,
    carbon_risk_score,
    classify_risk,
    compute_risk_scores,
    delivery_risk_score,
    financial_risk_score,
    quality_risk_score,
)
from swarm.domain.supplier import SupplierPerformance


def test_financial_risk_is_zero_free_below_half_ceiling() -> None:
    assert financial_risk_score(0.0, 5_000_000.0) == 0.0
    assert financial_risk_score(2_500_000.0, 5_000_000.0) == 0.25


def test_financial_risk_saturates_at_ceiling_and_double() -> None:
    assert financial_risk_score(5_000_000.0, 5_000_000.0) == 0.5
    assert financial_risk_score(10_000_000.0, 5_000_000.0) == 1.0


def test_delivery_risk_defaults_neutral_without_history() -> None:
    assert delivery_risk_score(None) == DEFAULT_NO_HISTORY_RISK
    assert delivery_risk_score(SupplierPerformance("S")) == DEFAULT_NO_HISTORY_RISK


def test_delivery_risk_from_reliability() -> None:
    perf = SupplierPerformance("S")
    perf.total_orders = 5
    perf.successful_orders = 5
    assert delivery_risk_score(perf) == 0.0


def test_delivery_risk_poor_history_is_maximum() -> None:
    perf = SupplierPerformance("S")
    perf.total_orders = 4
    perf.successful_orders = 0
    assert delivery_risk_score(perf) == 1.0


def test_quality_risk_defaults_and_from_signals() -> None:
    assert quality_risk_score(None, None) == DEFAULT_NO_HISTORY_RISK
    assert quality_risk_score(None, 0.858) == round(1.0 - 0.858, 4)
    perf = SupplierPerformance("S")
    perf.total_orders = 3
    perf.average_quality_score = 0.4
    assert quality_risk_score(perf, 0.9) == round(1.0 - 0.4, 4)


def test_carbon_risk_unknown_is_neutral() -> None:
    assert carbon_risk_score(None, None, 12000.0) == DEFAULT_NO_HISTORY_RISK


def test_carbon_risk_against_esg_baseline() -> None:
    assert carbon_risk_score(1800.0, None, 12000.0) == round(1800.0 / 12000.0, 4)


def test_carbon_risk_against_hard_constraint() -> None:
    assert carbon_risk_score(1800.0, 800.0, 12000.0) == 1.0


def test_classify_risk_thresholds() -> None:
    assert classify_risk(0.0) == RiskLevel.LOW
    assert classify_risk(0.34) == RiskLevel.LOW
    assert classify_risk(RISK_LEVEL_MEDIUM) == RiskLevel.MEDIUM
    assert classify_risk(RISK_LEVEL_HIGH) == RiskLevel.HIGH
    assert classify_risk(0.79) == RiskLevel.HIGH
    assert classify_risk(RISK_LEVEL_CRITICAL) == RiskLevel.CRITICAL
    assert classify_risk(1.0) == RiskLevel.CRITICAL


def test_compute_risk_scores_baseline_matches_smoke() -> None:
    scores = compute_risk_scores(
        purchase_amount=984000.0,
        max_purchase_amount=5_000_000.0,
        performance=None,
        evaluation_score=0.858,
        carbon_per_unit=1800.0,
        max_carbon_per_unit=None,
        esg_baseline=12000.0,
    )
    assert scores == {
        "financial_risk_score": 0.0984,
        "delivery_risk_score": DEFAULT_NO_HISTORY_RISK,
        "quality_risk_score": round(1.0 - 0.858, 4),
        "carbon_risk_score": round(1800.0 / 12000.0, 4),
        "overall_risk_score": 0.1178,
    }
    assert classify_risk(scores["overall_risk_score"]) == RiskLevel.LOW


def test_risk_assessment_from_signals_is_serializable_and_rebuildable() -> None:
    risk = RiskAssessment.from_signals(
        supplier_id="MinerCorp_A",
        decision_id="dec-1",
        purchase_amount=984000.0,
        max_purchase_amount=5_000_000.0,
        performance=None,
        evaluation_score=0.858,
        carbon_per_unit=1800.0,
        max_carbon_per_unit=None,
        esg_baseline=12000.0,
    )
    summary = risk.to_summary()
    assert summary["supplier_id"] == "MinerCorp_A"
    assert summary["risk_level"] == "LOW"
    assert summary["purchase_amount"] == 984000.0
    # from_artifact consumes the artifact-data shape stored by the agent, which
    # nests the sub-scores under "risk_scores" (a property, not a model field).
    artifact_data = {**risk.model_dump(), "risk_scores": risk.risk_scores}
    rebuilt = RiskAssessment.from_artifact(artifact_data)
    assert rebuilt.risk_level == RiskLevel.LOW
    assert rebuilt.risk_id == risk.risk_id
    assert rebuilt.to_summary() == summary
