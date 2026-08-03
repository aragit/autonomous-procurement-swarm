"""Deterministic risk assessment models for the Phase 6 procurement swarm.

A :class:`RiskAssessment` is a pure function of a decision's signals — purchase
amount, supplier delivery/quality history, quote carbon and the active governance
policy — so the same decision always yields the same risk record. There is no
LLM and no autonomous learning: the four sub-scores are fixed deterministic
mappings and ``overall_risk_score`` is a fixed weighted blend.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from swarm.domain.supplier import SupplierPerformance

#: Weighted blend of the four risk sub-scores. Sums to 1.0.
RISK_SCORE_WEIGHTS: dict[str, float] = {
    "financial": 0.35,
    "delivery": 0.25,
    "quality": 0.20,
    "carbon": 0.20,
}

#: Risk score applied to a supplier with no recorded history. Kept small so a
#: first-time supplier is not penalized as medium-risk by default.
DEFAULT_NO_HISTORY_RISK = 0.1

#: Risk-level thresholds on the blended ``overall_risk_score``.
RISK_LEVEL_MEDIUM = 0.35
RISK_LEVEL_HIGH = 0.60
RISK_LEVEL_CRITICAL = 0.80


class RiskLevel(StrEnum):
    """Ordinal risk levels used by governance to authorize or gate decisions."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


def clamp01(value: float) -> float:
    """Clamp a score into the [0, 1] range."""
    return min(1.0, max(0.0, value))


def financial_risk_score(
    purchase_amount: float, max_purchase_amount: float
) -> float:
    """Risk from the purchase size relative to the policy ceiling.

    A purchase at the ceiling is moderate risk (0.5); twice the ceiling is the
    maximum (1.0). Zero purchase amount is no risk.
    """
    ceiling = max(float(max_purchase_amount), 1.0)
    ratio = float(purchase_amount) / ceiling
    return round(clamp01(ratio * 0.5), 4)


def delivery_risk_score(performance: SupplierPerformance | None) -> float:
    """Risk from supplier delivery history.

    ``1 - delivery_reliability`` when history exists; a small neutral value
    otherwise so an unknown supplier is not flagged as risky by default.
    """
    if performance is None or performance.total_orders == 0:
        return DEFAULT_NO_HISTORY_RISK
    return round(1.0 - performance.delivery_reliability, 4)


def quality_risk_score(
    performance: SupplierPerformance | None, evaluation_score: float | None
) -> float:
    """Risk from supplier quality history.

    Uses the recorded average quality score when history exists, falls back to the
    evaluation score (a poor composite score is a poor-quality signal), and
    otherwise returns a neutral default.
    """
    if performance is not None and performance.total_orders > 0:
        return round(1.0 - performance.average_quality_score, 4)
    if evaluation_score is not None:
        return round(1.0 - float(evaluation_score), 4)
    return DEFAULT_NO_HISTORY_RISK


def carbon_risk_score(
    carbon_per_unit: float | None,
    max_carbon_per_unit: float | None,
    esg_baseline: float,
) -> float:
    """Risk from the supplier's per-unit carbon footprint.

    When a hard carbon constraint is present the footprint is measured against it;
    otherwise the material's ESG baseline is the ceiling. At the ceiling the risk
    is 1.0 (maximum). An unknown footprint yields a neutral default.
    """
    if carbon_per_unit is None:
        return DEFAULT_NO_HISTORY_RISK
    if max_carbon_per_unit is not None and float(max_carbon_per_unit) > 0:
        ceiling = float(max_carbon_per_unit)
    else:
        ceiling = float(esg_baseline) if float(esg_baseline) > 0 else 1.0
    per_unit = float(carbon_per_unit)
    return round(clamp01(per_unit / ceiling), 4)


def compute_risk_scores(
    *,
    purchase_amount: float,
    max_purchase_amount: float,
    performance: SupplierPerformance | None,
    evaluation_score: float | None,
    carbon_per_unit: float | None,
    max_carbon_per_unit: float | None,
    esg_baseline: float,
) -> dict[str, float]:
    """Compute all four risk sub-scores and the blended overall score."""
    financial = financial_risk_score(purchase_amount, max_purchase_amount)
    delivery = delivery_risk_score(performance)
    quality = quality_risk_score(performance, evaluation_score)
    carbon = carbon_risk_score(carbon_per_unit, max_carbon_per_unit, esg_baseline)
    overall = round(
        sum(
            RISK_SCORE_WEIGHTS[key] * value
            for key, value in (
                ("financial", financial),
                ("delivery", delivery),
                ("quality", quality),
                ("carbon", carbon),
            )
        ),
        4,
    )
    return {
        "financial_risk_score": financial,
        "delivery_risk_score": delivery,
        "quality_risk_score": quality,
        "carbon_risk_score": carbon,
        "overall_risk_score": overall,
    }


def classify_risk(overall: float) -> RiskLevel:
    """Map a blended ``overall_risk_score`` to an ordinal :class:`RiskLevel`."""
    if overall >= RISK_LEVEL_CRITICAL:
        return RiskLevel.CRITICAL
    if overall >= RISK_LEVEL_HIGH:
        return RiskLevel.HIGH
    if overall >= RISK_LEVEL_MEDIUM:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


class RiskAssessment(BaseModel):
    """Deterministic risk record produced for a procurement decision.

    ``data`` contract::

        {
            "risk_id": str,
            "supplier_id": str,
            "decision_id": str,
            "purchase_amount": float,
            "financial_risk_score": float,
            "delivery_risk_score": float,
            "quality_risk_score": float,
            "carbon_risk_score": float,
            "overall_risk_score": float,
            "risk_level": str,
            "created_at": str,
        }
    """

    risk_id: str = Field(default_factory=lambda: _uuid_hex())
    supplier_id: str = ""
    decision_id: str = ""
    purchase_amount: float = 0.0
    financial_risk_score: float = 0.0
    delivery_risk_score: float = 0.0
    quality_risk_score: float = 0.0
    carbon_risk_score: float = 0.0
    overall_risk_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def risk_scores(self) -> dict[str, float]:
        """The four sub-scores as a serializable dict."""
        return {
            "financial_risk_score": self.financial_risk_score,
            "delivery_risk_score": self.delivery_risk_score,
            "quality_risk_score": self.quality_risk_score,
            "carbon_risk_score": self.carbon_risk_score,
            "overall_risk_score": self.overall_risk_score,
        }

    @classmethod
    def from_signals(
        cls,
        *,
        supplier_id: str,
        decision_id: str,
        purchase_amount: float,
        max_purchase_amount: float,
        performance: SupplierPerformance | None,
        evaluation_score: float | None,
        carbon_per_unit: float | None,
        max_carbon_per_unit: float | None,
        esg_baseline: float,
    ) -> RiskAssessment:
        """Build a fully-populated, deterministic risk assessment."""
        scores = compute_risk_scores(
            purchase_amount=purchase_amount,
            max_purchase_amount=max_purchase_amount,
            performance=performance,
            evaluation_score=evaluation_score,
            carbon_per_unit=carbon_per_unit,
            max_carbon_per_unit=max_carbon_per_unit,
            esg_baseline=esg_baseline,
        )
        return cls(
            supplier_id=supplier_id,
            decision_id=decision_id,
            purchase_amount=round(float(purchase_amount), 2),
            financial_risk_score=scores["financial_risk_score"],
            delivery_risk_score=scores["delivery_risk_score"],
            quality_risk_score=scores["quality_risk_score"],
            carbon_risk_score=scores["carbon_risk_score"],
            overall_risk_score=scores["overall_risk_score"],
            risk_level=classify_risk(scores["overall_risk_score"]),
        )

    def to_summary(self) -> dict[str, Any]:
        """Serializable summary used by the API and tests."""
        return {
            "risk_id": self.risk_id,
            "supplier_id": self.supplier_id,
            "decision_id": self.decision_id,
            "purchase_amount": self.purchase_amount,
            "financial_risk_score": self.financial_risk_score,
            "delivery_risk_score": self.delivery_risk_score,
            "quality_risk_score": self.quality_risk_score,
            "carbon_risk_score": self.carbon_risk_score,
            "overall_risk_score": self.overall_risk_score,
            "risk_level": self.risk_level.value,
            "created_at": self.created_at,
        }

    @classmethod
    def from_artifact(cls, data: dict[str, Any]) -> RiskAssessment:
        """Rebuild a :class:`RiskAssessment` from a ``RiskAssessmentArtifact`` data dict."""
        scores = data.get("risk_scores", {}) or {}
        return cls(
            risk_id=data.get("risk_id", ""),
            supplier_id=data.get("supplier_id", ""),
            decision_id=data.get("decision_id", ""),
            purchase_amount=float(data.get("purchase_amount", 0.0)),
            financial_risk_score=float(scores.get("financial_risk_score", 0.0)),
            delivery_risk_score=float(scores.get("delivery_risk_score", 0.0)),
            quality_risk_score=float(scores.get("quality_risk_score", 0.0)),
            carbon_risk_score=float(scores.get("carbon_risk_score", 0.0)),
            overall_risk_score=float(scores.get("overall_risk_score", 0.0)),
            risk_level=RiskLevel(data.get("risk_level", RiskLevel.LOW.value)),
            created_at=data.get("created_at", ""),
        )


def _uuid_hex() -> str:
    from uuid import uuid4

    return uuid4().hex
