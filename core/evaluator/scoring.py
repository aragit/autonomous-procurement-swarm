"""Multi-criteria utility scoring for supplier bid evaluation."""

from typing import List, Dict
from pydantic import BaseModel, Field, field_validator, model_validator

class EvaluationWeights(BaseModel):
    price_weight: float = Field(default=0.40, ge=0.0, le=1.0)
    lead_time_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    esg_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    reliability_weight: float = Field(default=0.15, ge=0.0, le=1.0)

    @field_validator("price_weight")
    @classmethod
    def weights_sum_to_one(cls, v, info):
        # Pydantic v2: info.data contains already-validated fields
        # We'll do a post-validation check instead
        return v

    @model_validator(mode='after')
    def check_sum(self):
        total = (self.price_weight + self.lead_time_weight +
                 self.esg_weight + self.reliability_weight)
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        return self

class MultiCriteriaEvaluator:
    """
    Normalized multi-attribute utility function.
    Score range: [0.0, 1.0] where 1.0 = perfect bid.
    """

    def __init__(
        self,
        weights: EvaluationWeights,
        esg_baselines: Dict[str, float],
    ):
        self.weights = weights
        self.esg_baselines = esg_baselines

    def _score_price(self, unit_price: float, market_spot_price: float) -> float:
        """
        Price score: 1.0 at or below spot, linear decay above.
        At 2× spot, score = 0.0.
        """
        ratio = unit_price / max(market_spot_price, 1e-5)
        return max(0.0, 1.0 - (ratio / 2.0))

    def _score_lead_time(self, lead_time_days: int, target_lead_time: int) -> float:
        """
        Lead time score: 1.0 at or below target, linear decay above.
        At 2× target, score = 0.0.
        """
        if lead_time_days <= target_lead_time:
            return 1.0
        excess = lead_time_days - target_lead_time
        return max(0.0, 1.0 - (excess / max(target_lead_time, 1)))

    def _score_esg(self, carbon_footprint_kg: float, material: str) -> float:
        """
        ESG score: 1.0 at zero carbon, linear decay to baseline.
        Above baseline, score = 0.0.
        """
        baseline = self.esg_baselines.get(material, 2000.0)
        if baseline <= 0:
            return 1.0
        return max(0.0, 1.0 - (carbon_footprint_kg / baseline))

    def _score_reliability(self, reliability_score: float) -> float:
        """Reliability score is already [0, 1]."""
        return max(0.0, min(1.0, reliability_score))

    def score_bid(
        self,
        bid,  # BidPayload object
        market_spot_price: float,
        target_lead_time: int,
        material: str,
    ) -> float:
        """Compute composite weighted score for a single bid."""
        p = self._score_price(bid.unit_price, market_spot_price)
        lt = self._score_lead_time(bid.lead_time_days, target_lead_time)
        esg = self._score_esg(bid.carbon_footprint_kg, material)
        rel = self._score_reliability(bid.reliability_score)

        total = (
            self.weights.price_weight * p +
            self.weights.lead_time_weight * lt +
            self.weights.esg_weight * esg +
            self.weights.reliability_weight * rel
        )
        return round(total, 4)

    def rank_bids(
        self,
        bids: List,  # List[BidPayload]
        market_spot_price: float,
        target_lead_time: int,
        material: str,
    ) -> List[tuple]:
        """
        Score and rank bids descending by composite score.
        Returns: List[(score, bid)] sorted highest first.
        """
        scored = [
            (self.score_bid(b, market_spot_price, target_lead_time, material), b)
            for b in bids
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored
