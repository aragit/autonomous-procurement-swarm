"""Deterministic execution strategies for the Phase 4 procurement swarm.

A :class:`Strategy` turns the requirement's intent into a set of scoring
weights used by :class:`EvaluationAgent` (``price`` / ``score`` / ``carbon``).
Selection is a pure rule over the requirement's constraints — no randomness, no
LLM — so the same requirement always selects the same strategy and yields the
same ranking.
"""

from typing import Any

from pydantic import BaseModel, Field, model_validator

#: A budget is "tight" when it cannot cover half of the theoretical maximum
#: spend (``quantity * max_unit_price``). Ratios below this select the
#: cost-optimized strategy.
BUDGET_TIGHT_RATIO = 0.5


class Strategy(BaseModel):
    """Weighted scoring strategy for supplier evaluation.

    The three weights blend the normalized price, quality and carbon sub-scores
    into a single composite score in [0, 1]:

        composite = price_weight  * price_score
                  + score_weight  * quality_score
                  + carbon_weight * carbon_score

    Weights always sum to 1.0.
    """

    name: str
    description: str = ""
    price_weight: float = Field(ge=0.0, le=1.0)
    score_weight: float = Field(ge=0.0, le=1.0)
    carbon_weight: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_sum(self) -> "Strategy":
        total = self.price_weight + self.score_weight + self.carbon_weight
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Strategy weights must sum to 1.0, got {total}")
        return self

    def as_weights(self) -> dict[str, float]:
        """The three weights as a serializable dict."""
        return {
            "price_weight": self.price_weight,
            "score_weight": self.score_weight,
            "carbon_weight": self.carbon_weight,
        }


DEFAULT_STRATEGIES: dict[str, Strategy] = {
    "cost_optimized": Strategy(
        name="cost_optimized",
        description="Prioritizes the lowest unit price",
        price_weight=0.65,
        score_weight=0.25,
        carbon_weight=0.10,
    ),
    "balanced": Strategy(
        name="balanced",
        description="Balances price, quality and carbon impact",
        price_weight=0.40,
        score_weight=0.40,
        carbon_weight=0.20,
    ),
    "low_carbon": Strategy(
        name="low_carbon",
        description="Prioritizes suppliers with the lowest carbon footprint",
        price_weight=0.20,
        score_weight=0.25,
        carbon_weight=0.55,
    ),
}

#: The strategy used when no strategy artifact is present (e.g. unit tests
#: driving agents directly). Reproduces the Phase 3 scoring exactly.
BALANCED_STRATEGY = DEFAULT_STRATEGIES["balanced"]


def select_strategy(constraints: dict[str, Any]) -> Strategy:
    """Pick a strategy deterministically from the requirement constraints.

    Rules (checked in order):

    1. An explicit carbon constraint (``max_carbon_per_unit`` set) →
       ``low_carbon``.
    2. A tight budget (``budget`` below ``BUDGET_TIGHT_RATIO`` of
       ``quantity * max_unit_price``) → ``cost_optimized``.
    3. Otherwise → ``balanced``.
    """
    if constraints.get("max_carbon_per_unit") is not None:
        return DEFAULT_STRATEGIES["low_carbon"]

    budget = float(constraints.get("budget") or 0.0)
    quantity = int(constraints.get("quantity") or 1)
    max_unit_price = constraints.get("max_unit_price")
    theoretical_max = quantity * float(max_unit_price) if max_unit_price else 0.0
    if theoretical_max > 0 and budget < theoretical_max * BUDGET_TIGHT_RATIO:
        return DEFAULT_STRATEGIES["cost_optimized"]

    return DEFAULT_STRATEGIES["balanced"]
