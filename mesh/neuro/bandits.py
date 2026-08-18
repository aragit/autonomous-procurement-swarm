"""Contextual Bandit for Adaptive Negotiation Strategy Selection.

Implements LinUCB (Linear Upper Confidence Bound) for contextual bandits
to dynamically optimize counter-offer strategies while maintaining
strict OPA/Rego bounds and MCDA compliance.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


class NegotiationStrategy(StrEnum):
    """Discrete negotiation strategy profiles."""

    AGGRESSIVE_ANCHOR = "aggressive_anchor"
    BALANCED_CONCESSION = "balanced_concession"
    PAYMENT_TERMS_TRADE_OFF = "payment_terms_trade_off"
    RISK_AWARE_PACING = "risk_aware_pacing"
    RELATIONSHIP_BUILDING = "relationship_building"


@dataclass
class StrategyConfig:
    """Configuration parameters for each negotiation strategy."""

    name: NegotiationStrategy
    # Prompt modifiers for LLM generation
    anchor_multiplier: float = 1.0
    concession_rate: float = 0.15
    payment_terms_priority: list[str] = field(default_factory=lambda: ["net_30", "net_60", "cod"])
    risk_tolerance: float = 0.5
    relationship_weight: float = 0.0

    def to_prompt_hints(self) -> dict[str, Any]:
        """Convert to prompt hints for NeuroSymbolicBridge."""
        return {
            "anchor_multiplier": self.anchor_multiplier,
            "concession_rate": self.concession_rate,
            "preferred_payment_terms": self.payment_terms_priority,
            "risk_tolerance": self.risk_tolerance,
            "relationship_weight": self.relationship_weight,
        }


DEFAULT_STRATEGIES: dict[NegotiationStrategy, StrategyConfig] = {
    NegotiationStrategy.AGGRESSIVE_ANCHOR: StrategyConfig(
        name=NegotiationStrategy.AGGRESSIVE_ANCHOR,
        anchor_multiplier=1.25,
        concession_rate=0.05,
        payment_terms_priority=["net_60", "net_30", "letter_of_credit"],
        risk_tolerance=0.3,
        relationship_weight=0.1,
    ),
    NegotiationStrategy.BALANCED_CONCESSION: StrategyConfig(
        name=NegotiationStrategy.BALANCED_CONCESSION,
        anchor_multiplier=1.0,
        concession_rate=0.15,
        payment_terms_priority=["net_30", "net_60", "cod"],
        risk_tolerance=0.5,
        relationship_weight=0.3,
    ),
    NegotiationStrategy.PAYMENT_TERMS_TRADE_OFF: StrategyConfig(
        name=NegotiationStrategy.PAYMENT_TERMS_TRADE_OFF,
        anchor_multiplier=0.95,
        concession_rate=0.10,
        payment_terms_priority=["net_60", "letter_of_credit", "net_30"],
        risk_tolerance=0.4,
        relationship_weight=0.2,
    ),
    NegotiationStrategy.RISK_AWARE_PACING: StrategyConfig(
        name=NegotiationStrategy.RISK_AWARE_PACING,
        anchor_multiplier=1.05,
        concession_rate=0.08,
        payment_terms_priority=["net_30", "cod", "net_60"],
        risk_tolerance=0.2,
        relationship_weight=0.2,
    ),
    NegotiationStrategy.RELATIONSHIP_BUILDING: StrategyConfig(
        name=NegotiationStrategy.RELATIONSHIP_BUILDING,
        anchor_multiplier=0.9,
        concession_rate=0.20,
        payment_terms_priority=["net_30", "net_60", "cod"],
        risk_tolerance=0.6,
        relationship_weight=0.5,
    ),
}


@dataclass
class BanditContext:
    """Normalized context vector for the contextual bandit."""

    # Requirement urgency (0-1): inverse of target_lead_time_days normalized
    urgency: float = 0.5
    # Budget margin (0-1): (budget - estimated_cost) / budget
    budget_margin: float = 0.5
    # Supplier baseline rating (0-1): reliability_score
    supplier_rating: float = 0.5
    # Material category complexity (0-1): carbon intensity / max_carbon
    material_complexity: float = 0.5
    # Historical win rate for this supplier (0-1)
    historical_win_rate: float = 0.5
    # Negotiation round number (normalized)
    round_number: float = 0.0

    def to_vector(self) -> NDArray[np.float32]:
        """Convert to normalized feature vector."""
        return np.array([
            self.urgency,
            self.budget_margin,
            self.supplier_rating,
            self.material_complexity,
            self.historical_win_rate,
            self.round_number,
        ], dtype=np.float32)

    @classmethod
    def from_requirement(
        cls,
        requirement: dict[str, Any],
        supplier: dict[str, Any],
        pool_data: dict[str, Any],
        round_num: int = 0,
    ) -> BanditContext:
        """Build context from procurement requirement and supplier data."""
        constraints = requirement.get("constraints", {})
        budget = float(constraints.get("budget", 1.0))
        target_lead = int(constraints.get("target_lead_time_days", 30))
        # material = str(constraints.get("material", "steel"))  # reserved for future use

        # Urgency: inverse of lead time (shorter = more urgent)
        urgency = 1.0 - min(target_lead / 60.0, 1.0)

        # Budget margin: estimate cost from spot price
        spot_price = float(pool_data.get("spot_price", 0.0))
        quantity = int(pool_data.get("quantity", 1000))
        estimated_cost = spot_price * quantity * 1.2  # rough estimate
        budget_margin = max(0.0, min((budget - estimated_cost) / max(budget, 1.0), 1.0))

        # Supplier rating
        supplier_rating = float(supplier.get("reliability_score", 0.5))

        # Material complexity: carbon intensity
        carbon = float(supplier.get("esg_carbon_per_unit", 1000.0))
        max_carbon = 20000.0  # rough max for normalization
        material_complexity = min(carbon / max_carbon, 1.0)

        # Historical win rate (from supplier memory if available)
        historical_win_rate = supplier.get("historical_win_rate", 0.5)

        # Round number (normalized to 0-1 over max 10 rounds)
        round_number = min(round_num / 10.0, 1.0)

        return cls(
            urgency=urgency,
            budget_margin=budget_margin,
            supplier_rating=supplier_rating,
            material_complexity=material_complexity,
            historical_win_rate=historical_win_rate,
            round_number=round_number,
        )

    @property
    def dimension(self) -> int:
        return 6


class LinUCBBandit:
    """LinUCB (Linear Upper Confidence Bound) Contextual Bandit.

    For each action (strategy), maintains:
    - A: d x d matrix (context covariance)
    - b: d vector (context-reward sum)
    - theta = A^{-1} b: estimated reward weights

    Action selection: argmax_a (x^T theta_a + alpha * sqrt(x^T A_a^{-1} x))
    """

    def __init__(
        self,
        strategies: list[NegotiationStrategy] | None = None,
        alpha: float = 1.0,
        lambda_reg: float = 1.0,
        context_dim: int = 6,
    ) -> None:
        self.strategies = strategies or list(NegotiationStrategy)
        self.alpha = alpha  # exploration parameter
        self.lambda_reg = lambda_reg  # regularization
        self.context_dim = context_dim
        self.n_actions = len(self.strategies)

        # Initialize A matrices and b vectors for each action
        self.A = {
            a: np.eye(context_dim, dtype=np.float32) * lambda_reg
            for a in self.strategies
        }
        self.b = {
            a: np.zeros(context_dim, dtype=np.float32)
            for a in self.strategies
        }
        self.theta = {
            a: np.zeros(context_dim, dtype=np.float32)
            for a in self.strategies
        }
        self.A_inv = {
            a: np.eye(context_dim, dtype=np.float32) / lambda_reg
            for a in self.strategies
        }

        # Statistics
        self.action_counts = dict.fromkeys(self.strategies, 0)
        self.total_rewards = dict.fromkeys(self.strategies, 0.0)

    def select_action(self, context: BanditContext) -> NegotiationStrategy:
        """Select action using LinUCB criterion."""
        x = context.to_vector()
        best_action = self.strategies[0]
        best_score = -np.inf

        for action in self.strategies:
            # UCB score: x^T theta + alpha * sqrt(x^T A^{-1} x)
            theta_a = self.theta[action]
            a_inv_a = self.A_inv[action]

            estimated_reward = float(x @ theta_a)
            uncertainty = self.alpha * np.sqrt(max(0.0, float(x @ a_inv_a @ x)))
            score = estimated_reward + uncertainty

            # Tie-breaking: add tiny noise to break symmetry on first selections
            if self.action_counts[action] == 0:
                score += np.random.random() * 1e-10

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def update(self, action: NegotiationStrategy, context: BanditContext, reward: float) -> None:
        """Update bandit parameters with observed reward."""
        x = context.to_vector()

        # Update A and b
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x

        # Recompute inverse and theta
        self.A_inv[action] = np.linalg.inv(self.A[action]).astype(np.float32)
        self.theta[action] = (self.A_inv[action] @ self.b[action]).astype(np.float32)

        # Update statistics
        self.action_counts[action] += 1
        self.total_rewards[action] += reward

    def get_action_stats(self) -> dict[str, dict[str, float]]:
        """Get statistics for all actions."""
        return {
            action.value: {
                "count": self.action_counts[action],
                "total_reward": self.total_rewards[action],
                "avg_reward": (
                    self.total_rewards[action] / self.action_counts[action]
                    if self.action_counts[action] > 0
                    else 0.0
                ),
                "theta_norm": float(np.linalg.norm(self.theta[action])),
            }
            for action in self.strategies
        }

    def save_state(self, path: str | Path) -> None:
        """Serialize bandit state to JSON."""
        state = {
            "alpha": self.alpha,
            "lambda_reg": self.lambda_reg,
            "context_dim": self.context_dim,
            "strategies": [a.value for a in self.strategies],
            "A": {a.value: self.A[a].tolist() for a in self.strategies},
            "b": {a.value: self.b[a].tolist() for a in self.strategies},
            "action_counts": {a.value: self.action_counts[a] for a in self.strategies},
            "total_rewards": {a.value: self.total_rewards[a] for a in self.strategies},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load_state(cls, path: str | Path) -> LinUCBBandit:
        """Deserialize bandit state from JSON."""
        with open(path) as f:
            state = json.load(f)

        strategies = [NegotiationStrategy(s) for s in state["strategies"]]
        bandit = cls(
            strategies=strategies,
            alpha=state["alpha"],
            lambda_reg=state["lambda_reg"],
            context_dim=state["context_dim"],
        )

        for a in strategies:
            bandit.A[a] = np.array(state["A"][a.value], dtype=np.float32)
            bandit.b[a] = np.array(state["b"][a.value], dtype=np.float32)
            bandit.A_inv[a] = np.linalg.inv(bandit.A[a]).astype(np.float32)
            bandit.theta[a] = (bandit.A_inv[a] @ bandit.b[a]).astype(np.float32)
            bandit.action_counts[a] = state["action_counts"][a.value]
            bandit.total_rewards[a] = state["total_rewards"][a.value]

        return bandit


def compute_reward(
    decision: dict[str, Any],
    quote: dict[str, Any],
    requirement: dict[str, Any],
    negotiation_rounds: int = 1,
) -> float:
    """Compute scalar reward for a completed negotiation.

    Factors:
    - Cost reduction vs budget (0-1)
    - Payment terms favorability (0-1)
    - Convergence speed (0-1, fewer rounds better)
    - MCDA composite score alignment (0-1)
    """
    budget = float(requirement.get("constraints", {}).get("budget", 1.0))
    price = float(quote.get("price", 0.0))
    quantity = int(quote.get("metadata", {}).get("quantity", 1))
    total_cost = price * quantity

    # Cost reduction reward (0-1)
    cost_reward = max(0.0, min(1.0 - total_cost / max(budget, 1.0), 1.0))

    # Payment terms reward
    terms = quote.get("terms", "net_30")
    terms_score = {"cod": 1.0, "net_30": 0.7, "net_60": 0.4, "letter_of_credit": 0.3}.get(
        terms, 0.5
    )

    # Convergence speed reward (fewer rounds = better)
    speed_reward = max(0.0, 1.0 - (negotiation_rounds - 1) * 0.15)

    # MCDA alignment (from decision if available)
    mcdas_score = 0.5
    if decision and "composite_score" in decision:
        mcdas_score = float(decision["composite_score"])

    # Weighted combination
    reward = (
        0.4 * cost_reward
        + 0.2 * terms_score
        + 0.2 * speed_reward
        + 0.2 * mcdas_score
    )

    return float(np.clip(reward, 0.0, 1.0))


def get_default_bandit() -> LinUCBBandit:
    """Factory for a fresh bandit with default parameters."""
    return LinUCBBandit()


_BANDIT_STATE_PATH = os.environ.get(
    "BANDIT_STATE_PATH", "/app/data/bandit_state.json"
)


def get_persistent_bandit() -> LinUCBBandit:
    """Get or create persistent bandit instance."""
    global _persistent_bandit
    if _persistent_bandit is not None:
        return _persistent_bandit

    if os.path.exists(_BANDIT_STATE_PATH):
        try:
            _persistent_bandit = LinUCBBandit.load_state(_BANDIT_STATE_PATH)
        except Exception:
            _persistent_bandit = get_default_bandit()
    else:
        _persistent_bandit = get_default_bandit()

    return _persistent_bandit


_persistent_bandit: LinUCBBandit | None = None


async def save_bandit_state() -> None:
    """Persist bandit state to disk."""
    global _persistent_bandit
    if _persistent_bandit is not None:
        try:
            _persistent_bandit.save_state(_BANDIT_STATE_PATH)
        except Exception as e:
            import structlog
            structlog.get_logger(__name__).warning(
                "bandit_save_failed", error=str(e)
            )
