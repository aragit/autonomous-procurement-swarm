"""
Reward computation for multi-agent procurement.

Buyer reward: minimize total cost (price + risk + stockout penalty)
Seller reward: maximize margin (revenue - production_cost)
"""

from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class DealTerms:
    material: str
    quantity: int
    unit_price: float
    delivery_date: str
    payment_terms: str


class RewardComputer:
    """Computes rewards for buyer and seller given deal terms."""

    # Business constants
    STOCKOUT_COST_MULTIPLIER = 10.0
    HOLDING_COST_PER_UNIT = 0.05
    LOGISTICS_COST_BASE = 500.0
    RISK_PREMIUM_BASE = 0.02

    @staticmethod
    def compute_buyer_reward(
        deal: DealTerms,
        buyer_budget: float,
        buyer_inventory: float,
        buyer_consumption: float,
        geo_risk: float,
        spot_price: float,
    ) -> float:
        """
        Buyer reward: negative total cost (maximize = minimize cost).

        Components:
        - Purchase cost: quantity * unit_price
        - Risk premium: quantity * unit_price * geo_risk * 0.1
        - Logistics: base + distance factor (simplified)
        - Stockout penalty: if inventory < consumption * 30 days
        """
        purchase_cost = deal.quantity * deal.unit_price

        # Risk premium
        risk_premium = purchase_cost * geo_risk * RewardComputer.RISK_PREMIUM_BASE

        # Logistics (simplified)
        logistics = RewardComputer.LOGISTICS_COST_BASE

        # Stockout penalty
        days_of_supply = buyer_inventory / max(buyer_consumption, 1)
        stockout_penalty = 0.0
        if days_of_supply < 30:
            shortfall = (30 - days_of_supply) * buyer_consumption
            stockout_penalty = shortfall * spot_price * RewardComputer.STOCKOUT_COST_MULTIPLIER

        total_cost = purchase_cost + risk_premium + logistics + stockout_penalty

        # Reward is negative cost (we want to maximize)
        # Normalize by budget to keep scale reasonable
        reward = -total_cost / max(buyer_budget, 1)

        # Bonus for getting below spot price
        if deal.unit_price < spot_price * 0.95:
            reward += 0.5

        return round(reward, 4)

    @staticmethod
    def compute_seller_reward(
        deal: DealTerms,
        seller_production_cost: float,
        seller_capacity: float,
        seller_utilization: float,
    ) -> float:
        """
        Seller reward: margin + capacity utilization bonus.

        Components:
        - Revenue: quantity * unit_price
        - Production cost: quantity * production_cost
        - Margin: revenue - production_cost
        - Utilization bonus: if utilization was low, extra reward for filling capacity
        """
        revenue = deal.quantity * deal.unit_price
        production_cost = deal.quantity * seller_production_cost
        margin = revenue - production_cost

        # Utilization bonus
        utilization_bonus = 0.0
        if seller_utilization < 0.6:
            utilization_bonus = margin * 0.2  # 20% bonus for filling idle capacity

        total_reward = margin + utilization_bonus

        # Normalize by capacity
        return round(total_reward / max(seller_capacity, 1), 4)

    @staticmethod
    def compute_deal_metrics(
        deal: DealTerms,
        spot_price: float,
    ) -> Dict[str, float]:
        """Compute deal quality metrics."""
        price_vs_spot = (deal.unit_price - spot_price) / spot_price

        return {
            "buyer_savings": round(-price_vs_spot * 100, 2),  # % vs spot
            "seller_margin": round((deal.unit_price - spot_price * 0.7) / (spot_price * 0.7) * 100, 2),
            "deal_value": round(deal.quantity * deal.unit_price, 2),
            "price_efficiency": round(1.0 - abs(price_vs_spot), 4),
        }
