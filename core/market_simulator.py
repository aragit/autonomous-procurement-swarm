"""
Stochastic market simulation for procurement environment.

Models:
- Price dynamics: Geometric Brownian Motion with regime-switching
- Geopolitical risk: Markov chain (low/medium/high/crisis)
- Supply shocks: Poisson arrivals with magnitude
- Inventory dynamics: consumption and production
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import random


class RiskRegime(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRISIS = 3


@dataclass
class MarketState:
    date: str
    material: str
    spot_price: float
    volatility: float
    geo_risk: float
    risk_regime: RiskRegime
    supply_health: float
    price_trend: float  # 7-day % change
    events: List[str] = None

    def __post_init__(self):
        if self.events is None:
            self.events = []


class GeopoliticalRiskModel:
    """Markov chain for geopolitical risk regimes."""

    TRANSITION_MATRIX = np.array([
        [0.85, 0.12, 0.03, 0.00],  # LOW
        [0.20, 0.60, 0.15, 0.05],  # MEDIUM
        [0.10, 0.30, 0.40, 0.20],  # HIGH
        [0.05, 0.15, 0.30, 0.50],  # CRISIS
    ])

    REGIME_MULTIPLIERS = {
        RiskRegime.LOW: 1.0,
        RiskRegime.MEDIUM: 1.15,
        RiskRegime.HIGH: 1.40,
        RiskRegime.CRISIS: 2.00,
    }

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.current_regime = RiskRegime.LOW

    def step(self) -> RiskRegime:
        """Advance one time step."""
        probs = self.TRANSITION_MATRIX[self.current_regime.value]
        self.current_regime = RiskRegime(
            self.rng.choice(len(RiskRegime), p=probs)
        )
        return self.current_regime

    def get_risk_score(self) -> float:
        """Return continuous risk score 0-1."""
        return self.current_regime.value / 3.0  # 0, 0.33, 0.67, 1.0


class SupplyShockGenerator:
    """Poisson supply shocks with log-normal magnitude."""

    def __init__(self, lambda_rate: float = 0.05, seed: int = 42):
        self.lambda_rate = lambda_rate  # Expected shocks per period
        self.rng = np.random.RandomState(seed)

    def generate(self) -> Tuple[bool, float, str]:
        """Return (shock_occurred, magnitude, description)."""
        if self.rng.poisson(self.lambda_rate) > 0:
            magnitude = self.rng.lognormal(0, 0.5)  # Multiplier
            descriptions = [
                "Port closure due to labor strike",
                "Sanctions on major supplier country",
                "Natural disaster affecting mines",
                "Trade tariff increase announced",
                "Shipping lane blockage",
                "Factory fire at key producer",
            ]
            return True, magnitude, self.rng.choice(descriptions)
        return False, 1.0, ""


class PriceProcess:
    """Geometric Brownian Motion with regime-dependent drift/vol."""

    BASE_PRICES = {
        "steel": 450.0,
        "aluminum": 2200.0,
        "copper": 8500.0,
        "plastic": 1200.0,
        "lumber": 600.0,
        "rubber": 1800.0,
    }

    def __init__(self, material: str, seed: int = 42):
        self.material = material
        self.base_price = self.BASE_PRICES.get(material, 1000.0)
        self.price = self.base_price
        self.rng = np.random.RandomState(seed)
        self.history = [self.price]

    def step(self, regime: RiskRegime, shock: Tuple[bool, float, str]) -> float:
        """Advance one time step."""
        dt = 1 / 252  # Daily step

        # Regime-dependent parameters
        if regime == RiskRegime.LOW:
            mu, sigma = 0.05, 0.15
        elif regime == RiskRegime.MEDIUM:
            mu, sigma = 0.02, 0.25
        elif regime == RiskRegime.HIGH:
            mu, sigma = -0.05, 0.40
        else:  # CRISIS
            mu, sigma = -0.15, 0.60

        # GBM step
        dW = self.rng.normal(0, np.sqrt(dt))
        dP = mu * self.price * dt + sigma * self.price * dW
        self.price = max(self.price + dP, self.base_price * 0.1)

        # Apply shock
        if shock[0]:
            self.price *= shock[1]

        self.history.append(self.price)
        return self.price

    def get_trend(self, window: int = 7) -> float:
        """Return price change % over window."""
        if len(self.history) < window + 1:
            return 0.0
        return (self.history[-1] - self.history[-window-1]) / self.history[-window-1] * 100


class MarketSimulator:
    """Orchestrates all market dynamics."""

    MATERIALS = ["steel", "aluminum", "copper", "plastic", "lumber", "rubber"]

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.risk_model = GeopoliticalRiskModel(seed)
        self.shock_generator = SupplyShockGenerator(seed=seed)
        self.price_processes = {
            m: PriceProcess(m, seed + i)
            for i, m in enumerate(self.MATERIALS)
        }
        self.day = 0
        self.date = "2026-01-01"
        self.event_log = []

    def step(self) -> Dict[str, MarketState]:
        """Advance simulation one day. Return market states for all materials."""
        self.day += 1
        # Simple date increment (ignore month boundaries for simulation)
        from datetime import datetime, timedelta
        current = datetime.strptime(self.date, "%Y-%m-%d")
        current += timedelta(days=1)
        self.date = current.strftime("%Y-%m-%d")

        regime = self.risk_model.step()
        shock = self.shock_generator.generate()

        if shock[0]:
            self.event_log.append(f"{self.date}: {shock[2]} (magnitude: {shock[1]:.2f}x)")

        states = {}
        for material, process in self.price_processes.items():
            price = process.step(regime, shock)
            states[material] = MarketState(
                date=self.date,
                material=material,
                spot_price=round(price, 2),
                volatility=round(0.15 + regime.value * 0.15, 2),
                geo_risk=round(self.risk_model.get_risk_score(), 2),
                risk_regime=regime,
                supply_health=round(max(1.0 - regime.value * 0.25, 0.1), 2),
                price_trend=round(process.get_trend(), 2),
                events=self.event_log[-3:],  # Last 3 events
            )

        return states

    def get_state(self, material: str) -> MarketState:
        """Get current state for specific material."""
        # Step once if needed
        states = self.step()
        return states[material]
