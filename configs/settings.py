"""Application settings loaded from YAML and environment variables."""

from pathlib import Path
from typing import List, Dict
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml

_BASE_YAML_PATH = Path(__file__).parent / "base.yaml"


class LLMSettings(BaseSettings):
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    prefer_vllm: bool = False
    temperature: float = 0.7
    max_tokens: int = 512


class MarketSettings(BaseSettings):
    seed: int = 42
    materials: List[str] = Field(default_factory=lambda: [
        "steel", "aluminum", "copper", "plastic", "lumber", "rubber"
    ])
    shock_lambda: float = 0.05


class NegotiationSettings(BaseSettings):
    max_turns: int = 6
    valid_materials: List[str] = Field(default_factory=lambda: [
        "steel", "aluminum", "copper", "plastic", "lumber", "rubber"
    ])
    valid_payment_terms: List[str] = Field(default_factory=lambda: [
        "net_30", "net_60", "cod", "letter_of_credit"
    ])


class AgentSettings(BaseSettings):
    buyer_budget: float = 500_000.0
    buyer_inventory: float = 5_000.0
    buyer_consumption: float = 200.0
    buyer_risk_tolerance: float = 0.6
    buyer_credit_rating: float = 0.85
    seller_capacity: float = 10_000.0
    seller_utilization: float = 0.45
    seller_production_cost: float = 350.0
    seller_min_margin: float = 0.15


class EvaluationSettings(BaseSettings):
    pareto_plot_path: str = "data/pareto.png"
    ledger_dir: str = "data"
    esg_baselines: Dict[str, float] = Field(default_factory=lambda: {
        "steel": 1800.0,
        "aluminum": 12000.0,
        "copper": 3000.0,
        "plastic": 2500.0,
        "lumber": 200.0,
        "rubber": 2800.0,
    })
    scoring_weights: Dict[str, float] = Field(default_factory=lambda: {
        "price": 0.40,
        "lead_time": 0.25,
        "esg": 0.20,
        "reliability": 0.15,
    })
    shortlist_size: int = 2


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PROCUREMENT_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    llm: LLMSettings = Field(default_factory=LLMSettings)
    market: MarketSettings = Field(default_factory=MarketSettings)
    negotiation: NegotiationSettings = Field(default_factory=NegotiationSettings)
    agents: AgentSettings = Field(default_factory=AgentSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> "Settings":
        yaml_path = path or _BASE_YAML_PATH
        if yaml_path.exists():
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            return cls(**data)
        return cls()


# Singleton instance
settings = Settings.from_yaml()
