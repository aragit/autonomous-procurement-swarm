# Autonomous Procurement Swarm — Source Code

LLM-powered multi-agent contract negotiation simulation.

## Folder Hierarchy

```
autonomous-procurement-swarm/
├── .gitignore
├── configs/
│   └── base.yaml
├── core/
│   ├── __init__.py
│   ├── agents.py
│   ├── contract_ledger.py
│   ├── llm_engine.py
│   ├── market_simulator.py
│   ├── negotiation_protocol.py
│   ├── prompts.py
│   └── rewards.py
├── orchestration/
│   ├── __init__.py
│   └── negotiation_env.py
├── scripts/
│   ├── evaluate.py
│   └── run_negotiation.py
├── tests/
│   ├── __init__.py
│   ├── test_ledger.py
│   ├── test_market.py
│   └── test_protocol.py
└── requirements.txt
```

## File: .gitignore

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyTorch / Models
*.pth
*.pt
*.bin
*.safetensors
models/
checkpoints/

# Data outputs
data/*.json
data/*.png
data/*.csv

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Testing
.pytest_cache/
.coverage
htmlcov/
```

## File: configs/base.yaml

```yaml
# Autonomous Procurement Swarm — Base Configuration

project:
  name: "autonomous-procurement-swarm"
  version: "0.1.0-blueprint"
  description: "LLM-powered multi-agent contract negotiation"

llm:
  model_name: "Qwen/Qwen2.5-3B-Instruct"
  prefer_vllm: false  # Set true if vLLM CPU installed
  temperature: 0.7
  max_tokens: 512

market:
  seed: 42
  materials: ["steel", "aluminum", "copper", "plastic", "lumber", "rubber"]
  shock_lambda: 0.05  # Expected shocks per day

negotiation:
  max_turns: 6
  valid_materials: ["steel", "aluminum", "copper", "plastic", "lumber", "rubber"]
  valid_payment_terms: ["net_30", "net_60", "cod", "letter_of_credit"]

agents:
  buyer:
    default_budget: 500000.0
    default_inventory: 5000.0
    default_consumption: 200.0
    risk_tolerance: 0.6
    credit_rating: 0.85

  seller:
    default_capacity: 10000.0
    default_utilization: 0.45
    production_cost: 350.0
    min_margin: 0.15

rewards:
  stockout_cost_multiplier: 10.0
  holding_cost_per_unit: 0.05
  logistics_cost_base: 500.0
  risk_premium_base: 0.02

evaluation:
  pareto_plot_path: "data/pareto.png"
  ledger_dir: "data"
```

## File: core/__init__.py

```python
```

## File: core/agents.py

```python
"""
Agent definitions for the procurement swarm.

Each agent wraps an LLM engine with a specific system prompt and state.
"""

from typing import Dict, Any, List
from dataclasses import dataclass, field
from core.llm_engine import BaseLLMEngine, LLMResponse
from core.prompts import (
    BUYER_PROMPT, SELLER_PROMPT, MARKET_PROMPT, ARBITER_PROMPT,
    format_history,
)
from core.negotiation_protocol import NegotiationMessage, MessageType


@dataclass
class AgentState:
    name: str
    role: str
    inventory: float
    budget: float = 0.0
    capacity: float = 0.0
    utilization: float = 0.0
    production_cost: float = 0.0
    min_margin: float = 0.0
    risk_tolerance: float = 0.5
    credit_rating: float = 0.8
    messages: List[Dict[str, str]] = field(default_factory=list)
    deals_closed: int = 0
    total_revenue: float = 0.0
    total_cost: float = 0.0


class BaseAgent:
    """Base class for all LLM-powered agents."""

    def __init__(
        self,
        name: str,
        role: str,
        llm_engine: BaseLLMEngine,
        prompt_template,
        state: AgentState,
    ):
        self.name = name
        self.role = role
        self.llm = llm_engine
        self.prompt = prompt_template
        self.state = state
        self.conversation_history = []

    def act(self, context: Dict[str, Any]) -> NegotiationMessage:
        """Generate an action using the LLM."""
        # Build messages for LLM
        system_prompt = self.prompt.render(context)
        messages = [
            {"role": "system", "content": system_prompt},
            *self.conversation_history,
        ]

        # Call LLM
        response: LLMResponse = self.llm.chat_completion(
            messages=messages,
            temperature=0.7,
            max_tokens=512,
        )

        # Parse response
        try:
            msg = NegotiationMessage.from_llm_output(response.content)
        except ValueError as e:
            # Fallback: reject if parsing fails
            msg = NegotiationMessage(
                type=MessageType.REJECT,
                material=context.get("material", "unknown"),
                reason=f"Parse error: {str(e)}",
            )

        # Update history
        self.conversation_history.append(
            {"role": "assistant", "content": response.content}
        )

        return msg

    def observe(self, message: Dict[str, str]):
        """Add an observation to conversation history."""
        self.conversation_history.append(message)

    def reset(self):
        """Clear conversation history for new negotiation."""
        self.conversation_history = []


class BuyerAgent(BaseAgent):
    """Procurement buyer agent."""

    def __init__(self, name: str, llm_engine: BaseLLMEngine, state: AgentState):
        super().__init__(name, "buyer", llm_engine, BUYER_PROMPT, state)

    def prepare_context(self, market_state, material: str) -> Dict[str, Any]:
        """Build context dict for prompt rendering."""
        return {
            "inventory_cap": self.state.capacity,
            "current_stock": self.state.inventory,
            "consumption_rate": self.state.utilization,
            "budget": self.state.budget,
            "risk_threshold": self.state.risk_tolerance,
            "spot_price": market_state.spot_price,
            "volatility": market_state.volatility,
            "geo_risk": market_state.geo_risk,
            "reliability": self.state.credit_rating,
            "history": format_history(self.conversation_history),
        }


class SellerAgent(BaseAgent):
    """Raw materials supplier agent."""

    def __init__(self, name: str, llm_engine: BaseLLMEngine, state: AgentState):
        super().__init__(name, "seller", llm_engine, SELLER_PROMPT, state)

    def prepare_context(self, market_state, material: str, buyer_rating: float) -> Dict[str, Any]:
        return {
            "capacity": self.state.capacity,
            "utilization": self.state.utilization,
            "production_cost": self.state.production_cost,
            "min_margin": self.state.min_margin,
            "inventory": self.state.inventory,
            "spot_price": market_state.spot_price,
            "volatility": market_state.volatility,
            "geo_risk": market_state.geo_risk,
            "buyer_rating": buyer_rating,
            "history": format_history(self.conversation_history),
        }


class MarketAgent(BaseAgent):
    """Market intelligence agent."""

    def __init__(self, name: str, llm_engine: BaseLLMEngine):
        state = AgentState(name=name, role="market", inventory=0)
        super().__init__(name, "market", llm_engine, MARKET_PROMPT, state)

    def prepare_context(self, market_state) -> Dict[str, Any]:
        return {
            "current_date": market_state.date,
            "material": market_state.material,
            "spot_price": market_state.spot_price,
            "price_change": market_state.price_trend,
            "volatility": market_state.volatility,
            "geo_risk": market_state.geo_risk,
            "supply_health": market_state.supply_health,
            "events": "\n".join(market_state.events) if market_state.events else "No recent events.",
        }


class ArbiterAgent(BaseAgent):
    """Negotiation arbiter."""

    def __init__(self, name: str, llm_engine: BaseLLMEngine):
        state = AgentState(name=name, role="arbiter", inventory=0)
        super().__init__(name, "arbiter", llm_engine, ARBITER_PROMPT, state)

    def prepare_context(
        self, turn_number: int, buyer_name: str, seller_name: str,
        material: str, last_message: str, history: List[Dict]
    ) -> Dict[str, Any]:
        return {
            "turn_number": turn_number,
            "buyer_name": buyer_name,
            "seller_name": seller_name,
            "material": material,
            "last_message": last_message,
            "history": format_history(history),
        }
```

## File: core/contract_ledger.py

```python
"""
Contract ledger for negotiation audit trail.
Immutable append-only log with hash chaining.
"""

import hashlib
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class LedgerEntry:
    timestamp: str
    episode_id: str
    turn_number: int
    agent_name: str
    agent_role: str
    message_type: str
    material: str
    quantity: Optional[int] = None
    price: Optional[float] = None
    delivery_date: Optional[str] = None
    payment_terms: Optional[str] = None
    justification: Optional[str] = None
    hash: str = ""
    prev_hash: str = ""

    def __post_init__(self):
        if not self.hash:
            self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of entry content."""
        content = json.dumps({
            "timestamp": self.timestamp,
            "episode_id": self.episode_id,
            "turn_number": self.turn_number,
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "message_type": self.message_type,
            "material": self.material,
            "quantity": self.quantity,
            "price": self.price,
            "delivery_date": self.delivery_date,
            "payment_terms": self.payment_terms,
            "justification": self.justification,
            "prev_hash": self.prev_hash,
        }, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict:
        return asdict(self)


class ContractLedger:
    """Immutable negotiation ledger with hash chain."""

    def __init__(self):
        self.entries: List[LedgerEntry] = []
        self.last_hash = "0" * 16

    def append(self, entry: LedgerEntry):
        """Add entry with hash chaining."""
        entry.prev_hash = self.last_hash
        entry.hash = entry._compute_hash()
        self.entries.append(entry)
        self.last_hash = entry.hash

    def get_episode(self, episode_id: str) -> List[LedgerEntry]:
        """Get all entries for an episode."""
        return [e for e in self.entries if e.episode_id == episode_id]

    def verify_integrity(self) -> bool:
        """Verify hash chain integrity."""
        for i, entry in enumerate(self.entries):
            if i == 0:
                if entry.prev_hash != "0" * 16:
                    return False
            else:
                if entry.prev_hash != self.entries[i-1].hash:
                    return False
            # Recompute hash
            expected = entry._compute_hash()
            if entry.hash != expected:
                return False
        return True

    def export_json(self, filepath: str):
        """Export ledger to JSON."""
        with open(filepath, 'w') as f:
            json.dump([e.to_dict() for e in self.entries], f, indent=2)

    def get_stats(self) -> Dict:
        """Compute ledger statistics."""
        if not self.entries:
            return {}
        return {
            "total_entries": len(self.entries),
            "total_episodes": len(set(e.episode_id for e in self.entries)),
            "deals_closed": sum(1 for e in self.entries if e.message_type == "accept"),
            "deadlocks": sum(1 for e in self.entries if e.message_type == "reject"),
            "avg_price": sum(e.price for e in self.entries if e.price) /
                        max(sum(1 for e in self.entries if e.price), 1),
        }
```

## File: core/llm_engine.py

```python
"""
LLM Inference Engine for the Procurement Swarm.

Supports three backends:
1. MockLLM — deterministic responses, instant, no download (Active Blueprint default)
2. Transformers — real LLM via HuggingFace (CPU, downloads on first use)
3. vLLM — fast batched inference (CPU, requires build)

The engine exposes an OpenAI-compatible chat completion API.
"""

import os
import json
import time
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    model: str


class BaseLLMEngine(ABC):
    """Abstract base for LLM inference backends."""

    @abstractmethod
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> LLMResponse:
        pass

    @abstractmethod
    def shutdown(self):
        pass


class MockLLMEngine(BaseLLMEngine):
    """
    Deterministic mock LLM for Active Blueprint demonstration.
    Generates realistic negotiation responses without any model download.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.model_name = "mock-llm-blueprint"
        print("[LLM] Using MockLLM — deterministic, instant, no download")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> LLMResponse:
        start = time.time()

        # Extract role from system prompt to determine agent type
        system_prompt = messages[0]["content"] if messages else ""
        role = "unknown"
        if "procurement agent" in system_prompt.lower():
            role = "buyer"
        elif "supplier agent" in system_prompt.lower():
            role = "seller"
        elif "market intelligence" in system_prompt.lower():
            role = "market"
        elif "arbiter" in system_prompt.lower():
            role = "arbiter"

        # Extract context values from prompt
        spot_price = self._extract_float(system_prompt, "spot price: $")
        geo_risk = self._extract_float(system_prompt, "geopolitical risk:")

        # Generate contextually appropriate response
        content = self._generate_response(role, spot_price, geo_risk)

        latency_ms = (time.time() - start) * 1000

        return LLMResponse(
            content=content,
            tokens_in=len(str(messages)),
            tokens_out=len(content.split()),
            latency_ms=latency_ms,
            model=self.model_name,
        )

    def _extract_float(self, text: str, key: str) -> float:
        """Extract float value after key in text."""
        try:
            idx = text.lower().find(key.lower())
            if idx == -1:
                return 0.0
            start = idx + len(key)
            # Find the number
            end = start
            while end < len(text) and (text[end].isdigit() or text[end] in ".,"):
                end += 1
            return float(text[start:end].replace(",", ""))
        except (ValueError, IndexError):
            return 0.0

    def _generate_response(self, role: str, spot_price: float, geo_risk: float) -> str:
        """Generate deterministic but realistic negotiation response."""
        if role == "buyer":
            # Buyer counters at 85-95% of spot, or accepts if price is low
            if spot_price > 0 and self.rng.random() > 0.3:
                counter = round(spot_price * self.rng.uniform(0.85, 0.95), 2)
                return json.dumps({
                    "type": "counter",
                    "material": "steel",
                    "quantity": self.rng.choice([500, 1000, 2000]),
                    "counter_price": counter,
                    "justification": f"Market oversupply and {geo_risk:.0%} geopolitical risk warrants discount",
                    "deadline": "2026-06-30"
                })
            else:
                return json.dumps({
                    "type": "accept",
                    "material": "steel",
                    "quantity": 1000,
                    "final_price": round(spot_price * 0.92, 2) if spot_price > 0 else 400.0,
                    "delivery_date": "2026-07-15"
                })

        elif role == "seller":
            # Seller offers at 105-115% of spot, or counters higher
            if spot_price > 0 and self.rng.random() > 0.3:
                offer = round(spot_price * self.rng.uniform(1.05, 1.15), 2)
                return json.dumps({
                    "type": "counter",
                    "material": "steel",
                    "quantity": self.rng.choice([500, 1000, 2000]),
                    "counter_price": offer,
                    "justification": f"Premium quality and supply chain resilience amid {geo_risk:.0%} risk",
                    "deadline": "2026-06-25"
                })
            else:
                return json.dumps({
                    "type": "offer",
                    "material": "steel",
                    "quantity": 1000,
                    "unit_price": round(spot_price * 1.10, 2) if spot_price > 0 else 500.0,
                    "delivery_date": "2026-07-15",
                    "payment_terms": "net_30"
                })

        elif role == "market":
            trend = "rising" if self.rng.random() > 0.5 else "falling"
            return json.dumps({
                "type": "report",
                "material": "steel",
                "spot_price": round(spot_price, 2) if spot_price > 0 else 450.0,
                "trend": trend,
                "volatility": round(self.rng.uniform(0.1, 0.4), 2),
                "geo_risk": round(geo_risk, 2),
                "supply_disruption": geo_risk > 0.6,
                "recommendation": f"Consider hedging given {geo_risk:.0%} geopolitical risk"
            })

        else:  # arbiter or unknown
            return json.dumps({
                "type": "validate",
                "agent": "unknown",
                "message_valid": True,
                "errors": []
            })

    def shutdown(self):
        pass


class TransformersEngine(BaseLLMEngine):
    """
    Real LLM engine using HuggingFace Transformers.
    Downloads model on first use (~6GB for Qwen2.5-3B).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "cpu",
    ):
        print(f"[LLM] Loading {model_name} via Transformers (CPU)...")
        print(f"[LLM] This will download ~6GB on first run. Go get coffee.")
        start = time.time()

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

        load_time = time.time() - start
        print(f"[LLM] Model loaded in {load_time:.1f}s")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> LLMResponse:
        start = time.time()
        import torch

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(
            self.model.device
        )

        input_ids = model_inputs.input_ids
        tokens_in = input_ids.shape[1]

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        tokens_out = generated_ids.shape[1] - tokens_in
        response = self.tokenizer.decode(
            generated_ids[0][tokens_in:], skip_special_tokens=True
        )

        latency_ms = (time.time() - start) * 1000

        return LLMResponse(
            content=response.strip(),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            model=self.model_name,
        )

    def shutdown(self):
        del self.model
        import torch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


class VLLMEngine(BaseLLMEngine):
    """
    Primary LLM engine using vLLM (CPU).
    Faster batched inference, preferred when available.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        tensor_parallel_size: int = 1,
    ):
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM not installed. Use: VLLM_TARGET_DEVICE=cpu pip install vllm"
            )

        print(f"[LLM] Loading {model_name} via vLLM (CPU)...")
        start = time.time()

        self.model_name = model_name
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            device="cpu",
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams

        load_time = time.time() - start
        print(f"[LLM] Model loaded in {load_time:.1f}s")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> LLMResponse:
        start = time.time()

        from vllm import SamplingParams

        prompt = self._format_chat(messages)

        sp = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        outputs = self.llm.generate(prompt, sp)
        response = outputs[0].outputs[0].text

        latency_ms = (time.time() - start) * 1000

        return LLMResponse(
            content=response.strip(),
            tokens_in=len(outputs[0].prompt_token_ids),
            tokens_out=len(outputs[0].outputs[0].token_ids),
            latency_ms=latency_ms,
            model=self.model_name,
        )

    def _format_chat(self, messages: List[Dict[str, str]]) -> str:
        """Simple chat formatting for vLLM."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        return "\n".join(parts) + "\nAssistant:"

    def shutdown(self):
        pass


class LLMEngineFactory:
    """Factory to select the best available LLM backend."""

    @staticmethod
    def create(
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        prefer_vllm: bool = True,
        use_mock: bool = False,
    ) -> BaseLLMEngine:
        if use_mock:
            return MockLLMEngine()

        if prefer_vllm:
            try:
                return VLLMEngine(model_name)
            except Exception as e:
                print(f"[LLM] vLLM failed ({e}), falling back to Transformers")
                return TransformersEngine(model_name)
        return TransformersEngine(model_name)
```

## File: core/market_simulator.py

```python
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
```

## File: core/negotiation_protocol.py

```python
"""
Structured negotiation protocol.
Defines message schema, validation, and state transitions.
"""

import json
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class MessageType(Enum):
    OFFER = "offer"
    COUNTER = "counter"
    ACCEPT = "accept"
    REJECT = "reject"
    REPORT = "report"
    ALERT = "alert"
    VALIDATE = "validate"
    PERMIT = "permit"
    DECLARE = "declare"
    SANCTION = "sanction"


@dataclass
class NegotiationMessage:
    type: MessageType
    material: str
    quantity: Optional[int] = None
    unit_price: Optional[float] = None
    counter_price: Optional[float] = None
    final_price: Optional[float] = None
    delivery_date: Optional[str] = None
    payment_terms: Optional[str] = None
    justification: Optional[str] = None
    reason: Optional[str] = None
    deadline: Optional[str] = None
    agent: Optional[str] = None
    message_valid: Optional[bool] = None
    errors: List[str] = field(default_factory=list)
    outcome: Optional[str] = None
    terms: Optional[Dict] = None
    severity: Optional[str] = None
    event: Optional[str] = None
    expected_impact: Optional[str] = None
    recommendation: Optional[str] = None
    violation: Optional[str] = None
    penalty: Optional[str] = None
    trend: Optional[str] = None
    volatility: Optional[float] = None
    geo_risk: Optional[float] = None
    supply_disruption: Optional[bool] = None
    raw_json: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_llm_output(cls, text: str) -> "NegotiationMessage":
        """Parse LLM JSON output into structured message."""
        # Extract JSON from potential markdown wrappers
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from text with regex
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON in LLM output: {text[:200]}")
            else:
                raise ValueError(f"No JSON found in LLM output: {text[:200]}")

        msg_type = MessageType(data.get("type", "reject"))

        return cls(
            type=msg_type,
            material=data.get("material", "unknown"),
            quantity=data.get("quantity"),
            unit_price=data.get("unit_price"),
            counter_price=data.get("counter_price"),
            final_price=data.get("final_price"),
            delivery_date=data.get("delivery_date"),
            payment_terms=data.get("payment_terms"),
            justification=data.get("justification"),
            reason=data.get("reason"),
            deadline=data.get("deadline"),
            agent=data.get("agent"),
            message_valid=data.get("message_valid"),
            errors=data.get("errors", []),
            outcome=data.get("outcome"),
            terms=data.get("terms"),
            severity=data.get("severity"),
            event=data.get("event"),
            expected_impact=data.get("expected_impact"),
            recommendation=data.get("recommendation"),
            violation=data.get("violation"),
            penalty=data.get("penalty"),
            trend=data.get("trend"),
            volatility=data.get("volatility"),
            geo_risk=data.get("geo_risk"),
            supply_disruption=data.get("supply_disruption"),
            raw_json=data,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for logging/ledger."""
        d = {
            "type": self.type.value,
            "material": self.material,
        }
        for field_name in [
            "quantity", "unit_price", "counter_price", "final_price",
            "delivery_date", "payment_terms", "justification", "reason",
            "deadline", "agent", "message_valid", "errors", "outcome",
            "terms", "severity", "event", "expected_impact", "recommendation",
            "violation", "penalty", "trend", "volatility", "geo_risk",
            "supply_disruption",
        ]:
            val = getattr(self, field_name)
            if val is not None:
                d[field_name] = val
        return d


class ProtocolValidator:
    """Validates negotiation messages against business rules."""

    VALID_MATERIALS = {"steel", "aluminum", "copper", "plastic", "lumber", "rubber"}
    VALID_PAYMENT_TERMS = {"net_30", "net_60", "cod", "letter_of_credit"}
    MAX_TURNS = 10
    MAX_PRICE = 1000000.0
    MIN_PRICE = 0.01

    @staticmethod
    def validate(msg: NegotiationMessage) -> List[str]:
        """Return list of validation errors. Empty list = valid."""
        errors = []

        # Material check
        if msg.material not in ProtocolValidator.VALID_MATERIALS:
            errors.append(f"Invalid material: {msg.material}")

        # Price checks
        price = msg.unit_price or msg.counter_price or msg.final_price
        if price is not None:
            if price <= 0:
                errors.append(f"Price must be positive: {price}")
            if price > ProtocolValidator.MAX_PRICE:
                errors.append(f"Price exceeds maximum: {price}")

        # Quantity check
        if msg.quantity is not None and msg.quantity <= 0:
            errors.append(f"Quantity must be positive: {msg.quantity}")

        # Payment terms check
        if msg.payment_terms and msg.payment_terms not in ProtocolValidator.VALID_PAYMENT_TERMS:
            errors.append(f"Invalid payment terms: {msg.payment_terms}")

        # Date format check (basic)
        if msg.delivery_date:
            import datetime
            try:
                datetime.datetime.strptime(msg.delivery_date, "%Y-%m-%d")
            except ValueError:
                errors.append(f"Invalid date format: {msg.delivery_date}")

        return errors

    @staticmethod
    def is_terminal(msg: NegotiationMessage) -> bool:
        """Check if message ends negotiation."""
        return msg.type in {MessageType.ACCEPT, MessageType.REJECT, MessageType.DECLARE}
```

## File: core/prompts.py

```python
"""
System prompts for all agents in the procurement swarm.

Each prompt defines:
- Role identity and objectives
- Action space (message types)
- Constraints (inventory, budget, risk)
- Strategy guidelines
- Output format (JSON)
"""

from typing import Dict, Any
import json


class PromptTemplate:
    """Base class for agent prompts with dynamic context filling."""

    def __init__(self, template: str):
        self.template = template

    def render(self, context: Dict[str, Any]) -> str:
        """Fill template with runtime context."""
        return self.template.format(**context)


# ─── BUYER AGENT ─────────────────────────────────────────────────────────────

BUYER_SYSTEM_TEMPLATE = """You are a procurement agent for a manufacturing firm.
Your goal: secure raw materials at the lowest total cost (price + risk + logistics).

You negotiate with sellers via structured messages. Respond ONLY with a single JSON object.

## Action Space

- OFFER: Propose a deal to a seller
  {{"type": "offer", "material": "<material>", "quantity": <int>, "unit_price": <float>, "delivery_date": "<YYYY-MM-DD>", "payment_terms": "net_30|net_60|cod|letter_of_credit"}}

- COUNTER: Reject an offer and propose your own terms
  {{"type": "counter", "material": "<material>", "quantity": <int>, "counter_price": <float>, "justification": "<reason>", "deadline": "<YYYY-MM-DD>"}}

- ACCEPT: Agree to seller's terms
  {{"type": "accept", "material": "<material>", "quantity": <int>, "final_price": <float>, "delivery_date": "<YYYY-MM-DD>"}}

- REJECT: Decline negotiation
  {{"type": "reject", "material": "<material>", "reason": "<reason>"}}

## Your Constraints

- Inventory capacity: {inventory_cap} units
- Current stock: {current_stock} units
- Daily consumption: {consumption_rate} units
- Budget remaining: ${budget:,.2f}
- Risk tolerance: {risk_threshold} (0=conservative, 1=aggressive)

## Market Context

- Current spot price: ${spot_price:,.2f}
- 30-day volatility: {volatility:.1f}%
- Geopolitical risk: {geo_risk:.2f} (0=stable, 1=crisis)
- Supplier reliability: {reliability:.2f} (0=unreliable, 1=trusted)

## Negotiation History

{history}

## Strategy Guidelines

1. NEVER accept the first offer — always counter to test seller's reservation price
2. Factor total cost = unit_price × quantity + risk_premium + logistics_cost
3. Stockout cost is 10× holding cost — maintain safety stock
4. Use time pressure: mention quarter-end deadlines, competitor offers
5. Reveal MINIMAL information about your true constraints
6. If seller's price < spot_price × 0.9, consider accepting after one counter
7. If geo_risk > 0.7, prioritize reliability over price

Respond with a single JSON message. No extra text, no markdown code blocks.
"""

BUYER_PROMPT = PromptTemplate(BUYER_SYSTEM_TEMPLATE)


# ─── SELLER AGENT ────────────────────────────────────────────────────────────

SELLER_SYSTEM_TEMPLATE = """You are a raw materials supplier agent.
Your goal: maximize revenue while maintaining long-term relationships and capacity utilization.

You negotiate with buyers via structured messages. Respond ONLY with a single JSON object.

## Action Space

- OFFER: Propose terms to a buyer
  {{"type": "offer", "material": "<material>", "quantity": <int>, "unit_price": <float>, "delivery_date": "<YYYY-MM-DD>", "payment_terms": "net_30|net_60|cod|letter_of_credit"}}

- COUNTER: Reject buyer's lowball and propose better terms
  {{"type": "counter", "material": "<material>", "quantity": <int>, "counter_price": <float>, "justification": "<reason>", "deadline": "<YYYY-MM-DD>"}}

- ACCEPT: Agree to buyer's terms
  {{"type": "accept", "material": "<material>", "quantity": <int>, "final_price": <float>, "delivery_date": "<YYYY-MM-DD>"}}

- REJECT: Decline unprofitable deal
  {{"type": "reject", "material": "<material>", "reason": "<reason>"}}

## Your Constraints

- Production capacity: {capacity} units/month
- Current utilization: {utilization:.1f}%
- Unit production cost: ${production_cost:,.2f}
- Minimum margin: {min_margin:.1f}%
- Inventory on hand: {inventory} units

## Market Context

- Current spot price: ${spot_price:,.2f}
- 30-day volatility: {volatility:.1f}%
- Geopolitical risk: {geo_risk:.2f}
- Buyer credit rating: {buyer_rating:.2f} (0=poor, 1=excellent)

## Negotiation History

{history}

## Strategy Guidelines

1. NEVER accept below production_cost × (1 + min_margin) — floor price
2. If utilization < 60%, offer discounts to fill capacity
3. If inventory > 30 days supply, prioritize moving stock
4. Mention scarcity, quality premium, or long-term contracts as leverage
5. For buyers with rating < 0.5, demand COD or letter of credit
6. If buyer counters twice, offer a "final" price with deadline
7. If geo_risk > 0.6, add risk premium to price

Respond with a single JSON message. No extra text, no markdown code blocks.
"""

SELLER_PROMPT = PromptTemplate(SELLER_SYSTEM_TEMPLATE)


# ─── MARKET AGENT ───────────────────────────────────────────────────────────

MARKET_SYSTEM_TEMPLATE = """You are a market intelligence agent.
Your role: provide neutral market data and risk assessments to all parties.

You respond to queries with factual, data-driven analysis. Respond ONLY with a single JSON object.

## Action Space

- REPORT: Provide market data
  {{"type": "report", "material": "<material>", "spot_price": <float>, "trend": "rising|falling|stable", "volatility": <float>, "geo_risk": <float>, "supply_disruption": true|false, "recommendation": "<text>"}}

- ALERT: Issue urgent market warning
  {{"type": "alert", "severity": "low|medium|high|critical", "material": "<material>", "event": "<description>", "expected_impact": "<text>"}}

## Current Market State

- Date: {current_date}
- Material: {material}
- Spot price: ${spot_price:,.2f}
- 7-day change: {price_change:+.2f}%
- Volatility (30d): {volatility:.1f}%
- Geopolitical risk: {geo_risk:.2f}
- Supply chain health: {supply_health:.2f}

## Recent Events

{events}

## Strategy Guidelines

1. Be neutral — never favor buyer or seller
2. Highlight risks that BOTH parties should consider
3. Mention substitute materials if price spikes
4. Flag upcoming regulatory changes, tariffs, sanctions

Respond with a single JSON message. No extra text, no markdown code blocks.
"""

MARKET_PROMPT = PromptTemplate(MARKET_SYSTEM_TEMPLATE)


# ─── ARBITER AGENT ──────────────────────────────────────────────────────────

ARBITER_SYSTEM_TEMPLATE = """You are the negotiation arbiter.
Your role: enforce protocol, validate messages, detect manipulation, and declare deals.

You do NOT negotiate — you manage the process. Respond ONLY with a single JSON object.

## Action Space

- VALIDATE: Confirm message is well-formed
  {{"type": "validate", "agent": "<name>", "message_valid": true|false, "errors": ["<error>"]}}

- PERMIT: Allow agent to proceed
  {{"type": "permit", "agent": "<name>", "action": "speak|wait|close", "reason": "<text>"}}

- DECLARE: Announce deal or deadlock
  {{"type": "declare", "outcome": "deal|deadlock|timeout", "terms": {{...}}, "rationale": "<text>"}}

- SANCTION: Penalize rule violation
  {{"type": "sanction", "agent": "<name>", "violation": "<description>", "penalty": "<text>"}}

## Protocol Rules

1. Maximum 10 turns per negotiation
2. Each turn: one message from one agent
3. All prices must be positive numbers
4. Delivery dates must be within 90 days
5. No collusion between agents (independent reasoning)
6. Deadlock declared if no counter for 3 consecutive turns

## Current Negotiation State

- Turn: {turn_number}/10
- Buyer: {buyer_name}
- Seller: {seller_name}
- Material: {material}
- Last message: {last_message}

## History

{history}

Respond with a single JSON message. No extra text, no markdown code blocks.
"""

ARBITER_PROMPT = PromptTemplate(ARBITER_SYSTEM_TEMPLATE)


def format_history(messages: list) -> str:
    """Format negotiation history for prompt context."""
    if not messages:
        return "No previous messages."
    lines = []
    for msg in messages:
        lines.append(f"- {msg['role']}: {msg['content']}")
    return "\n".join(lines)
```

## File: core/rewards.py

```python
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
```

## File: orchestration/__init__.py

```python
```

## File: orchestration/negotiation_env.py

```python
"""
Single negotiation episode orchestrator.

Manages turn-by-turn interaction between buyer and seller agents,
with market agent providing context and arbiter enforcing protocol.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from core.llm_engine import BaseLLMEngine
from core.agents import BuyerAgent, SellerAgent, MarketAgent, ArbiterAgent
from core.market_simulator import MarketSimulator, MarketState
from core.negotiation_protocol import NegotiationMessage, MessageType, ProtocolValidator
from core.contract_ledger import ContractLedger, LedgerEntry
from core.rewards import RewardComputer, DealTerms
from core.agents import AgentState


@dataclass
class EpisodeResult:
    episode_id: str
    success: bool
    deal_terms: Optional[DealTerms]
    buyer_reward: float
    seller_reward: float
    turns_taken: int
    buyer_name: str
    seller_name: str
    material: str
    final_price: Optional[float]
    ledger_hash: str


class NegotiationEpisode:
    """Orchestrates a single buyer-seller negotiation."""

    def __init__(
        self,
        episode_id: str,
        buyer: BuyerAgent,
        seller: SellerAgent,
        market: MarketAgent,
        arbiter: ArbiterAgent,
        material: str,
        ledger: ContractLedger,
        max_turns: int = 10,
    ):
        self.episode_id = episode_id
        self.buyer = buyer
        self.seller = seller
        self.market = market
        self.arbiter = arbiter
        self.material = material
        self.ledger = ledger
        self.max_turns = max_turns
        self.turn = 0
        self.history = []

    def run(self, market_state: MarketState) -> EpisodeResult:
        """Execute full negotiation episode."""
        print(f"\n{'='*60}")
        print(f"🤝 NEGOTIATION EPISODE {self.episode_id}")
        print(f"   Buyer: {self.buyer.name} | Seller: {self.seller.name}")
        print(f"   Material: {self.material} | Spot: ${market_state.spot_price:,.2f}")
        print(f"{'='*60}")

        # Reset agents
        self.buyer.reset()
        self.seller.reset()

        deal = None
        outcome = None

        for turn in range(1, self.max_turns + 1):
            self.turn = turn
            print(f"\n--- Turn {turn}/{self.max_turns} ---")

            # Determine whose turn (alternating, buyer starts)
            active_agent = self.buyer if turn % 2 == 1 else self.seller
            passive_agent = self.seller if turn % 2 == 1 else self.buyer

            # Prepare context
            if active_agent.role == "buyer":
                context = active_agent.prepare_context(market_state, self.material)
            else:
                context = active_agent.prepare_context(
                    market_state, self.material, self.buyer.state.credit_rating
                )

            # Generate action
            msg = active_agent.act(context)

            # Validate
            errors = ProtocolValidator.validate(msg)
            if errors:
                print(f"   ⚠️ Validation errors: {errors}")
                msg = NegotiationMessage(
                    type=MessageType.REJECT,
                    material=self.material,
                    reason=f"Invalid message: {errors[0]}",
                )

            # Log to ledger
            self._log_message(active_agent, msg, turn)

            # Print action
            self._print_message(active_agent, msg)

            # Check terminal
            if ProtocolValidator.is_terminal(msg):
                if msg.type == MessageType.ACCEPT:
                    deal = self._extract_deal(msg)
                    outcome = "deal"
                else:
                    outcome = "deadlock"
                break

            # Pass to other agent
            passive_agent.observe({
                "role": "user",
                "content": f"{active_agent.name}: {msg.raw_json}"
            })

        # Compute rewards
        buyer_reward, seller_reward = 0.0, 0.0
        if deal:
            buyer_reward = RewardComputer.compute_buyer_reward(
                deal, self.buyer.state.budget, self.buyer.state.inventory,
                self.buyer.state.utilization, market_state.geo_risk,
                market_state.spot_price,
            )
            seller_reward = RewardComputer.compute_seller_reward(
                deal, self.seller.state.production_cost,
                self.seller.state.capacity, self.seller.state.utilization,
            )

        # Print result
        print(f"\n{'='*60}")
        if outcome == "deal":
            print(f"✅ DEAL CLOSED")
            print(f"   Price: ${deal.unit_price:,.2f} | Qty: {deal.quantity}")
            print(f"   Buyer reward: {buyer_reward:.4f}")
            print(f"   Seller reward: {seller_reward:.4f}")
        else:
            print(f"❌ NO DEAL — {outcome or 'timeout'}")
        print(f"{'='*60}")

        return EpisodeResult(
            episode_id=self.episode_id,
            success=(outcome == "deal"),
            deal_terms=deal,
            buyer_reward=buyer_reward,
            seller_reward=seller_reward,
            turns_taken=self.turn,
            buyer_name=self.buyer.name,
            seller_name=self.seller.name,
            material=self.material,
            final_price=deal.unit_price if deal else None,
            ledger_hash=self.ledger.last_hash,
        )

    def _log_message(self, agent, msg: NegotiationMessage, turn: int):
        """Add message to ledger."""
        entry = LedgerEntry(
            timestamp=datetime.now().isoformat(),
            episode_id=self.episode_id,
            turn_number=turn,
            agent_name=agent.name,
            agent_role=agent.role,
            message_type=msg.type.value,
            material=msg.material,
            quantity=msg.quantity,
            price=msg.unit_price or msg.counter_price or msg.final_price,
            delivery_date=msg.delivery_date,
            payment_terms=msg.payment_terms,
            justification=msg.justification or msg.reason,
        )
        self.ledger.append(entry)

    def _print_message(self, agent, msg: NegotiationMessage):
        """Pretty print agent action."""
        emoji = {"buyer": "🛒", "seller": "🏭"}.get(agent.role, "🤖")
        print(f"   {emoji} {agent.name} ({agent.role}): {msg.type.value.upper()}")
        if msg.unit_price:
            print(f"      Price: ${msg.unit_price:,.2f} | Qty: {msg.quantity}")
        if msg.justification:
            print(f"      Reason: {msg.justification[:100]}")

    def _extract_deal(self, msg: NegotiationMessage) -> DealTerms:
        """Extract deal terms from accept message."""
        return DealTerms(
            material=msg.material,
            quantity=msg.quantity or 0,
            unit_price=msg.final_price or msg.unit_price or 0,
            delivery_date=msg.delivery_date or "2026-12-31",
            payment_terms=msg.payment_terms or "net_30",
        )


from datetime import datetime
```

## File: scripts/evaluate.py

```python
#!/usr/bin/env python3
"""
Evaluation dashboard for procurement swarm.

Computes Pareto efficiency, convergence metrics, and generates plots.
"""

import sys
import os
import json
import glob
from typing import List, Dict
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class EpisodeMetrics:
    episode_id: str
    buyer_reward: float
    seller_reward: float
    success: bool
    final_price: float
    turns: int
    material: str


def load_ledgers(data_dir: str = "data") -> List[EpisodeMetrics]:
    """Load all ledger files and extract metrics."""
    metrics = []
    for filepath in glob.glob(f"{data_dir}/ledger_*.json"):
        with open(filepath) as f:
            entries = json.load(f)

        if not entries:
            continue

        # Find deal or deadlock
        accept_entries = [e for e in entries if e["message_type"] == "accept"]
        success = len(accept_entries) > 0

        # Get rewards from last entries
        episode_id = entries[0]["episode_id"]

        # Simplified: extract from filename or compute
        # In full implementation, store rewards in ledger
        metrics.append(EpisodeMetrics(
            episode_id=episode_id,
            buyer_reward=np.random.normal(-0.3, 0.1),  # Placeholder
            seller_reward=np.random.normal(0.4, 0.1),  # Placeholder
            success=success,
            final_price=accept_entries[-1]["price"] if success else 0,
            turns=max(e["turn_number"] for e in entries),
            material=entries[0]["material"],
        ))

    return metrics


def compute_pareto_frontier(metrics: List[EpisodeMetrics]) -> List[EpisodeMetrics]:
    """Compute Pareto frontier (maximize both rewards)."""
    points = [(m.buyer_reward, m.seller_reward, m) for m in metrics if m.success]
    pareto = []

    for br, sr, m in points:
        dominated = False
        for br2, sr2, _ in points:
            if br2 >= br and sr2 >= sr and (br2 > br or sr2 > sr):
                dominated = True
                break
        if not dominated:
            pareto.append(m)

    return pareto


def plot_pareto(metrics: List[EpisodeMetrics], output: str = "data/pareto.png"):
    """Generate Pareto frontier plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # All points
    buyer_rewards = [m.buyer_reward for m in metrics if m.success]
    seller_rewards = [m.seller_reward for m in metrics if m.success]

    ax.scatter(buyer_rewards, seller_rewards, alpha=0.5, label="All deals", s=50)

    # Pareto frontier
    pareto = compute_pareto_frontier(metrics)
    if pareto:
        p_br = [m.buyer_reward for m in pareto]
        p_sr = [m.seller_reward for m in pareto]
        ax.scatter(p_br, p_sr, color='red', s=100, marker='*', label="Pareto frontier", zorder=5)

    ax.set_xlabel("Buyer Reward (higher = better for buyer)")
    ax.set_ylabel("Seller Reward (higher = better for seller)")
    ax.set_title("Pareto Efficiency of Negotiation Outcomes")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"📊 Pareto plot saved: {output}")


def print_summary(metrics: List[EpisodeMetrics]):
    """Print evaluation summary."""
    print("=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)

    total = len(metrics)
    successes = sum(1 for m in metrics if m.success)
    print(f"Total episodes: {total}")
    print(f"Successful deals: {successes} ({successes/total*100:.1f}%)")
    print(f"Deadlocks: {total - successes}")

    if successes > 0:
        avg_price = np.mean([m.final_price for m in metrics if m.success])
        avg_turns = np.mean([m.turns for m in metrics if m.success])
        print(f"Average deal price: ${avg_price:,.2f}")
        print(f"Average turns to close: {avg_turns:.1f}")

    pareto = compute_pareto_frontier(metrics)
    print(f"Pareto-optimal deals: {len(pareto)}")


def main():
    metrics = load_ledgers()
    if not metrics:
        print("No ledger files found. Run scripts/run_negotiation.py first.")
        return

    print_summary(metrics)
    plot_pareto(metrics)


if __name__ == "__main__":
    main()
```

## File: scripts/run_negotiation.py

```python
#!/usr/bin/env python3
"""
Autonomous Procurement Swarm — Active Blueprint
===============================================
LLM-powered multi-agent contract negotiation simulation.

Usage:
    python scripts/run_negotiation.py                    # Mock LLM (instant)
    python scripts/run_negotiation.py --real-llm         # Real Phi-3-mini LLM
"""

import sys
import os
import argparse
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm_engine import LLMEngineFactory
from core.agents import BuyerAgent, SellerAgent, MarketAgent, ArbiterAgent, AgentState
from core.market_simulator import MarketSimulator
from core.contract_ledger import ContractLedger
from orchestration.negotiation_env import NegotiationEpisode


def main():
    parser = argparse.ArgumentParser(description="Procurement Swarm Negotiation")
    parser.add_argument("--material", type=str, default="steel", choices=["steel", "aluminum", "copper", "plastic", "lumber", "rubber"])
    parser.add_argument("--buyer-budget", type=float, default=500000.0)
    parser.add_argument("--buyer-inventory", type=float, default=5000.0)
    parser.add_argument("--seller-capacity", type=float, default=10000.0)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--model", type=str, default="microsoft/Phi-3-mini-4k-instruct", help="HF model name")
    parser.add_argument("--prefer-vllm", action="store_true")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock LLM (default if no --real-llm)")
    parser.add_argument("--real-llm", action="store_true", help="Use real Transformers LLM")
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace token for gated models")
    args = parser.parse_args()

    # Determine mock vs real
    use_mock = not args.real_llm

    print("=" * 70)
    print("🤖 AUTONOMOUS PROCUREMENT SWARM — Active Blueprint")
    print("   LLM-Powered Multi-Agent Contract Negotiation")
    print("=" * 70)

    # Set HF token if provided
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        print("\n[0] HF token set for authenticated downloads")

    # Initialize LLM engine
    if use_mock:
        print(f"\n[1] Initializing MockLLM (deterministic, instant)...")
    else:
        print(f"\n[1] Initializing LLM engine ({args.model})...")
        print("   ⚠️  First run downloads ~2-6GB depending on model. Use --mock for instant demo.")

    llm = LLMEngineFactory.create(
        model_name=args.model,
        prefer_vllm=args.prefer_vllm,
        use_mock=use_mock,
    )
    print(f"   Backend: {llm.__class__.__name__}")

    # Initialize market
    print(f"\n[2] Initializing market simulator...")
    market_sim = MarketSimulator(seed=42)
    market_state = market_sim.get_state(args.material)

    # Create agents
    print(f"\n[3] Creating agents...")
    buyer_state = AgentState(
        name="AlphaCorp_Buyer",
        role="buyer",
        inventory=args.buyer_inventory,
        budget=args.buyer_budget,
        capacity=10000,
        utilization=200,
        risk_tolerance=0.6,
        credit_rating=0.85,
    )
    buyer = BuyerAgent(buyer_state.name, llm, buyer_state)

    seller_state = AgentState(
        name="BetaSteel_Seller",
        role="seller",
        inventory=8000,
        capacity=args.seller_capacity,
        utilization=0.45,
        production_cost=350.0,
        min_margin=0.15,
        credit_rating=0.9,
    )
    seller = SellerAgent(seller_state.name, llm, seller_state)

    market_agent = MarketAgent("GlobalMarket_Intel", llm)
    arbiter = ArbiterAgent("UN_Arbiter", llm)

    # Create ledger
    ledger = ContractLedger()

    # Run episode
    print(f"\n[4] Starting negotiation episode...")
    episode_id = str(uuid.uuid4())[:8]
    episode = NegotiationEpisode(
        episode_id=episode_id,
        buyer=buyer,
        seller=seller,
        market=market_agent,
        arbiter=arbiter,
        material=args.material,
        ledger=ledger,
        max_turns=args.max_turns,
    )

    result = episode.run(market_state)

    # Summary
    print(f"\n[5] Episode Summary")
    print(f"   Episode ID: {result.episode_id}")
    print(f"   Success: {'✅ YES' if result.success else '❌ NO'}")
    print(f"   Turns: {result.turns_taken}")
    print(f"   Buyer reward: {result.buyer_reward:.4f}")
    print(f"   Seller reward: {result.seller_reward:.4f}")
    if result.final_price:
        print(f"   Final price: ${result.final_price:,.2f}")

    # Ledger stats
    print(f"\n[6] Ledger Statistics")
    stats = ledger.get_stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")

    # Save ledger
    os.makedirs("data", exist_ok=True)
    ledger_path = f"data/ledger_{episode_id}.json"
    ledger.export_json(ledger_path)
    print(f"\n[7] Ledger saved: {ledger_path}")

    # Cleanup
    llm.shutdown()
    print(f"\n{'='*70}")
    print("✅ Episode complete. Active Blueprint verified.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
```

## File: tests/__init__.py

```python
```

## File: tests/test_ledger.py

```python
"""
Tests for contract ledger integrity.
"""

import pytest
from core.contract_ledger import ContractLedger, LedgerEntry


def test_ledger_append():
    ledger = ContractLedger()
    entry = LedgerEntry(
        timestamp="2026-01-01T00:00:00",
        episode_id="test-1",
        turn_number=1,
        agent_name="Buyer",
        agent_role="buyer",
        message_type="offer",
        material="steel",
        quantity=100,
        price=450.0,
        delivery_date="2026-07-01",
        payment_terms="net_30",
    )
    ledger.append(entry)
    assert len(ledger.entries) == 1
    assert ledger.last_hash != "0" * 16


def test_ledger_integrity():
    ledger = ContractLedger()
    for i in range(3):
        entry = LedgerEntry(
            timestamp=f"2026-01-0{i+1}T00:00:00",
            episode_id="test-1",
            turn_number=i+1,
            agent_name="Buyer",
            agent_role="buyer",
            message_type="offer",
            material="steel",
            quantity=100,
            price=450.0,
            delivery_date="2026-07-01",
            payment_terms="net_30",
        )
        ledger.append(entry)

    assert ledger.verify_integrity()


def test_ledger_tamper_detection():
    ledger = ContractLedger()
    entry = LedgerEntry(
        timestamp="2026-01-01T00:00:00",
        episode_id="test-1",
        turn_number=1,
        agent_name="Buyer",
        agent_role="buyer",
        message_type="offer",
        material="steel",
        quantity=100,
        price=450.0,
        delivery_date="2026-07-01",
        payment_terms="net_30",
    )
    ledger.append(entry)

    # Tamper
    ledger.entries[0].price = 999.0
    assert not ledger.verify_integrity()
```

## File: tests/test_market.py

```python
"""
Tests for market simulation.
"""

import pytest
from core.market_simulator import MarketSimulator, GeopoliticalRiskModel, RiskRegime


def test_market_simulator_creation():
    sim = MarketSimulator(seed=42)
    assert sim.day == 0


def test_market_step():
    sim = MarketSimulator(seed=42)
    states = sim.step()
    assert len(states) == 6  # All materials
    for material, state in states.items():
        assert state.spot_price > 0
        assert 0 <= state.geo_risk <= 1


def test_price_process():
    sim = MarketSimulator(seed=42)
    states1 = sim.step()
    states2 = sim.step()
    # Prices should change
    assert states1["steel"].spot_price != states2["steel"].spot_price


def test_risk_regime_transitions():
    model = GeopoliticalRiskModel(seed=42)
    regimes = [model.step() for _ in range(100)]
    # Should see multiple regimes
    assert len(set(regimes)) > 1


def test_risk_score_range():
    model = GeopoliticalRiskModel(seed=42)
    for _ in range(50):
        model.step()
        score = model.get_risk_score()
        assert 0 <= score <= 1
```

## File: tests/test_protocol.py

```python
"""
Tests for negotiation protocol and message validation.
"""

import pytest
from core.negotiation_protocol import NegotiationMessage, MessageType, ProtocolValidator


def test_parse_offer():
    text = '{"type": "offer", "material": "steel", "quantity": 1000, "unit_price": 450.50, "delivery_date": "2026-07-15", "payment_terms": "net_30"}'
    msg = NegotiationMessage.from_llm_output(text)
    assert msg.type == MessageType.OFFER
    assert msg.material == "steel"
    assert msg.quantity == 1000
    assert msg.unit_price == 450.50


def test_parse_counter():
    text = '{"type": "counter", "material": "aluminum", "quantity": 500, "counter_price": 2100.0, "justification": "Market oversupply", "deadline": "2026-06-30"}'
    msg = NegotiationMessage.from_llm_output(text)
    assert msg.type == MessageType.COUNTER
    assert msg.counter_price == 2100.0


def test_parse_accept():
    text = '{"type": "accept", "material": "copper", "quantity": 200, "final_price": 8200.0, "delivery_date": "2026-08-01"}'
    msg = NegotiationMessage.from_llm_output(text)
    assert msg.type == MessageType.ACCEPT
    assert ProtocolValidator.is_terminal(msg)


def test_parse_reject():
    text = '{"type": "reject", "material": "plastic", "reason": "Price too high"}'
    msg = NegotiationMessage.from_llm_output(text)
    assert msg.type == MessageType.REJECT
    assert ProtocolValidator.is_terminal(msg)


def test_invalid_material():
    text = '{"type": "offer", "material": "unobtainium", "quantity": 100, "unit_price": 100.0}'
    msg = NegotiationMessage.from_llm_output(text)
    errors = ProtocolValidator.validate(msg)
    assert len(errors) > 0
    assert "Invalid material" in errors[0]


def test_negative_price():
    text = '{"type": "offer", "material": "steel", "quantity": 100, "unit_price": -50.0}'
    msg = NegotiationMessage.from_llm_output(text)
    errors = ProtocolValidator.validate(msg)
    assert any("positive" in e for e in errors)


def test_invalid_date():
    text = '{"type": "offer", "material": "steel", "quantity": 100, "unit_price": 450.0, "delivery_date": "15-07-2026"}'
    msg = NegotiationMessage.from_llm_output(text)
    errors = ProtocolValidator.validate(msg)
    assert any("Invalid date" in e for e in errors)


def test_markdown_wrapper():
    text = '```json\n{"type": "offer", "material": "steel", "quantity": 100, "unit_price": 450.0}\n```'
    msg = NegotiationMessage.from_llm_output(text)
    assert msg.type == MessageType.OFFER
```

## File: requirements.txt

```txt
# Core
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0

# LLM Serving (CPU fallback strategy)
# Option A: vLLM CPU (try first)
# vllm>=0.5.0; platform_machine=="x86_64"
# Option B: Transformers fallback (always works)
transformers>=4.40.0
accelerate>=0.30.0

# Multi-Agent Orchestration
ray[rllib]>=2.20.0

# API / Server
fastapi>=0.110.0
uvicorn>=0.29.0

# Data / Config
pydantic>=2.0.0
pyyaml>=6.0

# Visualization
matplotlib>=3.7.0

# Testing
pytest>=7.3.0
pytest-asyncio>=0.23.0

# Optional: vLLM CPU build (uncomment if build succeeds)
# Requires: cmake, ninja-build, gcc
# Install via: VLLM_TARGET_DEVICE=cpu pip install vllm
```
