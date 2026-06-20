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
