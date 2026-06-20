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
