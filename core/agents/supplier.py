"""Supplier agent for CNP sealed-bid auctions."""

from dataclasses import dataclass

from core.llm_engine import BaseLLMEngine
from core.protocol.schema import BidPayload, CNPMessage, MessageType, RFQPayload


@dataclass
class CostModel:
    """Heterogeneous cost structure per supplier."""

    base_cost_per_unit: float
    logistics_premium_per_unit: float
    capacity_units: int
    current_utilization: float  # 0.0 - 1.0
    min_margin_pct: float
    reliability_score: float
    esg_carbon_per_unit: float  # kg CO2e per unit


class SupplierAgent:
    """
    Supplier agent that responds to RFQs with sealed bids.
    Each supplier has a unique CostModel (miner vs distributor vs recycler).
    """

    def __init__(
        self,
        supplier_id: str,
        llm_engine: BaseLLMEngine,
        cost_model: CostModel,
    ):
        self.supplier_id = supplier_id
        self.llm = llm_engine
        self.cost = cost_model

    def _compute_floor_price(self, quantity: int) -> float:
        """Minimum viable price based on cost structure."""
        unit_cost = self.cost.base_cost_per_unit + self.cost.logistics_premium_per_unit
        return unit_cost * (1 + self.cost.min_margin_pct)

    def _has_capacity(self, quantity: int) -> bool:
        available = self.cost.capacity_units * (1 - self.cost.current_utilization)
        return available >= quantity

    async def respond_to_rfq(self, rfq: RFQPayload) -> CNPMessage:
        """
        Async method: generate a bid in response to RFQ.
        Uses LLM for strategic pricing above floor price.
        Timeout handled by orchestrator, not here.
        """
        if not self._has_capacity(rfq.quantity):
            return CNPMessage.from_payload(
                MessageType.REJECT_BID,
                {
                    "session_id": rfq.session_id,
                    "supplier_id": self.supplier_id,
                    "reason": "INSUFFICIENT_CAPACITY",
                },
            )

        floor_price = self._compute_floor_price(rfq.quantity)

        # Strategic markup (can be enhanced with memory in Sprint 5)
        # For now: bid between floor and max_unit_price
        import random

        markup = random.uniform(0.0, 0.15)  # 0-15% above floor
        bid_price = min(
            floor_price * (1 + markup),
            rfq.max_unit_price * 0.98,  # Leave slight room
        )

        # Ensure we don't bid below floor
        bid_price = max(bid_price, floor_price)

        bid = BidPayload(
            session_id=rfq.session_id,
            supplier_id=self.supplier_id,
            unit_price=round(bid_price, 2),
            lead_time_days=rfq.target_lead_time_days + random.randint(-5, 10),
            carbon_footprint_kg=self.cost.esg_carbon_per_unit * rfq.quantity,
            reliability_score=self.cost.reliability_score,
            bid_bond_amount=round(bid_price * rfq.quantity * rfq.required_bid_bond_pct, 2),
            delivery_date=rfq.delivery_window_start,
            justification=f"Bid from {self.supplier_id} at {self.cost.min_margin_pct:.0%} margin",
        )

        return CNPMessage.from_payload(MessageType.BID, bid)

    async def respond_to_offer(
        self,
        offer_price: float,
        offer_quantity: int,
        material: str,
        turn: int,
        max_turns: int,
    ) -> CNPMessage:
        """
        Deterministic bilateral bartering response.
        Uses floor price and margin rules — NO LLM randomness for reliability.
        """
        floor = self._compute_floor_price(offer_quantity)

        # Last turn: binary accept/reject
        if turn >= max_turns - 1:
            if offer_price >= floor:
                return CNPMessage.from_payload(
                    MessageType.ACCEPT,
                    {
                        "session_id": "bilateral",
                        "supplier_id": self.supplier_id,
                        "material": material,
                        "quantity": offer_quantity,
                        "final_price": round(offer_price, 2),
                        "delivery_date": "2026-08-15",
                    },
                )
            else:
                return CNPMessage.from_payload(
                    MessageType.REJECT,
                    {
                        "session_id": "bilateral",
                        "supplier_id": self.supplier_id,
                        "material": material,
                        "reason": f"Below floor price of ${floor:.2f}",
                    },
                )

        # Accept immediately if very profitable
        if offer_price >= floor * 1.10:
            return CNPMessage.from_payload(
                MessageType.ACCEPT,
                {
                    "session_id": "bilateral",
                    "supplier_id": self.supplier_id,
                    "material": material,
                    "quantity": offer_quantity,
                    "final_price": round(offer_price, 2),
                    "delivery_date": "2026-08-15",
                },
            )

        # Counter: move 25% from floor toward buyer's offer
        counter_price = floor + (offer_price - floor) * 0.75
        # Ensure counter is strictly better than previous offer for buyer
        counter_price = min(counter_price, offer_price * 0.98)
        counter_price = max(counter_price, floor)

        return CNPMessage.from_payload(
            MessageType.COUNTER,
            {
                "session_id": "bilateral",
                "supplier_id": self.supplier_id,
                "material": material,
                "quantity": offer_quantity,
                "counter_price": round(counter_price, 2),
                "justification": f"{self.supplier_id} counter — floor price plus margin",
                "deadline": "2026-08-10",
            },
        )
