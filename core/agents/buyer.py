"""Buyer orchestrator for 1×N sealed-bid CNP auctions."""

from typing import TYPE_CHECKING, Any

from core.llm_engine import BaseLLMEngine
from core.protocol.policy_engine import PolicyContext, PolicyEngine
from core.protocol.schema import BidPayload, CNPMessage, MessageType, RFQPayload

if TYPE_CHECKING:
    from core.agents.supplier import SupplierAgent
    from core.memory.heuristics import HeuristicReservationEstimator


class BuyerOrchestrator:
    """
    Buyer agent that:
    1. Creates and broadcasts RFQs
    2. Evaluates bids via PolicyEngine
    3. Ranks valid bids (scoring engine comes in Sprint 2)
    """

    def __init__(
        self,
        buyer_id: str,
        llm_engine: BaseLLMEngine,
        policy_engine: PolicyEngine,
    ) -> None:
        self.buyer_id = buyer_id
        self.llm = llm_engine
        self.policy = policy_engine

    def create_rfq(
        self,
        session_id: str,
        material: str,
        quantity: int,
        max_unit_price: float,
        target_lead_time_days: int,
        budget: float,
        blacklisted: set[str] | None = None,
    ) -> RFQPayload:
        """Construct a validated RFQ."""
        from datetime import datetime, timedelta

        today = datetime.now()

        return RFQPayload(
            session_id=session_id,
            material=material,
            quantity=quantity,
            max_unit_price=max_unit_price,
            target_lead_time_days=target_lead_time_days,
            delivery_window_start=(today + timedelta(days=14)).strftime("%Y-%m-%d"),
            delivery_window_end=(today + timedelta(days=90)).strftime("%Y-%m-%d"),
            payment_terms="net_30",
            required_bid_bond_pct=0.05,
        )

    def evaluate_bids(
        self,
        bids: list[CNPMessage],
        policy_context: PolicyContext,
    ) -> tuple[list[BidPayload], list[dict[str, str]]]:
        """
        Filter bids through PolicyEngine.
        Returns: (valid_bids, rejections)
        """
        valid = []
        rejections = []

        for bid_msg in bids:
            if bid_msg.type != MessageType.BID:
                continue

            bid_data = bid_msg.payload
            passed, reason = self.policy.evaluate_bid(bid_data, policy_context)

            if passed:
                valid.append(BidPayload(**bid_data))
            else:
                rejections.append(
                    {
                        "supplier_id": bid_data.get("supplier_id", "unknown"),
                        "reason": reason,
                    }
                )

        return valid, rejections

    async def barter_with_supplier(
        self,
        supplier: "SupplierAgent",
        rfq: RFQPayload,
        initial_bid_price: float,
        max_turns: int = 4,
        memory: "HeuristicReservationEstimator | None" = None,
        market_spot_price: float | None = None,
    ) -> dict[str, Any]:
        """
        Execute bilateral bartering thread with one shortlisted supplier.
        Returns: {"success": bool, "final_price": float|None, "turns": int, "history": list}
        """
        from core.protocol.fsm import BilateralFSM
        from core.protocol.schema import MessageType

        # BilateralFSM counts total messages (incl. the opening OFFER, and both
        # sides' counters). A full thread is OFFER + up to (max_turns-1) counter
        # pairs + final supplier accept/reject = 2*max_turns messages, so budget
        # accordingly or the last-turn accept/reject is never reached.
        fsm = BilateralFSM(max_turns=max_turns * 2)

        # If we have memory for this supplier, open closer to their estimated floor
        opening_discount = 0.92  # default
        if memory and market_spot_price:
            estimated_floor = memory.estimate_reservation_price(
                supplier.supplier_id, market_spot_price
            )
            # Open at 95% of estimated floor if it's lower than 92% of bid
            if estimated_floor < initial_bid_price * 0.92:
                opening_discount = estimated_floor / initial_bid_price
                opening_discount = max(opening_discount, 0.80)  # Never open below 80%

        # Buyer opens at a discount of supplier's original bid
        current_price = initial_bid_price * opening_discount

        # Initial OFFER
        offer_payload = {
            "session_id": rfq.session_id,
            "material": rfq.material,
            "quantity": rfq.quantity,
            "unit_price": round(current_price, 2),
            "delivery_date": rfq.delivery_window_start,
            "payment_terms": rfq.payment_terms,
        }
        fsm.record_message(MessageType.OFFER, self.buyer_id, offer_payload)

        for turn in range(max_turns):
            # Supplier responds
            supplier_msg = await supplier.respond_to_offer(
                current_price, rfq.quantity, rfq.material, turn, max_turns
            )

            if not fsm.record_message(
                supplier_msg.type, supplier.supplier_id, supplier_msg.payload
            ):
                break

            if supplier_msg.type == MessageType.ACCEPT:
                return {
                    "success": True,
                    "supplier_id": supplier.supplier_id,
                    "final_price": supplier_msg.payload.get("final_price", current_price),
                    "turns": fsm.turn_count,
                    "history": fsm.history,
                    "fsm_summary": fsm.get_summary(),
                }

            if supplier_msg.type == MessageType.REJECT:
                return {
                    "success": False,
                    "supplier_id": supplier.supplier_id,
                    "final_price": None,
                    "turns": fsm.turn_count,
                    "history": fsm.history,
                    "fsm_summary": fsm.get_summary(),
                }

            if supplier_msg.type == MessageType.COUNTER:
                counter_price = supplier_msg.payload.get("counter_price")
                if counter_price is None:
                    break

                # Buyer counters back: meet halfway
                current_price = (current_price + counter_price) / 2

                buyer_counter = {
                    "session_id": rfq.session_id,
                    "material": rfq.material,
                    "quantity": rfq.quantity,
                    "counter_price": round(current_price, 2),
                    "justification": "Buyer split-the-difference counter",
                    "deadline": rfq.delivery_window_end,
                }

                if not fsm.record_message(MessageType.COUNTER, self.buyer_id, buyer_counter):
                    break

        # Max turns reached without terminal
        return {
            "success": False,
            "supplier_id": supplier.supplier_id,
            "final_price": None,
            "turns": fsm.turn_count,
            "history": fsm.history,
            "fsm_summary": fsm.get_summary(),
        }
