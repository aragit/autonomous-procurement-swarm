"""Buyer orchestrator for 1×N sealed-bid CNP auctions."""

from typing import List, Dict, Any, Tuple
from core.llm_engine import BaseLLMEngine
from core.protocol.schema import RFQPayload, BidPayload, CNPMessage, MessageType
from core.protocol.policy_engine import PolicyEngine, PolicyContext

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
    ):
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
        blacklisted: set = None,
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
        bids: List[CNPMessage],
        policy_context: PolicyContext,
    ) -> Tuple[List[BidPayload], List[Dict[str, str]]]:
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
                rejections.append({
                    "supplier_id": bid_data.get("supplier_id", "unknown"),
                    "reason": reason,
                })
        
        return valid, rejections
