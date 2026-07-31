"""Deterministic policy engine for procurement compliance."""

from typing import Tuple, Set
from pydantic import BaseModel, Field

class PolicyContext(BaseModel):
    buyer_max_budget_total: float = Field(gt=0)
    blacklisted_vendors: Set[str] = Field(default_factory=set)
    max_carbon_limit_kg: float = Field(default=10_000.0, ge=0)
    max_unit_price: float = Field(gt=0)
    required_bid_bond_pct: float = Field(default=0.05, ge=0.0, le=1.0)

class PolicyEngine:
    """
    In-process, sub-millisecond policy validator.
    NO LLM involvement. Pure deterministic rules.
    """
    
    def evaluate_bid(self, bid: dict, context: PolicyContext) -> Tuple[bool, str]:
        """
        Returns (passed: bool, reason: str).
        If passed is False, reason contains the rejection code.
        """
        # Rule 1: Blacklist
        if bid.get("supplier_id") in context.blacklisted_vendors:
            return False, "REJECT_VENDOR_BLACKLISTED"
        
        # Rule 2: Unit price cap
        if bid.get("unit_price", float("inf")) > context.max_unit_price:
            return False, "REJECT_EXCEEDS_MAX_UNIT_PRICE"
        
        # Rule 3: Total budget cap
        total_amount = bid.get("unit_price", 0) * bid.get("quantity", 0)
        if total_amount > context.buyer_max_budget_total:
            return False, "REJECT_EXCEEDS_BUDGET"
        
        # Rule 4: ESG carbon limit
        if bid.get("carbon_footprint_kg", 0) > context.max_carbon_limit_kg:
            return False, "REJECT_ESG_VIOLATION"
        
        # Rule 5: Bid bond
        required_bond = total_amount * context.required_bid_bond_pct
        if bid.get("bid_bond_amount", 0) < required_bond:
            return False, "REJECT_INSUFFICIENT_BID_BOND"
        
        return True, "POLICY_PASSED"
    
    def evaluate_rfq(self, rfq: dict) -> Tuple[bool, str]:
        """Basic RFQ validation."""
        if rfq.get("quantity", 0) <= 0:
            return False, "REJECT_INVALID_QUANTITY"
        if rfq.get("max_unit_price", 0) <= 0:
            return False, "REJECT_INVALID_PRICE"
        return True, "POLICY_PASSED"
