"""Orchestrates parallel sealed-bid collection from N suppliers."""

import asyncio
from typing import List, Dict, Any, Tuple
from core.protocol.schema import RFQPayload, CNPMessage, MessageType, BidPayload
from core.protocol.fsm import GlobalAuctionFSM, GlobalAuctionState
from core.protocol.policy_engine import PolicyEngine, PolicyContext
from core.agents.supplier import SupplierAgent
from core.evaluator.scoring import MultiCriteriaEvaluator, EvaluationWeights
from configs.settings import settings

try:
    from core.ledger.repository import PostgresLedgerRepository  # type: ignore  # noqa: F401
except ImportError:  # pragma: no cover - not implemented until Sprint 4
    PostgresLedgerRepository = None  # type: ignore

class AuctionOrchestrator:
    """
    Runs the complete CNP auction lifecycle:
    1. Broadcast RFQ
    2. Collect bids in parallel (asyncio.gather)
    3. Policy validation
    4. Multi-criteria scoring + shortlist (Sprint 2)
    5. Award / Reject
    """

    def __init__(
        self,
        policy_engine: PolicyEngine,
        evaluator: MultiCriteriaEvaluator = None,
        ledger: Any = None,  # Will wire in Sprint 4
    ):
        self.policy = policy_engine
        self.evaluator = evaluator
        self.ledger = ledger

    async def run_sealed_bid_auction(
        self,
        session_id: str,
        rfq: RFQPayload,
        suppliers: List[SupplierAgent],
        policy_context: PolicyContext,
        market_spot_price: float,
        timeout_sec: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Execute full 1×N sealed bid auction.
        Returns dict with: session_id, fsm_state, valid_bids, rejections, winner, awards,
        scored_bids, shortlist.
        """
        fsm = GlobalAuctionFSM(session_id)
        
        # Phase 1: RFQ Broadcast
        if not fsm.transition(GlobalAuctionState.RFQ_BROADCAST):
            raise RuntimeError("FSM failed to transition to RFQ_BROADCAST")
        
        # Phase 2: Bid Collection (PARALLEL)
        if not fsm.transition(GlobalAuctionState.BID_COLLECTION):
            raise RuntimeError("FSM failed to transition to BID_COLLECTION")
        
        tasks = [
            asyncio.wait_for(
                s.respond_to_rfq(rfq),
                timeout=timeout_sec
            )
            for s in suppliers
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        bids: List[CNPMessage] = []
        failures = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failures.append({
                    "supplier_id": suppliers[i].supplier_id,
                    "error": str(result),
                })
            else:
                bids.append(result)
        
        # Phase 3: Evaluation
        if not fsm.transition(GlobalAuctionState.EVALUATION):
            raise RuntimeError("FSM failed to transition to EVALUATION")
        
        # Extract bid payloads
        bid_messages = [b for b in bids if b.type == MessageType.BID]
        reject_messages = [b for b in bids if b.type == MessageType.REJECT_BID]
        
        # Policy validation
        valid_bids = []
        policy_rejections = []
        
        for bid_msg in bid_messages:
            bid_data = bid_msg.payload
            passed, reason = self.policy.evaluate_bid(bid_data, policy_context)
            if passed:
                valid_bids.append(BidPayload(**bid_data))
            else:
                policy_rejections.append({
                    "supplier_id": bid_data.get("supplier_id"),
                    "reason": reason,
                })
        
        # Sprint 2: Multi-criteria scoring + shortlist
        if self.evaluator is None:
            evaluator = MultiCriteriaEvaluator(
                weights=EvaluationWeights(),
                esg_baselines=settings.evaluation.esg_baselines,
            )
        else:
            evaluator = self.evaluator

        ranked = evaluator.rank_bids(
            valid_bids,
            market_spot_price=market_spot_price,
            target_lead_time=rfq.target_lead_time_days,
            material=rfq.material,
        )

        ranked_bids = [bid for _, bid in ranked]
        shortlist_size = settings.evaluation.shortlist_size
        shortlist = ranked[:shortlist_size]   # List[(score, BidPayload)]
        losers = ranked[shortlist_size:]      # List[(score, BidPayload)]

        # Phase 4: Award or Terminate
        if ranked_bids:
            winner = ranked_bids[0]
            if not fsm.transition(GlobalAuctionState.AWARDED):
                raise RuntimeError("FSM failed to transition to AWARDED")
            
            award = CNPMessage.from_payload(
                MessageType.AWARD,
                {
                    "session_id": session_id,
                    "supplier_id": winner.supplier_id,
                    "unit_price": winner.unit_price,
                    "quantity": rfq.quantity,
                    "delivery_date": winner.delivery_date,
                    "payment_terms": rfq.payment_terms,
                }
            )
            
            # Rejections for non-winners
            loser_rejects = []
            for _, bid in ranked[1:]:
                loser_rejects.append(CNPMessage.from_payload(
                    MessageType.REJECT_BID,
                    {
                        "session_id": session_id,
                        "supplier_id": bid.supplier_id,
                        "reason": "OUTBID",
                    }
                ))
            
            return {
                "session_id": session_id,
                "fsm_state": fsm.state.name,
                "winner": winner.model_dump(),
                "award": award.model_dump(),
                "valid_bids": [b.model_dump() for b in ranked_bids],
                "rejections": policy_rejections + [r.payload for r in reject_messages] + [r.payload for r in loser_rejects],
                "failures": failures,
                "scored_bids": [
                    {
                        "supplier_id": bid.supplier_id,
                        "unit_price": bid.unit_price,
                        "lead_time_days": bid.lead_time_days,
                        "carbon_footprint_kg": bid.carbon_footprint_kg,
                        "reliability_score": bid.reliability_score,
                        "composite_score": score,
                    }
                    for score, bid in ranked
                ],
                "shortlist": [
                    {
                        "supplier_id": bid.supplier_id,
                        "composite_score": score,
                    }
                    for score, bid in shortlist
                ],
                "success": True,
            }
        else:
            if not fsm.transition(GlobalAuctionState.TERMINATED):
                raise RuntimeError("FSM failed to transition to TERMINATED")
            
            return {
                "session_id": session_id,
                "fsm_state": fsm.state.name,
                "winner": None,
                "valid_bids": [],
                "rejections": policy_rejections + [r.payload for r in reject_messages],
                "failures": failures,
                "scored_bids": [],
                "shortlist": [],
                "success": False,
            }
