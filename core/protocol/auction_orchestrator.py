"""Orchestrates parallel sealed-bid collection from N suppliers."""

import asyncio
from typing import Any

from configs.settings import settings
from core.agents.buyer import BuyerOrchestrator
from core.agents.supplier import SupplierAgent
from core.evaluator.scoring import EvaluationWeights, MultiCriteriaEvaluator
from core.ledger.repository import PostgresLedgerRepository
from core.memory.heuristics import HeuristicReservationEstimator
from core.memory.semantic import PgVectorMemoryStore
from core.protocol.fsm import GlobalAuctionFSM, GlobalAuctionState
from core.protocol.policy_engine import PolicyContext, PolicyEngine
from core.protocol.schema import BidPayload, CNPMessage, MessageType, RFQPayload


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
        evaluator: MultiCriteriaEvaluator | None = None,
        ledger: PostgresLedgerRepository | None = None,
        memory: HeuristicReservationEstimator | None = None,
        vector_store: PgVectorMemoryStore | None = None,
    ) -> None:
        self.policy = policy_engine
        self.evaluator = evaluator
        self.ledger = ledger
        self.memory = memory or HeuristicReservationEstimator()
        self.vector_store = vector_store
        self._supplier_registry: dict[str, SupplierAgent] = {}
        self._bid_registry: dict[str, BidPayload] = {}

    async def run_sealed_bid_auction(
        self,
        session_id: str,
        rfq: RFQPayload,
        suppliers: list[SupplierAgent],
        policy_context: PolicyContext,
        market_spot_price: float,
        timeout_sec: float = 30.0,
    ) -> dict[str, Any]:
        """
        Execute full 1×N sealed bid auction.
        Returns dict with: session_id, fsm_state, valid_bids, rejections, winner, awards,
        scored_bids, shortlist.
        """
        fsm = GlobalAuctionFSM(session_id)
        self._supplier_registry = {s.supplier_id: s for s in suppliers}

        # Phase 1: RFQ Broadcast
        if not fsm.transition(GlobalAuctionState.RFQ_BROADCAST):
            raise RuntimeError("FSM failed to transition to RFQ_BROADCAST")

        if self.ledger:
            await self.ledger.append_event(
                session_id=session_id,
                turn=0,
                sender="buyer",
                message_type="rfq",
                payload=rfq.model_dump(),
            )

        # Phase 2: Bid Collection (PARALLEL)
        if not fsm.transition(GlobalAuctionState.BID_COLLECTION):
            raise RuntimeError("FSM failed to transition to BID_COLLECTION")

        tasks = [asyncio.wait_for(s.respond_to_rfq(rfq), timeout=timeout_sec) for s in suppliers]

        results: list[Any] = await asyncio.gather(*tasks, return_exceptions=True)

        bids: list[CNPMessage] = []
        failures: list[dict[str, str]] = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failures.append(
                    {
                        "supplier_id": suppliers[i].supplier_id,
                        "error": str(result),
                    }
                )
            else:
                bids.append(result)
                if self.ledger and result.type == MessageType.BID:
                    await self.ledger.append_event(
                        session_id=session_id,
                        turn=i + 1,
                        sender=result.payload.get("supplier_id", "unknown"),
                        message_type="bid",
                        payload=result.payload,
                    )

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
                policy_rejections.append(
                    {
                        "supplier_id": bid_data.get("supplier_id"),
                        "reason": reason,
                    }
                )

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
            quantity=rfq.quantity,
        )

        ranked_bids = [bid for _, bid in ranked]
        self._bid_registry = {bid.supplier_id: bid for bid in ranked_bids}
        shortlist_size = settings.evaluation.shortlist_size
        shortlist = ranked[:shortlist_size]  # List[(score, BidPayload)]

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
                },
            )

            # Rejections for non-winners
            loser_rejects = []
            for _, bid in ranked[1:]:
                loser_rejects.append(
                    CNPMessage.from_payload(
                        MessageType.REJECT_BID,
                        {
                            "session_id": session_id,
                            "supplier_id": bid.supplier_id,
                            "reason": "OUTBID",
                        },
                    )
                )

            # Persist award to ledger
            if self.ledger:
                award_payload = award.model_dump().get("payload", award.model_dump())
                await self.ledger.append_event(
                    session_id=session_id,
                    turn=999,
                    sender="orchestrator",
                    message_type="award",
                    payload=award_payload,
                )

            return {
                "session_id": session_id,
                "fsm_state": fsm.state.name,
                "winner": winner.model_dump(),
                "award": award.model_dump(),
                "valid_bids": [b.model_dump() for b in ranked_bids],
                "rejections": (
                    policy_rejections
                    + [r.payload for r in reject_messages]
                    + [r.payload for r in loser_rejects]
                ),
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

    async def run_bilateral_bartering(
        self,
        session_id: str,
        buyer: BuyerOrchestrator,
        shortlist: list[Any],  # List[(score, BidPayload)] or List[dict]
        rfq: RFQPayload,
        max_turns: int = 4,
        market_spot_price: float | None = None,
    ) -> dict[str, Any]:
        """
        Run parallel bilateral threads with shortlisted suppliers.
        Awards the supplier with the lowest successful final price.
        """
        fsm = GlobalAuctionFSM(session_id)
        # Fast-forward FSM to bartering state (assumes auction completed)
        fsm.state = GlobalAuctionState.SHORTLIST_BARTER

        threads = []
        initial_prices = []
        supplier_ids = []

        for item in shortlist:
            if isinstance(item, dict):
                supplier_id = item["supplier_id"]
                bid = self._bid_registry.get(supplier_id)
            else:
                _, bid = item
                supplier_id = bid.supplier_id
            supplier = self._supplier_registry.get(supplier_id)
            if supplier and bid:
                threads.append(
                    buyer.barter_with_supplier(
                        supplier,
                        rfq,
                        bid.unit_price,
                        max_turns,
                        memory=self.memory,
                        market_spot_price=market_spot_price,
                    )
                )
                initial_prices.append(bid.unit_price)
                supplier_ids.append(supplier_id)

        if not threads:
            return {
                "success": False,
                "reason": "NO_VALID_SHORTLIST",
                "best_deal": None,
                "all_threads": [],
            }

        results: list[Any] = await asyncio.gather(*threads, return_exceptions=True)

        # Find best successful deal (lowest final price)
        best_deal = None
        best_price = float("inf")

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                continue
            if result.get("success") and result.get("final_price", float("inf")) < best_price:
                best_price = result["final_price"]
                best_deal = {
                    "supplier_id": result["supplier_id"],
                    "final_price": result["final_price"],
                    "original_bid_price": round(initial_prices[i], 2),
                    "savings_vs_bid": round(initial_prices[i] - result["final_price"], 2),
                    "turns": result["turns"],
                    "history": result["history"],
                }

        # Record outcomes to memory for all threads (winners and losers)
        if market_spot_price is not None:
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    continue

                supplier_id = supplier_ids[i]
                final_price = result.get("final_price")
                won = best_deal is not None and best_deal.get("supplier_id") == supplier_id

                self.memory.record_auction_result(
                    supplier_id=supplier_id,
                    original_bid_price=initial_prices[i],
                    final_price=final_price,
                    spot_price=market_spot_price,
                    turns_taken=result.get("turns", 0),
                    won=won,
                )

                if self.vector_store:
                    profile = self.memory.get_profile(supplier_id)
                    if profile:
                        await self.vector_store.index_supplier(supplier_id, profile.to_dict())

        if best_deal:
            fsm.transition(GlobalAuctionState.AWARDED)

            # Persist bilateral award to ledger
            if self.ledger:
                await self.ledger.append_event(
                    session_id=session_id,
                    turn=1000,
                    sender="orchestrator",
                    message_type="award",
                    payload=best_deal,
                )

            return {
                "session_id": session_id,
                "fsm_state": fsm.state.name,
                "success": True,
                "best_deal": best_deal,
                "all_threads": results,
            }
        else:
            fsm.transition(GlobalAuctionState.TERMINATED)
            return {
                "session_id": session_id,
                "fsm_state": fsm.state.name,
                "success": False,
                "best_deal": None,
                "all_threads": results,
            }
