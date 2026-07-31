#!/usr/bin/env python3
"""Demo script for 1×N sealed bid CNP auction."""

import asyncio
import uuid
from core.llm_engine import LLMEngineFactory
from core.agents.supplier import SupplierAgent, CostModel
from core.agents.buyer import BuyerOrchestrator
from core.protocol.auction_orchestrator import AuctionOrchestrator
from core.protocol.policy_engine import PolicyEngine, PolicyContext
from core.market_simulator import MarketSimulator

async def main():
    print("=" * 70)
    print("🤖 AUTONOMOUS PROCUREMENT SWARM — CNP Sealed Bid Auction")
    print("=" * 70)
    
    # Init
    llm = LLMEngineFactory.create(use_mock=True)
    market = MarketSimulator(seed=42)
    market_state = market.get_current_state("steel")
    
    # Create 5 heterogeneous suppliers
    suppliers = [
        SupplierAgent("MinerCorp_A", llm, CostModel(
            base_cost_per_unit=market_state.spot_price * 0.35,
            logistics_premium_per_unit=50.0,
            capacity_units=5000,
            current_utilization=0.3,
            min_margin_pct=0.20,
            reliability_score=0.85,
            esg_carbon_per_unit=1800.0,
        )),
        SupplierAgent("DistribCorp_B", llm, CostModel(
            base_cost_per_unit=market_state.spot_price * 0.75,
            logistics_premium_per_unit=20.0,
            capacity_units=10000,
            current_utilization=0.6,
            min_margin_pct=0.12,
            reliability_score=0.90,
            esg_carbon_per_unit=1200.0,
        )),
        SupplierAgent("RecycleCorp_C", llm, CostModel(
            base_cost_per_unit=market_state.spot_price * 0.80,
            logistics_premium_per_unit=30.0,
            capacity_units=3000,
            current_utilization=0.4,
            min_margin_pct=0.15,
            reliability_score=0.75,
            esg_carbon_per_unit=400.0,
        )),
        SupplierAgent("TraderCorp_D", llm, CostModel(
            base_cost_per_unit=market_state.spot_price * 0.90,
            logistics_premium_per_unit=10.0,
            capacity_units=8000,
            current_utilization=0.8,
            min_margin_pct=0.08,
            reliability_score=0.70,
            esg_carbon_per_unit=1500.0,
        )),
        SupplierAgent("PremiumSteel_E", llm, CostModel(
            base_cost_per_unit=market_state.spot_price * 0.50,
            logistics_premium_per_unit=80.0,
            capacity_units=2000,
            current_utilization=0.2,
            min_margin_pct=0.30,
            reliability_score=0.95,
            esg_carbon_per_unit=2000.0,
        )),
    ]
    
    # Buyer setup
    buyer = BuyerOrchestrator("AlphaCorp_Buyer", llm, PolicyEngine())
    session_id = str(uuid.uuid4())[:8]
    
    rfq = buyer.create_rfq(
        session_id=session_id,
        material="steel",
        quantity=1000,
        max_unit_price=market_state.spot_price * 1.2,
        target_lead_time_days=30,
        budget=500_000.0,
    )
    
    policy_ctx = PolicyContext(
        buyer_max_budget_total=500_000.0,
        blacklisted_vendors=set(),
        max_carbon_limit_kg=2_000_000.0,
        max_unit_price=rfq.max_unit_price,
        required_bid_bond_pct=0.05,
    )
    
    # Run auction
    orchestrator = AuctionOrchestrator(PolicyEngine())
    result = await orchestrator.run_sealed_bid_auction(
        session_id=session_id,
        rfq=rfq,
        suppliers=suppliers,
        policy_context=policy_ctx,
        market_spot_price=market_state.spot_price,
        timeout_sec=10.0,
    )
    
    # Report
    print(f"\n{'='*70}")
    print(f"AUCTION RESULT — Session: {result['session_id']}")
    print(f"FSM State: {result['fsm_state']}")
    print(f"Success: {'✅ YES' if result['success'] else '❌ NO'}")
    print(f"{'='*70}")
    
    if result['winner']:
        w = result['winner']
        print(f"\n🏆 WINNER: {w['supplier_id']}")
        print(f"   Price: ${w['unit_price']:,.2f}/unit")
        print(f"   Lead Time: {w['lead_time_days']} days")
        print(f"   Carbon: {w['carbon_footprint_kg']:,.0f} kg")
        print(f"   Reliability: {w['reliability_score']:.0%}")
    
    print(f"\n📋 ALL VALID BIDS ({len(result['valid_bids'])}):")
    for i, bid in enumerate(result['valid_bids'], 1):
        print(f"   {i}. {bid['supplier_id']}: ${bid['unit_price']:,.2f} | "
              f"Lead: {bid['lead_time_days']}d | Carbon: {bid['carbon_footprint_kg']:,.0f}kg")
    
    print(f"\n📊 COMPOSITE SCORE BREAKDOWN (Spot: ${market_state.spot_price:,.2f}):")
    for bid_info in result.get("scored_bids", []):
        print(f"   {bid_info['supplier_id']}: "
              f"Score={bid_info['composite_score']:.4f} | "
              f"Price=${bid_info['unit_price']:,.2f} | "
              f"Lead={bid_info['lead_time_days']}d | "
              f"Carbon={bid_info['carbon_footprint_kg']:,.0f}kg | "
              f"Rel={bid_info['reliability_score']:.0%}")
    
    if result.get("shortlist"):
        print(f"\n🔀 SHORTLIST (top-{len(result['shortlist'])} for Sprint 3 bartering):")
        for s in result["shortlist"]:
            print(f"   {s['supplier_id']} (score={s['composite_score']:.4f})")
    
    if result['rejections']:
        print(f"\n❌ REJECTIONS ({len(result['rejections'])}):")
        for r in result['rejections']:
            print(f"   {r.get('supplier_id', 'unknown')}: {r.get('reason', r.get('error', 'N/A'))}")
    
    if result['failures']:
        print(f"\n⚠️  FAILURES ({len(result['failures'])}):")
        for f in result['failures']:
            print(f"   {f['supplier_id']}: {f['error']}")
    
    print(f"\n{'='*70}")
    print("✅ CNP Auction complete.")
    print(f"{'='*70}")

if __name__ == "__main__":
    asyncio.run(main())
