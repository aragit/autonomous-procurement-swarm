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
