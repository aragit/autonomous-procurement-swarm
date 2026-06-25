# 🤖 Autonomous Procurement Swarm

<p align="center"><b>LLM-Powered Multi-Agent Contract Negotiation for Supply Chain Optimization</b></p>

<p align="center"><sub>Ray · Transformers · vLLM · Multi-Agent RL</sub></p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-📐%20Active%20Blueprint-blue" alt="Active Blueprint">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Tests-16%20passing-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT">
</p>

---


## 🎯 Problem Statement

Procurement teams waste millions on suboptimal contract negotiations. Buyers and sellers operate in information asymmetry — neither knows the other's true reservation price, inventory constraints, or risk tolerance. Traditional e-procurement platforms are rigid rule-based systems that cannot adapt to volatile markets, geopolitical shocks, or complex multi-variable tradeoffs.

**Autonomous Procurement Swarm** deploys LLM-powered agents that negotiate procurement contracts autonomously, incorporating live market pricing, geopolitical risk scores, and inventory constraints into multi-turn strategic bargaining.

---

## 🏗️ System Architecture

**Agent Layer:** Buyer Agent, Seller Agent, Market Intelligence Agent, Arbiter Agent — each powered by role-specific LLM system prompts with structured JSON output.

**Orchestration Layer:** Turn-based negotiation protocol with Ray actor framework, shared context window, and immutable contract ledger.

**Market Layer:** Stochastic price simulation (Geometric Brownian Motion with regime-switching), geopolitical risk Markov chain, Poisson supply shocks.

**Validation Layer:** Protocol validation, hash-chained audit trail, Pareto efficiency analysis, reward computation.

---

## 🔬 Core Components

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| **LLM Engine** | `core/llm_engine.py` | Functional | MockLLM + Transformers + vLLM backends |
| **Prompts** | `core/prompts.py` | Functional | Role-specific system prompts (buyer/seller/market/arbiter) |
| **Protocol** | `core/negotiation_protocol.py` | Functional | JSON DSL for offers/counters/accepts/rejects |
| **Market Sim** | `core/market_simulator.py` | Functional | GBM prices, geopolitical risk, supply shocks |
| **Agents** | `core/agents.py` | Functional | Buyer/Seller/Market/Arbiter agent classes |
| **Ledger** | `core/contract_ledger.py` | Functional | Immutable hash-chained negotiation log |
| **Rewards** | `core/rewards.py` | Functional | Buyer cost minimization, seller margin maximization |
| **Orchestration** | `orchestration/negotiation_env.py` | Functional | Episode runner with turn management |

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/aragit/autonomous-procurement-swarm.git
cd autonomous-procurement-swarm
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Instant Demo (MockLLM — No Download)
``` bash
python scripts/run_negotiation.py --material steel --max-turns 4
```

Runs in under 5 seconds with deterministic but realistic negotiation behavior.

Expected Output:
```text
======================================================================
🤖 AUTONOMOUS PROCUREMENT SWARM — Active Blueprint
   LLM-Powered Multi-Agent Contract Negotiation
======================================================================

[1] Initializing MockLLM (deterministic, instant)...
[LLM] Using MockLLM — deterministic, instant, no download
   Backend: MockLLMEngine

[2] Initializing market simulator...

[3] Creating agents...

[4] Starting negotiation episode...

============================================================
🤝 NEGOTIATION EPISODE 8e64ebd2
   Buyer: AlphaCorp_Buyer | Seller: BetaSteel_Seller
   Material: steel | Spot: $452.20
============================================================

--- Turn 1/4 ---
   🛒 AlphaCorp_Buyer (buyer): COUNTER
      Reason: Market oversupply and 0% geopolitical risk warrants discount

--- Turn 2/4 ---
   🏭 BetaSteel_Seller (seller): OFFER
      Price: $497.42 | Qty: 1000

--- Turn 3/4 ---
   🛒 AlphaCorp_Buyer (buyer): ACCEPT

============================================================
✅ DEAL CLOSED
   Price: $416.02 | Qty: 1000
   Buyer reward: -9.3770
   Seller reward: 7.9224
============================================================
```

### 3. Real LLM (Optional — Phi-3-mini, ~2GB Download)
``` bash
# Phi-3-mini: open license, no HF auth needed
python scripts/run_negotiation.py --real-llm --model microsoft/Phi-3-mini-4k-instruct --material steel --max-turns 4
```

For gated models (Qwen2.5-3B), set your HuggingFace token:
```bash
export HF_TOKEN=your_token_here
python scripts/run_negotiation.py --real-llm --model Qwen/Qwen2.5-3B-Instruct --hf-token $HF_TOKEN
```
Note: Real LLM mode downloads 2-6GB on first run and requires ~8GB RAM.

### 4. Run Tests
```bash
pytest tests/ -v
```
All 16 tests pass — protocol validation, market simulation, ledger integrity.

