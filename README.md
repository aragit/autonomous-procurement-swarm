<p align="center">
  <h1 align="center">Autonomous Procurement Swarm</h1>
  <p align="center">
    <b>Agentic e-Procurement Auction Platform</b><br>
    Sealed-bid reverse auctions · Multi-criteria scoring · Bilateral bartering · Cryptographic audit trails
  </p>
  <p align="center">
    <a href="https://github.com/aragit/autonomous-procurement-swarm/actions/workflows/ci.yml">
      <img src="https://github.com/aragit/autonomous-procurement-swarm/actions/workflows/ci.yml/badge.svg" alt="CI">
    </a>
    <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
    <img src="https://img.shields.io/badge/fastapi-0.110%2B-009688" alt="FastAPI">
    <img src="https://img.shields.io/badge/postgres-16-336791" alt="PostgreSQL 16">
  </p>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Swarm Architecture](#swarm-architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [CLI Usage](#cli-usage)
- [Testing](#testing)
- [Deployment](#deployment)
- [Performance](#performance)
- [Security](#security)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**Autonomous Procurement Swarm** is a production-grade, asynchronous multi-agent system for autonomous procurement negotiations. It executes **1×N sealed-bid reverse auctions** where a buyer broadcasts a Request for Quote (RFQ) to N heterogeneous suppliers, evaluates bids across multiple business criteria, and enters concurrent bilateral bartering threads with shortlisted candidates.

Built for **determinism, observability, and auditability**, every state transition is governed by formal finite state machines, every message is validated by a deterministic policy engine, and every event is cryptographically chained in PostgreSQL.

### Design Philosophy

| Principle | Implementation |
|-----------|---------------|
| **Determinism over Stochasticity** | Compliance, scoring, and state transitions are pure code — LLMs only generate strategic bids |
| **Observability by Default** | Structured JSON logging (structlog), OpenAPI docs, ledger introspection endpoints |
| **Auditability as a Feature** | SHA-256 hash-chained append-only ledger with cryptographic verification |
| **Extensibility via Composition** | New scoring criteria, policy rules, and LLM backends are plug-and-play |

---

## Key Features

### Auction Engine
- **Contract Net Protocol (CNP)** — 1×N sealed-bid reverse auction with `asyncio.gather` parallel bid collection
- **Timeout-Resilient** — Suppliers that fail to respond within the deadline are gracefully excluded
- **Heterogeneous Suppliers** — Each supplier has a unique cost model (miner, distributor, recycler, trader)

### Multi-Criteria Scoring
- **Price (40%)** — Normalized against market spot price
- **Lead Time (25%)** — Penalty for delivery beyond target window
- **ESG / Carbon Footprint (20%)** — Material-specific baselines (kg CO2e per unit)
- **Supplier Reliability (15%)** — Historical performance score

### Bilateral Bartering
- **Top-K Shortlist** — Highest-scoring suppliers enter concurrent negotiation threads
- **Per-Pair FSM** — Formal state machines enforce legal transitions (no `ACCEPT` without prior `OFFER`)
- **Concession Tracking** — Buyer opens at 92% of bid, suppliers counter based on floor price + margin

### Policy & Governance
- **Deterministic Rule Engine** — Sub-millisecond validation of spend caps, blacklists, ESG limits, and bid bonds
- **Zero LLM in Compliance Loop** — Policy decisions are reproducible and explainable

### Immutable Ledger
- **SHA-256 Hash Chain** — Full 64-character digests, parent-linked
- **PostgreSQL Backend** — Asyncpg + SQLAlchemy 2.0 with ACID guarantees
- **Chain Verification** — `GET /auctions/{id}` returns `chain_valid: true/false`

### Semantic Memory
- **pgvector Integration** — Supplier behavioral embeddings stored in the same database as the ledger
- **Heuristic Reservation Estimator** — Tracks concession slopes, classifies suppliers as fast/medium/slow conceders
- **Similarity Search** — Find suppliers with comparable negotiation behavior

---

## Architecture

```mermaid
flowchart TB
    subgraph AuctionPhase["Sealed-Bid Auction Phase"]
        B[Buyer Orchestrator] -->|RFQ Broadcast| S1[Supplier A]
        B -->|RFQ Broadcast| S2[Supplier B]
        B -->|RFQ Broadcast| S3[Supplier C]
        S1 -->|Bid| PE[Policy Engine]
        S2 -->|Bid| PE
        S3 -->|Bid| PE
        PE -->|Valid Bids| MCE[Multi-Criteria Evaluator]
    end

    subgraph BarteringPhase["Bilateral Bartering Phase"]
        MCE -->|Shortlist Top-K| FSM1[Bilateral FSM: Buyer vs A]
        MCE -->|Shortlist Top-K| FSM2[Bilateral FSM: Buyer vs B]
        FSM1 -->|Best Deal| AWARD[Award Engine]
        FSM2 -->|Best Deal| AWARD
    end

    subgraph Persistence["Persistence Layer"]
        AWARD -->|Append-Only Events| PG[(PostgreSQL + pgvector)]
        PG -->|Hash Chain| VERIFY[Chain Verification]
        PG -->|Vector Search| SIM[Similarity Queries]
    end

    subgraph API["FastAPI Control Plane"]
        HEALTH[health]
        AUCTIONS[auctions]
        LEDGER[ledger-stats]
        SUPPLIERS[suppliers]
    end

    AuctionPhase --> BarteringPhase
    BarteringPhase --> Persistence
    Persistence --> API
```

### State Machines

**Global Auction FSM**
```
INIT → RFQ_BROADCAST → BID_COLLECTION → EVALUATION → SHORTLIST_BARTER → AWARDED
                                                          ↓
                                                    TERMINATED
```

**Bilateral FSM (per supplier)**
```
INIT → OFFER_RECEIVED → COUNTER_SENT → OFFER_RECEIVED → ACCEPTED
              ↓                ↓
            REJECTED        TIMEOUT
```

---

## Swarm Architecture

The repository also ships a self-contained **swarm runtime** (`swarm/`) as the
foundation for autonomous multi-agent procurement. It is intentionally
deterministic and rule-based — no planning, no LLM, no autonomous learning. A
**domain layer** (`swarm/domain/`) adapts the existing `core/` procurement logic
(RFQ normalization, market simulation, multi-criteria scoring, the policy
engine) into domain agents that cooperate over the event bus.

The current flow is **parallel, per-supplier, and strategy-driven** (Phases 3–6):
discovery announces each supplier individually, evaluation and negotiation run
one specialized step per supplier, a completion tracker gates the decision
until every expected evaluation and quote artifact exists, the strategy agent
picks a deterministic scoring strategy before any supplier is discovered, and the
outcome + supplier-intelligence agents close a deterministic feedback loop that
improves future evaluations. After the decision, a deterministic **governance
tail** (risk → governance → approval) gates whether the award is safe and
authorized to execute — all through artifacts and events on a single
`correlation_id`.

```mermaid
flowchart LR
    USER[User Request] --> SWARM[Swarm facade]
    SWARM -->|CreateRequirement message| RA[RequirementAgent]
    RA -->|RequirementCreated event| BUS[Event Bus]
    BUS --> ST[StrategyAgent]
    ST -->|StrategySelected event| SDA[SupplierDiscoveryAgent]
    SDA -->|SupplierDiscovered x5| EAV[EvaluationAgent]
    EAV -->|SupplierEvaluated x5| NA[NegotiationAgent]
    NA -->|QuoteGenerated x5| BUS
    EAV --> CT[CompletionTracker]
    NA --> CT
    CT -->|QuotesCompleted| DA[DecisionAgent]
    DA -->|DecisionMade| RSA[RiskAssessmentAgent]
    RSA -->|RiskAssessmentCompleted| GA[GovernanceAgent]
    GA -->|GovernanceDecisionMade| APA[ApprovalAgent]
    APA -->|ApprovalGranted / ApprovalRequired / ApprovalRejected| AUTH[(ExecutionAuthorization)]
    USER -->|ApproveDecision (POST /approve)| APA
    STATE-.->|OutcomeRecorded message| OA[OutcomeAgent]
    OA -->|OutcomeRecorded event| SIA[SupplierIntelligenceAgent]
    SIA -->|SupplierPerformanceUpdated| SM[(SupplierMemoryStore)]
    SM -.->|history| EAV
    RA --> STATE
    SDA --> STATE
    EAV --> STATE
    NA --> STATE
    DA --> STATE
    RSA --> STATE
    GA --> STATE
    APA --> STATE
    STATE[(Shared SwarmState)]
```

The four pillars:

| Pillar | Module | Responsibility |
|--------|--------|----------------|
| **Agents** | `swarm/core/agent.py` | `BaseAgent` with a `perceive → reason → act` lifecycle, unique name, capabilities, tags and status |
| **Events** | `swarm/core/event.py` | `Event` + async `EventBus`; agents communicate only by publishing/consuming events, never by holding references to each other |
| **Coordinator** | `swarm/core/__init__.py` | Public `Swarm` facade: register agents, route events, maintain shared state |
| **Shared State** | `swarm/core/state.py` | Serializable `SwarmState` of typed `Artifact`s (with `parent_ids` lineage), events, completion expectations and results that every agent can read and mutate |

A more detailed walkthrough — including sequence diagrams and the hardening
guarantees (`correlation_id`, event replay, capability schema, structured
logging) — lives in [docs/swarm-architecture.md](docs/swarm-architecture.md).

Phase 3–6 demo — one `CreateRequirement` message fans out through the deterministic
agents to a final supplier decision, then through the governance tail
(risk → governance → approval) to an execution authorization; optionally followed
by an outcome record that seeds the supplier performance memory:

```bash
python -m examples.procurement_swarm_demo
```

1. A `CreateRequirement` message reaches the swarm and the requirement agent
   normalizes it (RFQ defaults, market-derived price cap) into a requirement
   artifact, then announces `RequirementCreated`.
2. The strategy agent picks a deterministic scoring strategy (`cost_optimized`,
   `balanced` or `low_carbon`) from the requirement's constraints and announces
   `StrategySelected`.
3. The supplier discovery agent (gated on `StrategySelected`) samples the
   deterministic market (`MarketSimulator(seed=42)`), declares the expected
   evaluation/quote counts, publishes the five-supplier pool and announces one
   `SupplierDiscovered` event per supplier.
4. Each `SupplierDiscovered` is routed to the evaluation agent, which scores
   that supplier — blending the strategy weights with optional supplier-history
   from the `SupplierMemoryStore` — and announces `SupplierEvaluated`; the
   negotiation agent turns each evaluation into a deterministic
   `QuoteGenerated`. The `CompletionTracker` fires `EvaluationCompleted` /
   `QuotesCompleted` (once, idempotently) once every expected artifact for the
   conversation exists.
5. The decision agent reacts only to `QuotesCompleted`, filters the quotes
   through the policy engine and picks the winner, then emits a
   `DecisionExplanationArtifact` explaining *why*. All within one
   `correlation_id`.
6. The risk agent assesses the decision against supplier history, the purchase
   amount and carbon footprint → `RiskAssessmentCompleted`.
7. The governance agent applies the active `GovernancePolicy` →
   `GovernanceDecisionMade` (`APPROVED` / `APPROVAL_REQUIRED` / `REJECTED`).
8. The approval agent closes the gate into an `ExecutionAuthorizationArtifact`
   (`authorized` / `pending` / blocked). A pending authorization is resolved by
   `POST /swarm/{request_id}/approve` (deterministic simulated approval).
9. A `RecordProcurementOutcome` message records what actually happened; the
   outcome agent writes an `OutcomeArtifact` (lineaged to the `DecisionArtifact`)
   and the supplier-intelligence agent folds it into a deterministic
   `SupplierPerformanceArtifact` — visible to future evaluations.

The `Swarm` facade exposes the read-only execution trace:

```python
from swarm.domain.wiring import build_procurement_swarm

swarm = build_procurement_swarm(request_id="REQ-002", goal="Source aluminum")
await swarm.start()
await swarm.send_message("CreateRequirement", {...}, correlation_id="REQ-002-CONV")
await swarm.shutdown()

trace = swarm.get_execution_trace("REQ-002-CONV")  # events, artifacts, agent_actions
```

Phase 1 runtime demo (runtime primitives only):

```bash
python -m examples.run_swarm_demo
```

Tests: `tests/unit/test_agent.py`, `tests/unit/test_event.py`,
`tests/unit/test_registry.py`, `tests/unit/test_coordinator.py`,
`tests/unit/test_swarm_core.py`, `tests/unit/test_completion.py`,
`tests/unit/test_requirement_agent.py`, `tests/unit/test_supplier_agent.py`,
`tests/unit/test_evaluation_agent.py`, `tests/unit/test_negotiation_agent.py`,
`tests/unit/test_decision_agent.py`, `tests/unit/test_full_procurement_flow.py`,
`tests/unit/test_strategy_selection.py`,
`tests/unit/test_weighted_evaluation.py`,
`tests/unit/test_decision_explanation.py`,
`tests/unit/test_supplier_memory.py`, `tests/unit/test_outcome_agent.py`,
`tests/unit/test_supplier_intelligence.py`,
`tests/unit/test_supplier_history_evaluation.py`, and
`tests/integration/test_feedback_cycle.py`.

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- 2GB free RAM (for PostgreSQL + API)

### 1. Clone & Install

```bash
git clone https://github.com/aragit/autonomous-procurement-swarm.git
cd autonomous-procurement-swarm
pip install -e .
```

### 2. Start Infrastructure

```bash
docker compose up -d
# PostgreSQL with pgvector will be available on localhost:5433
```

### 3. Run CLI Demo

```bash
python scripts/run_cnp_auction.py
```

Expected output:
```
🤖 AUTONOMOUS PROCUREMENT SWARM — CNP Sealed Bid Auction
...
🏆 WINNER: MinerCorp_A @ $276.94/unit
📊 COMPOSITE SCORE: 0.6609
💰 SAVINGS FROM BARTERING: $42.04/unit
```

### 4. Start API Server

```bash
uvicorn api.main:app --reload --port 8000
```

### 5. Create Your First Auction

```bash
curl -X POST http://localhost:8000/auctions \
  -H "Content-Type: application/json" \
  -d '{
    "material": "steel",
    "quantity": 1000,
    "supplier_count": 5,
    "enable_bartering": true
  }'
```

---

## Installation

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install base package (no PyTorch — uses MockLLM)
pip install -e .

# Install with real LLM support (downloads ~6GB)
pip install -e ".[llm]"

# Install with all dev tools
pip install -e ".[dev]"
```

### Docker (Production)

```bash
docker compose build
docker compose up -d
# API available at http://localhost:8000
```

---

## Configuration

All configuration is managed via `configs/base.yaml` and environment variables (override via `pydantic-settings`).

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PROCUREMENT_LLM__MODEL_NAME` | `Qwen/Qwen2.5-3B-Instruct` | HuggingFace model ID |
| `PROCUREMENT_LLM__PREFER_VLLM` | `false` | Use vLLM over Transformers |
| `DATABASE_URL` | `postgresql+asyncpg://procurement:procurement@localhost:5433/procurement` | PostgreSQL connection string |
| `PROCUREMENT_EVALUATION__SHORTLIST_SIZE` | `2` | Number of suppliers entering bartering |

### Base Configuration

```yaml
llm:
  model_name: "Qwen/Qwen2.5-3B-Instruct"
  prefer_vllm: false
  temperature: 0.7
  max_tokens: 512

market:
  seed: 42
  materials: ["steel", "aluminum", "copper", "plastic", "lumber", "rubber"]
  shock_lambda: 0.05

negotiation:
  max_turns: 6
  valid_materials: ["steel", "aluminum", "copper", "plastic", "lumber", "rubber"]
  valid_payment_terms: ["net_30", "net_60", "cod", "letter_of_credit"]

evaluation:
  esg_baselines:
    steel: 1800.0
    aluminum: 12000.0
    copper: 3000.0
    plastic: 2500.0
    lumber: 200.0
    rubber: 2800.0
  scoring_weights:
    price: 0.40
    lead_time: 0.25
    esg: 0.20
    reliability: 0.15
  shortlist_size: 2
```

---

## API Reference

Interactive docs available at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc` (ReDoc).

### `POST /auctions`

Start a new sealed-bid procurement auction.

**Request:**
```json
{
  "material": "steel",
  "quantity": 1000,
  "max_unit_price": 540.0,
  "target_lead_time_days": 30,
  "supplier_count": 5,
  "enable_bartering": true
}
```

**Response:**
```json
{
  "session_id": "a1b2c3d4",
  "status": "AWARDED",
  "winner": {
    "supplier_id": "MinerCorp_A",
    "unit_price": 276.94,
    "quantity": 1000,
    "delivery_date": "2026-08-15",
    "payment_terms": "net_30"
  },
  "final_price": 276.94,
  "scored_bids": [
    {
      "supplier_id": "MinerCorp_A",
      "unit_price": 276.94,
      "lead_time_days": 32,
      "carbon_footprint_kg": 1800000.0,
      "reliability_score": 0.85,
      "composite_score": 0.6609
    }
  ],
  "shortlist": [
    {"supplier_id": "MinerCorp_A", "composite_score": 0.6609},
    {"supplier_id": "DistribCorp_B", "composite_score": 0.5954}
  ],
  "bartering_result": {
    "best_deal": {
      "supplier_id": "MinerCorp_A",
      "final_price": 276.94,
      "original_bid_price": 318.98,
      "savings_vs_bid": 42.04,
      "turns": 8
    }
  }
}
```

### `GET /auctions/{session_id}`

Retrieve full auction events with ledger chain verification.

**Response:**
```json
{
  "session_id": "a1b2c3d4",
  "events": [
    {
      "id": 1,
      "turn_number": 0,
      "sender_id": "API_Buyer",
      "message_type": "rfq",
      "payload": { "material": "steel", "quantity": 1000 },
      "prev_hash": "0000000000000000000000000000000000000000000000000000000000000000",
      "current_hash": "a3f2...",
      "timestamp": "2026-08-01T12:00:00Z"
    }
  ],
  "chain_valid": true
}
```

### `GET /ledger/stats`

Global ledger statistics.

**Response:**
```json
{
  "total_events": 1247,
  "total_sessions": 89,
  "deals_awarded": 87
}
```

### `GET /suppliers/{supplier_id}/profile`

Retrieve learned supplier profile.

**Response:**
```json
{
  "heuristic_profile": {
    "supplier_id": "MinerCorp_A",
    "auctions_participated": 12,
    "auctions_won": 8,
    "avg_concession_slope": 15.4,
    "avg_margin_at_win": 0.1523,
    "concession_speed": "medium",
    "reliability_score": 0.85
  },
  "vector_metadata": {
    "concession_speed": "medium",
    "avg_margin_at_win": 0.1523,
    "reliability_score": 0.85
  }
}
```

### `GET /suppliers`

List all suppliers with accumulated memory profiles.

### `GET /suppliers/similar?supplier_id=MinerCorp_A&n=3`

Find behaviorally similar suppliers via pgvector.

**Response:**
```json
{
  "query_supplier": "MinerCorp_A",
  "matches": [
    {
      "supplier_id": "MinerCorp_A",
      "distance": 0.0
    },
    {
      "supplier_id": "RecycleCorp_C",
      "distance": 0.975
    }
  ]
}
```

### `POST /swarm/requirements`

Dispatch a requirement into the deterministic swarm (no LLM, no planner) and run
the full parallel flow to a decision. Pass an optional `max_carbon_per_unit` to
select the low-carbon strategy.

**Request:**
```json
{
  "material": "aluminum",
  "quantity": 1000,
  "budget": 2000000.0,
  "target_lead_time_days": 30,
  "max_carbon_per_unit": 800.0
}
```

**Response:**
```json
{
  "request_id": "REQ-3F9A2C1B",
  "correlation_id": "REQ-3F9A2C1B-CONV",
  "decision": {
    "selected_supplier": "MinerCorp_A",
    "ranked": [{"supplier_id": "MinerCorp_A", "score": 0.858, "price": 984.0}]
  },
  "explanation": {
    "strategy_used": "low_carbon",
    "top_factors": ["Selection followed the low_carbon strategy..."],
    "rejected_suppliers": [
      {"supplier_id": "DistribCorp_B", "reason": "Lower composite score than the selected supplier"}
    ]
  },
  "completions": {"REQ-3F9A2C1B-CONV": ["evaluation", "quote"]},
  "event_count": 31,
  "artifact_count": 12
}
```

### `POST /swarm/{request_id}/outcome`

Feed a post-decision outcome back into the deterministic in-memory supplier
memory. The `decision_id` lineage anchor is resolved from the remembered run.

**Request:**
```json
{
  "supplier_id": "MinerCorp_A",
  "delivered_on_time": true,
  "quality_score": 0.92,
  "actual_price": 984.0,
  "carbon_score": 1800.0
}
```

### `GET /swarm/explanation/{request_id}`

Human-readable decision explanation artifact (strategy used, top factors, and
why each other supplier was rejected).

### `GET /swarm/supplier/{supplier_id}/performance`

Deterministic supplier performance summary (order count, delivery/quality/price
competitiveness/carbon averages) from the in-memory `SupplierMemoryStore`.

**Response:**
```json
{
  "supplier_id": "MinerCorp_A",
  "performance": {
    "total_orders": 15,
    "successful_orders": 14,
    "delivery_score": 0.94,
    "quality_score": 0.91,
    "price_competitiveness": 0.99,
    "carbon_score": 1800.0
  }
}
```

### `GET /swarm/risk/{request_id}`

Deterministic risk assessment for the selected decision of a swarm run
(`financial` / `delivery` / `quality` / `carbon` sub-scores, `overall_risk_score`,
and `risk_level`: `LOW` / `MEDIUM` / `HIGH` / `CRITICAL`).

### `GET /swarm/governance/{request_id}`

Governance decision (`APPROVED` / `APPROVAL_REQUIRED` / `REJECTED`), the policy
applied, and the `required_approver` (when approval is required).

### `POST /swarm/{request_id}/approve`

Resolve a pending (`APPROVAL_REQUIRED`) execution authorization via the
deterministic simulated approval. Governance has already excluded rejected
decisions, so a pending authorization is granted.

**Request:**
```json
{ "approver": "governance_sim" }
```

### `GET /swarm/authorization/{request_id}`

Final execution authorization status (`authorized` / `pending` / `rejected`),
parented through the full control-layer lineage:

    Requirement → Strategy → SupplierList → Evaluation → Quote → Decision
      → RiskAssessment → GovernanceDecision → ExecutionAuthorization

Each link is by `Artifact.id`; the outcome-feedback loop (`POST /swarm/{id}/outcome`)
attaches alongside the decision for audit.

### `GET /swarm/trace/{request_id}`

Read-only execution trace (events, artifacts, agent actions) for one swarm run.

**Response:**
```json
{
  "correlation_id": "REQ-3F9A2C1B-CONV",
  "events": [
    {"type": "CreateRequirement", "source": "user", "correlation_id": "REQ-3F9A2C1B-CONV"},
    {"type": "SupplierEvaluated", "source": "evaluation_agent", "correlation_id": "REQ-3F9A2C1B-CONV"}
  ],
  "artifacts": [
    {"kind": "evaluation", "name": "evaluation_MinerCorp_A", "score": 0.858}
  ],
  "agent_actions": [
    {"agent": "evaluation_agent", "action": "artifact_created", "kind": "evaluation"},
    {"agent": "evaluation_agent", "action": "event_published", "event_type": "SupplierEvaluated"}
  ]
}
```

### `GET /swarm/trace/{request_id}/completions`

Completion groups closed per correlation id for a swarm run.

### `GET /swarm/state/{request_id}`

Full serialized, read-only snapshot of a swarm run's shared state.

---

## CLI Usage

### Run Sealed-Bid Auction Demo

```bash
python scripts/run_cnp_auction.py
```

This demonstrates the full CNP auction lifecycle: RFQ broadcast → parallel bid collection → policy validation → multi-criteria scoring → shortlist → bilateral bartering → hash-chained ledger write. Edit the script to change `material`, `quantity`, or supplier configs.

---

## Testing

### Unit & Integration Tests

Requires PostgreSQL running on port 5433.

```bash
# Full suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=core --cov=api --cov-report=html

# Specific modules
pytest tests/unit/test_scoring.py -v
pytest tests/unit/test_bilateral.py -v
pytest tests/unit/test_swarm_core.py tests/unit/test_coordinator.py -v
pytest tests/integration/test_api.py -v
pytest tests/integration/test_load.py -v
```

### Terminal Functional & Resilience Suite

```bash
chmod +x test_suite.sh
./test_suite.sh
```

Covers:
- 20 functional checks (health, auction lifecycle, chain integrity, multi-material, input validation)
- 13 resilience checks (concurrent auctions, bartering stress, DB persistence, chain integrity under load, memory accumulation)

### Load Testing

```bash
python stress_test.py
```

> **Note:** `stress_test.py` requires `pip install -e ".[dev]"` for the `httpx`/`h2` dependencies. Expected throughput on a modern laptop: **15-30 req/s** with p50 latency < 200ms.

---

## Deployment

### Docker Compose (Recommended)

```bash
docker compose build
docker compose up -d
# API available at http://localhost:8000
# PostgreSQL available at localhost:5433
```

Services:
- `postgres` — PostgreSQL 16 with pgvector (port 5433)
- `api` — FastAPI application (port 8000)

### Environment-Specific Overrides

Create `.env` in project root:

```bash
DATABASE_URL=postgresql+asyncpg://user:pass@prod-db:5432/procurement
PROCUREMENT_LLM__PREFER_VLLM=true
```

### Production Deployment Notes

Beyond Docker Compose, for production-scale deployments:

- **API**: Stateless FastAPI app — scale horizontally behind a load balancer
- **Database**: Use a managed PostgreSQL 16+ instance with the `pgvector` extension enabled (AWS RDS, Azure Database for PostgreSQL, Google Cloud SQL)
- **Configuration**: Mount `configs/base.yaml` via ConfigMap; inject `DATABASE_URL` via Secret
- **Observability**: structlog JSON output is compatible with Datadog, Splunk, or ELK stack ingestion
- **TLS**: Terminate TLS at the reverse proxy (nginx, Traefik, or cloud load balancer)

---

## Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **Auction throughput** | 15–30 req/s | End-to-end sealed-bid + bartering, 5 suppliers |
| **p50 latency** | ~150ms | MockLLM backend |
| **p99 latency** | ~600ms | Under 20-concurrent load |
| **Policy validation** | <1ms | Deterministic Python, zero LLM overhead |
| **Ledger append** | ~5ms | Asyncpg, append-only, no locks |
| **Hash chain verification** | O(n), ~50ms/1000 events | Cryptographic integrity check |

### Scaling Dimensions

| Bottleneck | Mitigation Strategy |
|------------|---------------------|
| LLM inference latency | Use vLLM with continuous batching; or pre-compute supplier strategies |
| Bilateral bartering serialization | Shard by `session_id`; each auction is independent |
| Ledger read amplification | Verify chain asynchronously (background job), not on every `GET` |

### Benchmarks

```bash
python stress_test.py
# Expected: 15-30 req/s on a 4-core laptop with local PostgreSQL
```

---

## Security

### Current Protections
- **Input validation**: Pydantic v2 schemas reject malformed requests (HTTP 422)
- **Material whitelist**: Only configured commodities are accepted
- **Spend caps**: Hard budget limits enforced by PolicyEngine (no LLM override)
- **Audit trail**: SHA-256 hash chain detects tampering

### Production Hardening Checklist
Before exposing to untrusted clients:
- [ ] Add OAuth2 / API key authentication (FastAPI `Depends` + `HTTPBearer`)
- [ ] Add rate limiting (`slowapi` or nginx `limit_req`)
- [ ] Terminate TLS at reverse proxy
- [ ] Add buyer organization isolation (multi-tenant `session_id` prefixing)

See [SECURITY.md](SECURITY.md) for responsible disclosure policy.

---

## Troubleshooting

### `Connection refused` to PostgreSQL
```bash
docker compose ps
# If postgres is unhealthy:
docker compose logs postgres
# Common fix: port 5433 is already in use
docker compose down
docker compose up -d
```

### `ModuleNotFoundError: No module named 'core'`
```bash
pip install -e .
# Do NOT use sys.path hacks. The package must be installed.
```

### `ImportError: cannot import name 'Vector' from 'pgvector'`
The `pgvector` Python package is required, but the PostgreSQL extension is separate. Ensure your Docker image is `pgvector/pgvector:16`, not plain `postgres:16`.

### Tests fail with `connection refused`
PostgreSQL must be running before tests:
```bash
docker compose up -d
pytest tests/ -v
```

### Slow Docker build
The production Dockerfile intentionally excludes PyTorch (the API uses MockLLM by default). If you need real LLM inference in the container:
```dockerfile
# In Dockerfile, change:
RUN pip install --no-cache-dir -e .
# To:
RUN pip install --no-cache-dir -e ".[llm]"
```
Expect 3-5 minute build time due to torch download (~2 GB).

---

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature/your-feature`
3. **Install dev dependencies**: `pip install -e ".[dev]"`
4. **Run quality checks**: `ruff check . && mypy core/ api/ --ignore-missing-imports`
5. **Run tests**: `pytest tests/ -v`
6. **Submit a Pull Request**

### Code Standards
- **Type hints**: All public functions must have return type annotations
- **Async**: All I/O-bound operations use `async`/`await`
- **Pydantic**: All data schemas use Pydantic v2
- **Logging**: Use `structlog`, never `print()`
- **Tests**: All new features require unit tests; API changes require integration tests


---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with determinism, auditability, and game theory in mind.
</p>
