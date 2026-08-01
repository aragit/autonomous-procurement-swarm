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
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  </p>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [CLI Usage](#cli-usage)
- [Testing](#testing)
- [Deployment](#deployment)
- [Performance](#performance)
- [Security & Compliance](#security--compliance)
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

### Kubernetes

No Helm chart yet. Key resources needed:
- Deployment with 2+ replicas for API
- PostgreSQL StatefulSet with `pgvector/pgvector:16` image
- ConfigMap for `base.yaml`
- Secret for `DATABASE_URL`

---

## Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Auction throughput | 15-30 req/s | MockLLM, 5 suppliers, bartering enabled |
| p50 latency | ~150ms | End-to-end POST /auctions |
| p99 latency | ~600ms | Under concurrent load |
| Policy evaluation | <1ms | Deterministic Python, no LLM |
| Ledger append | ~5ms | Asyncpg + local PostgreSQL |
| Hash chain verify | O(n) | 1000 events ≈ 50ms |

### Bottlenecks

1. **LLM inference** — Real Transformers/vLLM backends add 2-10s per bid. Use MockLLM for load testing.
2. **Bilateral bartering** — Sequential per-supplier turns; K suppliers × 4 turns = 4K sequential LLM calls.
3. **PostgreSQL writes** — Append-only is fast, but chain verification is O(n). Do not verify on every read.

---

## Security & Compliance

### Cryptographic Ledger
- Full SHA-256 (64 hex chars), not truncated
- Parent hash linking prevents tampering without invalidating the chain
- `verify_chain()` walks the entire history and recomputes every digest

### Input Validation
- Pydantic v2 schemas enforce type safety and business rules
- Unknown materials return HTTP 422
- Zero/negative quantities return HTTP 422
- Payment terms whitelist: `net_30`, `net_60`, `cod`, `letter_of_credit`

### Policy Enforcement
- Spend caps are hard limits — no LLM can override
- Blacklisted vendors are rejected before scoring
- ESG carbon limits are enforced per material baseline

### What Is NOT Secured (Yet)
- 🔓 No authentication on API endpoints
- 🔓 No rate limiting
- 🔓 No TLS termination (use a reverse proxy)
- 🔓 No RBAC for buyer organizations

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
