# Autonomous Procurement Swarm

**Agentic e-Procurement Auction Platform**

A production-grade, asynchronous multi-agent system for autonomous procurement negotiations. Executes 1×N sealed-bid reverse auctions with multi-criteria scoring, bilateral bartering, and immutable audit trails.

## What This Is

- **Reverse Auction Engine**: One buyer broadcasts an RFQ to N suppliers who bid in parallel
- **Multi-Criteria Scoring**: Awards based on Price (40%) + Lead Time (25%) + ESG/Carbon (20%) + Reliability (15%)
- **Bilateral Bartering**: Top-K shortlisted suppliers enter concurrent negotiation threads with formal FSMs
- **Policy Enforcement**: Deterministic rule engine validates spend caps, blacklists, ESG limits, and bid bonds
- **Immutable Ledger**: SHA-256 hash-chained events in PostgreSQL with chain verification
- **Semantic Memory**: pgvector-backed supplier profiles with concession slope tracking
- **FastAPI Control Plane**: HTTP endpoints for auction execution, audit, and analytics

## What This Is NOT

- ❌ Not a reinforcement learning system (no policy gradients, no training loop)
- ❌ Not a general supply chain optimizer (no inventory forecasting, no logistics routing)
- ❌ Not a legally binding contract generator (generates structured data, not legal documents)
- ❌ Not connected to live market data (uses synthetic GBM simulation)

## Architecture

```
[Buyer Orchestrator]
│
▼ RFQ Broadcast (asyncio.gather)
[Supplier Pool: 1..N]
│
▼ Bids
[Policy Engine] ──► [Multi-Criteria Evaluator]
│
▼ Shortlist Top-K
[Bilateral Threads: Buyer ↔ Supplier (per-pair FSM)]
│
▼ Best Deal
[PostgreSQL Ledger] + [pgvector Memory]
```

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/aragit/autonomous-procurement-swarm.git
cd autonomous-procurement-swarm
pip install -e .

# 2. Start PostgreSQL with pgvector
docker compose up -d

# 3. Run the CLI demo
python scripts/run_cnp_auction.py

# 4. Or start the API
uvicorn api.main:app --reload --port 8000

# 5. Create an auction via API
curl -X POST http://localhost:8000/auctions \
  -H "Content-Type: application/json" \
  -d '{"material":"steel","quantity":1000,"supplier_count":5,"enable_bartering":true}'
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service and database status |
| `/auctions` | POST | Start new sealed-bid auction |
| `/auctions/{id}` | GET | Retrieve auction events + chain validity |
| `/auctions/{id}/stats` | GET | Auction statistics |
| `/ledger/stats` | GET | Global ledger statistics |
| `/suppliers` | GET | List all suppliers with memory profiles |
| `/suppliers/{id}/profile` | GET | Supplier heuristic + vector profile |
| `/suppliers/similar` | GET | Find behaviorally similar suppliers |

## Testing

```bash
# Unit + integration tests (requires PostgreSQL on port 5433)
pytest tests/ -v

# With coverage
pytest tests/ --cov=core --cov=api

# Load test
pytest tests/integration/test_load.py -v
```

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Async Runtime | asyncio |
| Web Framework | FastAPI |
| Database | PostgreSQL 16 + pgvector |
| ORM | SQLAlchemy 2.0 (async) |
| Vector Search | pgvector |
| Logging | structlog (JSON) |
| Validation | Pydantic v2 |
| Testing | pytest + pytest-asyncio + httpx |
| Linting | ruff |
| Type Checking | mypy (strict) |
| Packaging | hatchling |

## Project Structure

```
autonomous-procurement-swarm/
├── api/                    # FastAPI application
├── core/
│   ├── agents/             # Buyer orchestrator, Supplier agents
│   ├── evaluator/          # Multi-criteria scoring engine
│   ├── engine/             # LLM backends (Mock, Transformers, vLLM)
│   ├── ledger/             # PostgreSQL hash-chained repository
│   ├── memory/             # Heuristic estimator + pgvector store
│   ├── protocol/           # CNP schemas, FSM, policy engine, auction orchestrator
│   └── simulator/          # Stochastic market simulation (GBM)
├── configs/                # Pydantic-settings + YAML
├── policy/                 # Rego policies (optional enterprise extension)
├── scripts/                # CLI demos
├── tests/
│   ├── unit/               # Protocol, scoring, memory, FSM tests
│   └── integration/        # API + load tests
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## License

MIT
