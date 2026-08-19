<p align="center">
  <h1 align="center">Autonomous Procurement Swarm</h1>
  <p align="center">
    <b>Distributed Neuro-Symbolic Multi-Agent Mesh · Ray Actor Infrastructure · OPA/Rego Policy-as-Code · LinUCB Contextual Bandits</b>
  </p>
  <p align="center">
    <img src="https://github.com/aragit/autonomous-procurement-swarm/actions/workflows/ci.yml/badge.svg" alt="CI">
    <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python: 3.11+">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT">
    <img src="https://img.shields.io/badge/version-v2.1.0-brightgreen" alt="Version: v2.1.0">
    <img src="https://img.shields.io/badge/tests-200%2B%20unit%2C%20mesh%2C%20integration-brightgreen" alt="Tests: 200+ unit + mesh + integration">
    <img src="https://img.shields.io/badge/ray-2.55%2B-06B7AC" alt="Ray">
    <img src="https://img.shields.io/badge/FastAPI-0.110%2B-009700?logo=fastapi&logoColor=white" alt="FastAPI">
    <img src="https://img.shields.io/badge/pydantic-v2-7B3FBF?logo=pydantic&logoColor=white" alt="Pydantic v2">
    <img src="https://img.shields.io/badge/PostgreSQL-16%2B-4169E1?logo=postgresql&logoColor=white" alt="PostgreSQL">
    <img src="https://img.shields.io/badge/NumPy-1.24%2B-01326D?logo=numpy&logoColor=white" alt="NumPy">
    <img src="https://img.shields.io/badge/SciPy-1.10%2B-8C4A9F?logo=scipy&logoColor=white" alt="SciPy">
    <img src="https://img.shields.io/badge/OPA%2FRego-policy%20as%20code-00AEA9" alt="OPA/Rego policy-as-code">
    <img src="https://img.shields.io/badge/LinUCB-contextual%20bandits-FE8003" alt="LinUCB contextual bandits">
  </p>
</p>

---

## Table of Contents

- [Overview](#overview)
- [V2 Mesh Architecture](#v2-mesh-architecture)
- [Design Philosophy](#design-philosophy)
- [Adaptive Intelligence Layer (LinUCB Bandits)](#adaptive-intelligence-layer-linucb-bandits)
- [Procurement Lifecycle](#procurement-lifecycle)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Testing, Verification & CI/CD Pipeline](#testing-verification--cicd-pipeline)
- [Security](#security)
- [Changelog](#changelog)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**Autonomous Procurement Swarm** is a distributed neuro-symbolic system for autonomous procurement negotiations. The V2 runtime is built on a Ray actor mesh with a typed blackboard, capability-scoped communication, OPA/Rego policy enforcement, and LinUCB contextual bandits for adaptive strategy selection.

### 🔄 Architecture Transition

The system has evolved from a single-process `asyncio` engine into a distributed Ray actor mesh:

| Aspect | V2 Distributed Mesh (`mesh/`) | V1 Legacy Engine (`legacy/`) |
|---|---|---|
| **Execution Model** | Ray actors across multiple nodes | Single-process asyncio EventBus |
| **Communication** | DistributedBlackboard with typed channels + capability ACLs | Shared EventBus, zero coupling |
| **Governance** | SafetyKernelActor with OPA/Rego rules | Hardcoded safety checks |
| **Adaptation** | LinUCB contextual bandits (online learning) | Offline replay-based policy learning |
| **Scalability** | Elastic actor pools (Scout, Evaluator, Negotiator) | Single-process, bounded scalability |
| **Determinism** | Deterministic MCDA in BuyerActor (singleton) | 14-agent deterministic orchestration |
| **Primary Runtime** | **V2 Mesh** — recommended for all new work | Preserved for backward compatibility |

> The original V1 asyncio runtime (`swarm.core.EventBus`, `swarm.orchestration.SwarmCoordinator`) is preserved in `legacy/` and accessible for backward-compatible sealed-bid auctions. New code should use the V2 mesh runtime (`mesh/` package + `api.v2`).

---

## V2 Mesh Architecture


**Distributed Blackboard** — `DistributedBlackboard` (Ray actor) provides typed channels with capability-scoped ACLs:

```

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                          DISTRIBUTED BLACKBOARD (Ray Actor)                             │
│                      [Append-Only Traces | Capability-Scoped ACLs]                      │
└──────┬───────────────────────┬─────────────────────────┬─────────────────────────┬──────┘
       │ REQUIREMENT           │ DISCOVERY               │ SCORE / RISK            │ DEAL
       ▼                       ▼                         ▼                         ▼
┌──────────────┐        ┌──────────────┐         ┌──────────────┐          ┌──────────────┐
│  ScoutActor  │        │EvaluatorActor│         │NegotiatorActor│         │  BuyerActor  │
│    (Pool)    │        │    (Pool)    │         │     (Pool)   │          │ (Singleton)  │
└──────┬───────┘        └──────┬───────┘         └───────┬──────┘          └───────┬──────┘
       │                       │                         │                         │
       │                       │                         ▼ (Strategy Selection)    │
       │                       │                  ┌──────────────┐                 │
       │                       │                  │ LinUCBBandit │                 │
       │                       │                  └──────────────┘                 │
       │                       ▼                         │                         │ (No LLM)
       │              (Deterministic MCDA)               │                         │
       │                                                 │                         │
       └─────────────────────────┬───────────────────────┘                         │
                                 ▼                                                 ▼
                      ┌─────────────────────┐                          ┌──────────────────────┐
                      │ NeuroSymbolicBridge │                          │      DECISION        │
                      └──────────┬──────────┘                          │ (Deterministic MCDA) │
                                 ▼                                     └──────────────────────┘
                      ┌─────────────────────┐
                      │  SafetyKernelActor  │ 
                      │ (OPA/Rego Policies) │
                      └─────────────────────┘

```

| Channel | Description |
|---|---|
| `REQUIREMENT` | Purchase requirements ingested |
| `DISCOVERY` | Supplier discovery results (parallel ScoutActor pool) |
| `SCORE` | Multi-criteria evaluation scores (parallel EvaluatorActor pool) |
| `RISK` | Financial, delivery, quality, carbon risk metrics |
| `DEAL` | Schema-constrained quotes (parallel NegotiatorActor pool + LinUCB) |
| `DECISION` | Winning award from centralized BuyerActor (deterministic MCDA) |

**Signal path:** `REQUIREMENT → DISCOVERY → SCORE/RISK → DEAL → DECISION → REWARD`

### 🗺️ Component Map

| Component | Path | Description |
|---|---|---|
| **DistributedBlackboard** | `mesh/blackboard.py` | Ray actor with typed channels (`REQUIREMENT`, `DISCOVERY`, `SCORE`, `RISK`, `DEAL`, `DECISION`) |
| **Channel ACLs** | `mesh/channels.py` | Capability-scoped read/write permissions per channel |
| **ScoutActor** | `mesh/actors/scout.py` | Elastic pool: reads `REQUIREMENT` → writes `DISCOVERY` |
| **EvaluatorActor** | `mesh/actors/evaluator.py` | Elastic pool: reads `DISCOVERY` → writes `SCORE`, `RISK` |
| **NegotiatorActor** | `mesh/actors/negotiator.py` | Elastic pool: reads `SCORE` → writes `DEAL` with LinUCB strategy selection |
| **BuyerActor** | `mesh/actors/buyer.py` | Singleton: reads `DEAL`/`RISK`/`SCORE` → writes `DECISION` (deterministic MCDA) |
| **SafetyKernelActor** | `mesh/actors/base.py` | Singleton: validates all neural proposals via OPA/Rego rules |
| **NeuroSymbolicBridge** | `mesh/neuro/bridge.py` | Retry loop: LLM generates → kernel validates → auto-correct or fallback |
| **LinUCBBandit** | `mesh/neuro/bandits.py` | Online contextual bandit for adaptive negotiation strategy selection |
| **ProcurementCluster** | `mesh/cluster.py` | Ray cluster lifecycle (head/worker) with elastic actor scaling + persistence |

---

## Design Philosophy

| Principle | Implementation |
|---|---|
| **Neuro-Symbolic Bounding** | Non-deterministic LLM proposals are generated under schema constraints (Pydantic) and must strictly pass the symbolic `SafetyKernelActor` (OPA/Rego rules) before hitting the blackboard. |
| **Ray Actor Scalability** | Scout, Evaluator, and Negotiator archetypes scale elastically across nodes; the Buyer agent remains a single centralized MCDA decision actor for compliance. |
| **Capability-Scoped Communication** | Cross-agent communication occurs on `DistributedBlackboard` with enforced read/write permissions per channel (`REQUIREMENT`, `DISCOVERY`, `SCORE`, `RISK`, `DEAL`, `DECISION`). |
| **Deterministic Decisioning** | The `BuyerActor` computes the final award using deterministic MCDA math — no LLM in the control path. |
| **Online Policy Learning** | LinUCB contextual bandits select negotiation strategies based on real-time context, with asynchronous reward feedback from BuyerActor decisions. |
| **Policy-as-Code Governance** | All proposals are validated against OPA/Rego rulesets that enforce budgets, lead-time bounds, ESG material whitelists, and payment term policies. |
| **Observability by Default** | Every event, artifact, and external call is recorded with full lineage through the distributed blackboard. |

---

## Adaptive Intelligence Layer (LinUCB Bandits)

Phase 5 introduces **Online Policy Learning** with LinUCB (Linear Upper Confidence Bound) contextual bandits for adaptive negotiation strategy selection in `NegotiatorActor`.

### 📐 Context Vector ($x_t \in \mathbb{R}^6$)

| Component | Description | Normal Range |
|---|---|---|
| Urgency | Negotiation time pressure | [0, 1] |
| Target budget margin | (budget - unit_price) / budget | [0, 1] |
| Supplier historical rating | Supplier quality score | [0, 1] |
| Material complexity | Normalized complexity score | [0, 1] |
| Historical win rate | Supplier past award success | [0, 1] |
| Negotiation round | Current round index | Normalized |

### 🎮 Discrete Action Space

| Strategy | Description |
|---|---|
| `AGGRESSIVE_ANCHOR` | Start with aggressive price anchor to maximize concessions |
| `BALANCED_CONCESSION` | Even split of concession burden across dimensions |
| `PAYMENT_TERMS_TRADE_OFF` | Trade price concessions for favorable payment terms |
| `RISK_AWARE_PACING` | Slower concessions, emphasizing risk metrics |
| `RELATIONSHIP_BUILDING` | Prioritize trust-building for long-term partnerships |

### 🏆 Closed-Loop Reward ($r_t$)

| Component | Weight | Description |
|---|---|---|
| Cost reduction | 40% | $(bid - award) / bid$ |
| Payment terms favorability | 20% | Earlier payment terms rewarded |
| Convergence speed | 20% | Fewer negotiation rounds = higher reward |
| MCDA alignment | 20% | How well the negotiated quote aligns with final MCDA score |

Reward feedback is **asynchronous** — computed when `BuyerActor` writes a `DECISION` to the blackboard, then fed back via `DistributedBlackboard` to update all `NegotiatorActor` bandit parameters. Bandit state is **persisted** across cluster restarts via JSON serialization.

### 🧠 Cognition (SLM-First, LLM-Augmented)

Agent cognition lives entirely in `mesh/neuro`. Every `NegotiatorActor` proposal is generated by the **`NeuroSymbolicBridge`** (`mesh/neuro/bridge.py`), which mediates a structured language backend against the `SafetyKernelActor` in a bounded retry loop:

1. **SLM-first.** The default backend is a small open-weight model (`gemma-2b-it` via `NEURO_LLM_MODEL`) served over a local OpenAI-compatible endpoint (`NEURO_LLM_BASE_URL` → vLLM, Ollama, or llama.cpp). The SLM makes the first proposal, keeping the common path cheap and offline.
2. **LLM-augmented.** Confidence is reported per proposal (defaults to `1.0`); low-confidence, empty, or repeatedly-rejected outputs escalate to a larger fall-back model until the retry budget (`SAFETY_KERNEL_MAX_RETRIES`) is exhausted.
3. **Structured output.** Proposals are emitted as typed Pydantic models through `instructor` (`mesh/neuro/backend.py`, `OpenAICompatibleBackend`), guaranteeing schema conformance on the `DEAL` channel.
4. **Neuro-symbolic bounding.** The bridge enforces the contract `LLM generates → SafetyKernelActor validates → auto-correct or fallback`. No proposal reaches the blackboard unverified, and the kernel (OPA/Rego) is the hard stop on spend caps and material whitelists.
5. **Testability.** A no-LLM, no-network `StubBackend` lets `pytest` exercise the retry loop deterministically in CI without any model serving.

---

## Procurement Lifecycle

The V2 distributed mesh executes a 6-phase procurement lifecycle:

1. **`REQUIREMENT`** — Purchase requirement ingested into `DistributedBlackboard`.
2. **`DISCOVERY`** — Parallel `ScoutActor` pool identifies and validates suppliers.
3. **`SCORE` & `RISK`** — Parallel `EvaluatorActor` pool computes multi-criteria scores and risk metrics.
4. **`DEAL`** — `NegotiatorActor` pool applies LinUCB strategy hints, generates schema-constrained quotes via LLM, passes `SafetyKernelActor` verification, and writes to `DEAL`.
5. **`DECISION`** — Centralized `BuyerActor` executes deterministic MCDA math and selects winning award.
6. **`REWARD`** — Decision listener feeds scalar rewards back to update bandit parameters for future rounds.


---

## Quick Start

### 📋 Prerequisites

- Python 3.11+
- Docker & Docker Compose
- 4GB free RAM (Ray cluster + API)

### 🚀 Start the V2 Distributed Mesh

```bash
# Clone and spin up Ray cluster + API v2 + Ray Dashboard
docker compose -f docker-compose.mesh.yml up -d
```

**Services:**
- API v2: `http://localhost:8000`
- Ray Dashboard: `http://localhost:8265`

### 📤 Submit a Procurement Run

```bash
curl -X POST http://localhost:8000/v2/procurement/run \
  -H "Content-Type: application/json" \
  -d '{
    "material": "steel",
    "quantity": 1000,
    "budget": 500000,
    "target_lead_time_days": 30,
    "enable_neuro": true
  }'
```

### 🔍 Check Status

```bash
# Blackboard snapshot + stats
curl http://localhost:8000/v2/procurement/{trace_id}/status

# Health check
curl http://localhost:8000/v2/procurement/health
```

### 📊 Run Benchmark Verification

```bash
python scripts/benchmark_mesh.py
```

---

## Configuration

All configuration is managed via `configs/base.yaml`, `configs/settings.py`, and the environment variables below — the single source of truth (the Quick-Start table was merged here).

| Variable | Default | Description |
|---|---|---|
| `MESH_ROLE` | `api` | Container role: `head`, `worker`, or `api` |
| `RAY_ADDRESS` | — | Ray cluster address |
| `MESH_N_SCOUTS` | `3` | ScoutActor pool size |
| `MESH_N_EVALUATORS` | `3` | EvaluatorActor pool size |
| `MESH_N_NEGOTIATORS` | `2` | NegotiatorActor pool size |
| `BANDIT_ALPHA` | `0.5` | LinUCB exploration parameter |
| `BANDIT_STATE_PATH` | `/app/data/bandit_state.json` | Bandit state persistence path |
| `SAFETY_KERNEL_MAX_RETRIES` | `3` | Max LLM retry attempts per quote |
| `OA_POLICY_PATH` | `/app/policies/` | OPA/Rego rules directory |
| `NEURO_LLM_BASE_URL` | — | LLM endpoint for the neuro-symbolic bridge |
| `NEURO_LLM_MODEL` | `gemma-2b-it` | LLM model name |

---

## API Reference

Interactive docs: [Swagger UI](http://localhost:8000/docs) and [ReDoc](http://localhost:8000/redoc).

### 🌐 V2 Distributed Mesh Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v2/procurement/run` | Start a procurement run on the mesh |
| `GET` | `/v2/procurement/{trace_id}/status` | Blackboard snapshot + stats |
| `GET` | `/v2/procurement/{trace_id}/timeline` | Causally ordered timeline |
| `GET` | `/v2/procurement/{trace_id}/snapshot` | Point-in-time state snapshot for auditability |
| `GET` | `/v2/procurement/health` | Cluster health check |

### 🏃 V2 Procurement Run

**Request:**

```json
{
  "material": "steel",
  "quantity": 1000,
  "budget": 500000,
  "target_lead_time_days": 30,
  "enable_neuro": true
}
```

**Response:**

```json
{
  "trace_id": "RUN-20260819-143200",
  "status": "complete",
  "channels": {
    "REQUIREMENT": 1,
    "DISCOVERY": 12,
    "SCORE": 12,
    "RISK": 12,
    "DEAL": 12,
    "DECISION": 1
  },
  "winning_supplier": "GlobalSteel Inc.",
  "winning_quote": 98.50,
  "total_savings": 15000.00,
  "bandit_strategy_used": "BALANCED_CONCESSION"
}
```

---

## Testing, Verification & CI/CD Pipeline

The project maintains a full test pyramid on a single `pytest` runner
(`pyproject.toml`: `asyncio_mode = "auto"`, `testpaths = ["tests"]`), backed by
**ruff** linting, **mypy** type checking across every runtime package, and
**Codecov** coverage upload. A deterministic end-to-end terminal suite
(`test_suite.sh`) and a mesh smoke benchmark (`scripts/benchmark_mesh.py`)
guard the full signal path.

### 🧪 Test Suite Overview

| Layer | Directory | Files | Scope |
|---|---|---|---|
| Unit | `tests/unit/` | 59 | `swarm.core` (agent, artifact, event, state, timeline, registry, completion), `swarm.domain.*` agents, `swarm.learning`, `swarm.utils.llm_*`, `swarm.integrations`, `swarm.storage.event_store`, plus the ray-free `mesh.neuro` modules (bandits, bridge, schemas, kernel, backend) and `core/` primitives. |
| Integration | `tests/integration/` | 10 | Full HTTP + flow runs: `test_api.py`, `test_v2_api.py`, `test_execution_flow.py`, `test_governance_flow.py`, `test_feedback_cycle.py`, `test_contract_validation.py`, `test_erp_connector.py`, `test_external_sync_api.py`, `test_external_calls.py`, `test_load.py`. |
| Mesh | `tests/mesh/` | 5 | `test_actors.py`, `test_bandits.py`, `test_neuro_actors.py`, `test_neuro_bridge.py`, `test_neuro_schemas.py` — actor behaviour, LinUCB bandits, neuro-symbolic retry loop, schema constraints (Ray-free). |

Total: **74 test files / ~820 test functions**, split 59 unit / 10 integration / 5 mesh.

### ▶️ Running Tests

```bash
# Full suite (requires Postgres on 5433 for integration tests)
pytest tests/ -v

# Unit layer only
pytest tests/unit/ -v

# V2 mesh (ray-free neuro tests run without a cluster)
pytest tests/mesh/ -v

# LinUCB bandit only
pytest tests/mesh/test_bandits.py -v

# With coverage (CI uses --cov=core --cov=api; add mesh locally)
pytest tests/ --cov=core --cov=api --cov=mesh --cov-report=term --cov-report=html
```

> The `mesh` package is optional at import time — neuro/symbol tests run in any
> environment. Tests that require a live Ray cluster are exercised via Docker
> Compose and the benchmark script, not the headless CI job.

### 🔄 CI/CD Pipeline

`.github/workflows/ci.yml` runs on every `push` and `pull_request` to `main`.
A `pgvector/pgvector:pg16` service is exposed on port **`5433`**
(`DATABASE_URL=postgresql+asyncpg://procurement:procurement@localhost:5433/procurement`),
Python 3.11 is installed, and `pip install -e ".[dev,ray]"` pulls in Ray. The job is:

1. **Lint** — `ruff check core/ api/ swarm/ examples/ tests/ scripts/ mesh/ legacy/`
2. **Type check** — `mypy core/ api/ swarm/ examples/ mesh/ legacy/ --ignore-missing-imports`
3. **Test** — `pytest tests/ -v --tb=short --cov=core --cov=api --cov-report=xml`
4. **Coverage upload** — `codecov/codecov-action@v4` publishes `coverage.xml`

Both lint and type-check sweep **all** runtime packages (`core`, `api`, `swarm`,
`mesh`, `legacy`, `examples`, `scripts`), so the Ray mesh and the legacy engine
share the same quality bar.

### ✅ End-to-End Verification

`test_suite.sh` is a terminal integration suite that hits the running HTTP API
on `http://localhost:8000` (start the stack with `docker compose up -d`). It is
split into two phases:

- **Functional** — health/DB check, sealed-bid auction → `AWARDED`, ledger
  hash-chain integrity, auction stats, supplier profiles + similarity search,
  and validation guards (invalid material → 422, zero quantity → 422, negative
  supplier count → 422, missing session → 404).
- **Resilience** — 10 concurrent auctions in a fan-out, 20 sequential bartering
  auctions, DB persistence across runs, chain integrity under load, a 100 000
  unit stress run, supplier-memory accumulation, and a determinism check on the
  similarity endpoint.

```bash
./test_suite.sh        # exit 0 = all assertions passed
```

### 📈 Mesh Benchmark

`scripts/benchmark_mesh.py` validates the **V2** distributed mesh end-to-end
against a live Docker Compose mesh deployment. It posts a requirement to
`POST /v2/procurement/run`, polls `GET /v2/procurement/{trace_id}/status`
until completion, and asserts the full typed-channel signal path executed:

```
REQUIREMENT → DISCOVERY → SCORE / RISK → DEAL → DECISION
```

```bash
docker compose -f docker-compose.mesh.yml up --build -d
python scripts/benchmark_mesh.py        # exit 0 = all phases verified
```

### 📊 LinUCB Strategy Convergence Profiling

The `NegotiatorActor` LinUCB bandit is validated for strategy-convergence in the ray-free mesh unit tests, which drive the `DEAL` phase (where `LinUCBBandit` selects strategies) and assert that selected strategies stabilize toward the deterministic `BuyerActor` MCDA optimum:

```bash
pytest tests/mesh/test_bandits.py -v
```

End-to-end convergence against a live mesh is covered by `scripts/benchmark_mesh.py` (see Mesh Benchmark, above).

---

## Security

### 🛡️ Current Protections

1. **Input validation** — Pydantic v2 schemas reject malformed requests (HTTP 422)
2. **Material whitelist** — Only configured commodities are accepted
3. **Spend caps** — Hard budget limits enforced by `SafetyKernelActor` (OPA/Rego) — no LLM override
4. **Capability-scoped ACLs** — Cross-agent communication enforced via typed blackboard channels
5. **Secret masking** — Timeline projection redacts sensitive fields
6. **Contract pre-gate** — Invalid/expired contracts short-circuit to REJECTED
7. **Hash chain integrity** — All artifacts carry parentage for auditability
8. **Neuro-symbolic bounding** — All LLM proposals must pass OPA/Rego validation before entering the blackboard

### 📜 OPA/Rego Policy Rules

Safety policies are defined as Rego files in `/app/policies/`:

| Rule File | Enforced Constraint |
|---|---|
| `budget_limit.rego` | Quote must not exceed negotiated budget |
| `lead_time_bound.rego` | Delivery date within target lead time |
| `material_whitelist.rego` | Material must be on approved list |
| `payment_terms.rego` | Payment terms must be from approved set |

See `SECURITY.md` for responsible disclosure policy.

---

## Changelog

### 🏷️ [v2.1.0] — Contextual Bandits & Adaptive Negotiation Mesh

- **LinUCB Contextual Bandits**: Online policy learning in `NegotiatorActor` with 6-dimensional context vectors and 5 discrete strategy profiles
- **Strategy-Guided Negotiation**: `NegotiatorActor` selects negotiation strategy via `bandit.select_action()` and injects strategy hints into LLM prompts
- **Closed-Loop Rewards**: Asynchronous reward computation triggered when `BuyerActor` writes `DECISION`, evaluating cost reduction (40%), payment terms (20%), convergence speed (20%), MCDA alignment (20%)
- **Bandit Persistence**: `save_state()`/`load_state()` JSON serialization for persistence across Ray cluster restarts via `BANDIT_STATE_PATH`
- **Cold-Start Fallback**: Uniform exploration with tie-breaking noise when no history exists
- **Context Normalization**: Urgency, budget margin, supplier rating, material complexity, historical win rate, negotiation round

### 🏷️ [v2.0.0-rc1] — Distributed Neuro-Symbolic Mesh

- **Ray Mesh Runtime**: Distributed blackboard (Ray actor) with capability-scoped ACLs for 6 typed channels (`REQUIREMENT`, `DISCOVERY`, `SCORE`, `RISK`, `DEAL`, `DECISION`)
- **Four Elastic Agent Archetypes**: ScoutActor, EvaluatorActor, NegotiatorActor (all elastic), BuyerActor (singleton MCDA)
- **SafetyKernelActor**: Singleton symbolic validator enforcing budget clamping, lead-time bounds, ESG material whitelist, and payment term policy
- **NeuroSymbolicBridge**: Retry loop — LLM generates schema-constrained proposals → kernel validates → auto-correct on failure → deterministic fallback after `neuro_max_retries`
- **V2 API**: `POST /v2/procurement/run`, `GET /v2/procurement/{trace_id}/status`, `GET /v2/procurement/health`
- **Docker Mesh Deployment**: Multi-stage Dockerfile with `mesh.entrypoint` supporting head/worker/api roles; `docker-compose.mesh.yml` with Ray head + 2 workers + FastAPI v2 + Ray Dashboard
- **V1 Deprecation**: Original asyncio runtime moved to `legacy/` package with backward compatibility

---

## Contributing

Fork the repository → Create a feature branch → Install dev dependencies → Run quality checks → Submit a Pull Request.

```bash
# Create a feature branch
git checkout -b feature/your-feature

# Install dev + Ray dependencies (as CI does)
pip install -e ".[dev,ray]"

# Lint — all packages, exactly as CI runs
ruff check core/ api/ swarm/ examples/ tests/ scripts/ mesh/ legacy/

# Type check — all runtime packages, exactly as CI runs
mypy core/ api/ swarm/ examples/ mesh/ legacy/ --ignore-missing-imports

# Full test suite (unit + mesh + integration)
pytest tests/ -v
```

> These commands mirror `.github/workflows/ci.yml`, which runs ruff + mypy over
> every package and the full `pytest tests/` suite against a
> `pgvector/pgvector:pg16` service on port **5433**
> (`DATABASE_URL=postgresql+asyncpg://procurement:procurement@localhost:5433/procurement`).
> Start the stack with `docker compose up -d` before running integration tests.

## Citation

If you use the Autonomous Procurement Swarm in your work, we'd appreciate a citation.
The canonical citation metadata lives in [`CITATION.cff`](CITATION.cff) (CFF 1.2.0);
GitHub also renders a **Cite this repository** button from it (APA, BibTeX, etc.).

For v2.1.0 (the distributed Ray mesh runtime), the BibTeX entry is:

```bibtex
@software{procurement_swarm_v210,
  author       = {{Autonomous Procurement Swarm}},
  title        = {Autonomous Procurement Swarm},
  version      = {v2.1.0},
  year         = {2026},
  month        = aug,
  url          = {https://github.com/aragit/autonomous-procurement-swarm},
  note         = {Distributed neuro-symbolic multi-agent mesh with LinUCB contextual bandits}
}
```

Please cite the specific tagged release you build on rather than the repository HEAD.

## License

Distributed under the MIT License. See `LICENSE` for details.
