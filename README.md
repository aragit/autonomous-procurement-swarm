<p align="center">
  <h1 align="center">Autonomous Procurement Swarm</h1>
  <p align="center">
    <b>Production-grade, deterministic multi-agent procurement</b><br>
    Cryptographic audit trails · Enterprise ERP integration · Zero-LLM governance · Offline policy learning
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python: 3.11+">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT">
    <img src="https://img.shields.io/badge/status-production--ready-brightgreen" alt="Status: Production-ready">
    <img src="https://img.shields.io/badge/tests-47%2B%20unit-brightgreen" alt="Tests: 47+ unit + integration">
    <img src="https://img.shields.io/badge/version-v0.8.2--unreleased-orange" alt="Version">
  </p>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Design Philosophy](#design-philosophy)
- [Adaptive Intelligence Layer](#adaptive-intelligence-layer)
  - [Policy Learning](#policy-learning)
  - [Adaptive Routing](#adaptive-routing)
  - [Contextual Parameter Adaptation (Step 25)](#contextual-parameter-adaptation-step-25)
- [Architecture](#architecture)
  - [Dual-Runtime Model](#dual-runtime-model)
  - [Swarm Agent Topology](#swarm-agent-topology)
  - [Enterprise Integration Layer](#enterprise-integration-layer)
  - [Artifact Lineage & Audit Trail](#artifact-lineage--audit-trail)
  - [State Machines](#state-machines)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Reference](#api-reference)
  - [Auction Engine (Legacy Core)](#auction-engine-legacy-core)
  - [Deterministic Swarm](#deterministic-swarm)
  - [Policy & Learning](#policy--learning)
  - [LLM Observability](#llm-observability)
- [Determinism vs Adaptation](#determinism-vs-adaptation)
- [Replay-Based Evaluation](#replay-based-evaluation)
- [CLI & Examples](#cli--examples)
- [Testing](#testing)
- [Performance](#performance)
- [Security](#security)
- [Troubleshooting](#troubleshooting)
- [Changelog](#changelog)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**Autonomous Procurement Swarm** is a dual-runtime, production-grade system for autonomous procurement negotiations. It combines two execution models under a single FastAPI control plane:

> **Note on terminology:** This system is not a decentralized swarm in the academic sense. The term "swarm" refers to a **deterministic, event-driven multi-agent orchestration system** designed for auditability, control, and bounded adaptation. Agents communicate via a shared EventBus with zero direct coupling.

| Feature | Core Auction Engine (`core/`) | Deterministic Orchestration (`swarm/`) |
|---|---|---|
| **Execution Model** | LLM-powered 1×N sealed-bid reverse auctions | 14-agent event-driven system, zero LLM in control path |
| **Scoring** | Multi-criteria weighted scoring | Deterministic, policy-driven scoring |
| **Memory** | pgvector embeddings + heuristic estimator | Structured feedback folding |
| **Ledger** | SHA-256 hash-chained PostgreSQL | Append-only event store |
| **Integration** | Hardcoded adapters | Runtime-configurable connector factory |

The orchestration runtime (colloquially "the swarm") is the **primary execution path** for new work. The core auction engine remains available for backward-compatible sealed-bid auctions and as the LLM-powered bid-generation backend.

### What Makes This Different

| Aspect | Traditional Procurement SaaS | This System |
|---|---|---|
| Decision Logic | Rule-based workflows | 14 specialized deterministic agents with capability routing |
| Risk Assessment | Manual approval queues | Automated financial/delivery/quality/carbon scoring |
| Governance | Human-in-the-loop only | Policy-driven with simulated or real approval gates |
| External Integration | Hardcoded ERP adapters | Runtime-configurable connector factory (Mock → SAP/Oracle/Coupa) |
| Auditability | Transaction logs | Cryptographic hash chains + artifact lineage + deterministic timeline projection |
| Supplier Memory | Static vendor master | Behavioral embeddings + performance feedback loops |
| LLM Usage | Everywhere (unpredictable) | Isolated to bid generation and cognitive analysis only |
| Adaptation | None | Offline replay-based policy learning with explicit promotion |

---

## Design Philosophy

| Principle | Implementation |
|---|---|
| **Determinism over Stochasticity** | All control-path decisions (scoring, risk, governance, execution) are pure Python. LLMs only generate strategic bids and read-only cognitive analysis. |
| **Observability by Default** | Every event, artifact, and external call is recorded with full lineage. |
| **Auditability as a Feature** | SHA-256 hash-chained ledger, artifact parentage DAGs, timeline projection. |
| **Replay Safety** | All decisions are pure functions of state — re-runs produce identical results. |
| **Controlled Adaptation** | Policy learning + contextual overrides enable bounded, auditable system evolution without sacrificing determinism |
| **Extensibility via Composition** | New scoring criteria, policy rules, LLM backends, ERP connectors, and domain agents are plug-and-play via capability registration. |
| **Zero Agent Coupling** | Agents communicate exclusively through the EventBus. No agent holds a direct reference to another. |

---

## Adaptive Intelligence Layer

The system is no longer purely static. It incorporates a **controlled adaptation layer** — offline, replay-based policy learning and contextual parameter adjustment — that evolves system behavior without compromising determinism or auditability.

> Learning is **offline** and **explicit**. No policy changes occur during live execution. Adaptation happens between runs via explicit policy promotion. The execution path for any given input + policy version is always deterministic and reproducible.

### Policy Learning

Policies are learned from historical traces and feedback signals:

1. **Trace Gathering** — Historical procurement runs are collected from the event store, each with artifacts (decisions, scores) and feedback (success, outcome scores).
2. **Candidate Generation** — A deterministic grid of strategy + parameter variants is generated around the active policy.
3. **Replay Evaluation** — Each candidate is evaluated by replaying historical traces through the deterministic replay engine. No live system calls or state mutations occur.
4. **Promotion** — Only candidates that exceed the active policy's metric by a safety margin are eligible for promotion. Promotion is **explicit** — no automatic unsafe mutation is permitted.

**Safety guarantees:**
- All candidate parameters (thresholds, weights) are clamped to safe bounds.
- Drift from the active policy is bounded by `POLICY_MAX_PARAM_DELTA` (±0.20 per parameter).
- Zero-delta and out-of-bounds overrides are rejected during validation.
- Force-promote is available only as a rollback mechanism with explicit operator action.

### Adaptive Routing

Strategy selection is **context-aware** but remains fully deterministic:

- A **routing strategy** maps trace context (budget level, urgency, supplier pool size) to a strategy type (`balanced`, `cost_optimized`, `quality_first`, `trust_weighted`) plus **parameter overrides**.
- Routing rules are evaluated deterministically using **priority-based conflict resolution**:
  - All matching rules are evaluated.
  - The **highest priority** rule wins.
  - Ties are broken by **fewer conditions** (more general rule wins).
  - Further ties are resolved by **stable `rule_id` ordering**.
- Context is **normalized** (lowercased keys, stripped strings, `None` values removed) before matching to ensure deterministic rule evaluation regardless of input formatting.
- If no rule matches, the **default strategy** is used.

### Contextual Parameter Adaptation (Step 25)

Parameters can be contextually overridden at runtime by routing rules:

| Parameter Type | Overrideable Parameters | Default Delta Bound | Clamp Range |
|---|---|---|---|
| Evaluation Thresholds | `confidence_threshold`, `stability_threshold`, `trust_threshold` | ±0.05 | [0.6, 0.9] |
| Scoring Weights | `price_weight`, `score_weight`, `carbon_weight` | ±0.10 | [0.1, 0.8], re-normalized |

**Guardrails:**
- Each rule may specify **at most 2 parameter deltas** (`MAX_PARAMS_PER_RULE`).
- Total absolute delta across all overrides in a single rule is capped at **0.15** (`PARAM_OVERRIDE_TOTAL_DELTA_CAP`).
- Zero-delta overrides are rejected (noise filter).
- All overrides are validated at policy creation time via `validate_param_overrides()`.
- Thresholds and weights are **clamped** and **re-normalized** after applying deltas, ensuring safety bounds are never violated.
- An **idempotency flag** (`_overrides_applied_flag`) prevents double application when both the runtime path and the replay path would apply the same overrides.

---

## Architecture

### Dual-Runtime Model

```
flowchart TB
subgraph ControlPlane["FastAPI Control Plane"]
  API["api.main:app"]
  SWARM_API["swarm.api.*"]
  LLM_OBS["/llm-obs (v0.9)"]
end
subgraph CoreEngine["Core Auction Engine (core/)"]
  BUYER["BuyerOrchestrator"]
  SUPPLIERS["SupplierAgent × N (LLM-powered)"]
  POLICY["PolicyEngine"]
  EVAL["MultiCriteriaEvaluator"]
  ORCH["AuctionOrchestrator"]
  LEDGER["PostgresLedgerRepository"]
  MEM_HEUR["HeuristicReservationEstimator"]
  MEM_VEC["PgVectorMemoryStore"]
end
subgraph SwarmRuntime["Deterministic Swarm Runtime (swarm/)"]
  SWARM["Swarm Facade"]
  BUS["EventBus"]
  STATE["SwarmState (Artifacts + Events)"]
  AGENTS["14 Domain Agents"]
  TRACKER["CompletionTracker"]
  TIMELINE["Timeline Projection"]
end
subgraph Enterprise["Enterprise Integration (swarm/integrations/)"]
  FACTORY["ConnectorConfig + build_connector"]
  MOCK["MockConnector"]
  SAP["SAPConnector"]
  ORACLE["OracleConnector"]
  COUPA["CoupaConnector"]
  IDEMP["IdempotencyGuard"]
end

API -->|POST /auctions| CoreEngine
API -->|POST /swarm/requirements| SwarmRuntime
API -->|GET /swarm/timeline/{id}| TIMELINE
API -->|mount /llm-obs| LLM_OBS
CoreEngine -->|hash-chained events| LEDGER
CoreEngine -->|semantic memory| MEM_VEC
CoreEngine -->|heuristic profiles| MEM_HEUR
SwarmRuntime -->|idempotent external calls| Enterprise
SwarmRuntime -->|read-only projection| TIMELINE
Enterprise -->|ExternalCallArtifact| STATE
```

**Key architectural note:** Strategy selection and parameter adaptation are now **dynamic** — the `StrategyAgent` resolves a context-aware strategy at runtime via `AdaptivePolicyEngine`, and evaluation thresholds/weights can be contextually overridden via `ContextualParameterAdapter`. The `ReplayEngine` (offline) uses the same deterministic logic to evaluate candidate policies before promotion.

### Swarm Agent Topology

The swarm runtime wires 14 specialized agents through a shared EventBus. Each agent implements `perceive → act` and publishes domain events. Agents are routed by event type and capability, not by direct reference.

```
flowchart LR
USER["User / API"] -->|CreateRequirement| RA["RequirementAgent"]
RA -->|RequirementCreated| ST["StrategyAgent"]
ST -->|StrategySelected| SDA["SupplierDiscoveryAgent"]
SDA -->|SupplierDiscovered × N| EA["EvaluationAgent"]
EA -->|SupplierEvaluated × N| NA["NegotiationAgent"]
NA -->|QuoteGenerated × N| CT["CompletionTracker"]
CT -->|QuotesCompleted| DA["DecisionAgent"]
DA -->|DecisionMade| CVA["ContractValidationAgent"]
CVA -->|ContractValidated| RSA["RiskAssessmentAgent"]
CVA -->|ContractRejected| GA["GovernanceAgent"]
RSA -->|RiskAssessmentCompleted| GA
GA -->|GovernanceDecisionMade| APA["ApprovalAgent"]
APA -->|ApprovalGranted| POA["PurchaseOrderAgent"]
POA -->|PurchaseOrderCreated| ETA["ExecutionTrackingAgent"]
ETA -->|ExecutionStatusUpdated| OA["OutcomeAgent"]
OA -->|OutcomeRecorded| SIA["SupplierIntelligenceAgent"]
SIA -->|SupplierPerformanceUpdated| SM["SupplierMemoryStore"]
SM -.->|history| EA
DA -->|QuotesCompleted| LLMA["SupplierAnalysisLLMAgent"]
LLMA -.->|cognitive analysis| DA
USER -->|POST /approve| APA
USER -->|POST /execute| POA
USER -->|POST /outcome| OA
USER -->|POST /swarm/requirements| RA
```

#### Agent Responsibility Matrix

| # | Agent | Capability | Trigger Event | Output Event | Key Artifact |
|---|---|---|---|---|---|
| 1 | RequirementAgent | `procurement.requirement` | `CreateRequirement` | `RequirementCreated` | `requirement` |
| 2 | StrategyAgent | `procurement.strategy` | `RequirementCreated` | `StrategySelected` | `strategy` |
| 3 | SupplierDiscoveryAgent | `procurement.supplier.discover` | `StrategySelected` | `SupplierDiscovered × N` | `supplier_list` |
| 4 | EvaluationAgent | `supplier.evaluate` | `SupplierDiscovered` | `SupplierEvaluated` | `evaluation` |
| 5 | NegotiationAgent | `procurement.negotiate` | `SupplierEvaluated` | `QuoteGenerated` | `quote` |
| 6 | DecisionAgent | `procurement.decide` | `QuotesCompleted` | `DecisionMade` | `decision` + `decision_explanation` |
| 7 | SupplierAnalysisLLMAgent | `procurement.analysis.llm` | `QuotesCompleted` | — | `llm / llm_consensus` |
| 8 | ContractValidationAgent | `procurement.contract.validate` | `DecisionMade` | `ContractValidated` / `ContractRejected` | `contract_validation` |
| 9 | RiskAssessmentAgent | `procurement.risk.assess` | `ContractValidated` | `RiskAssessmentCompleted` | `risk_assessment` |
| 10 | GovernanceAgent | `procurement.governance.apply` | `RiskAssessmentCompleted` | `GovernanceDecisionMade` / `ContractRejected` | `governance_decision` |
| 11 | ApprovalAgent | `procurement.approval.resolve` | `GovernanceDecisionMade` | `ApprovalGranted` / `ApprovalRequired` / `ApprovalRejected` | `authorization` |
| 12 | PurchaseOrderAgent | `procurement.order.create` | `ApprovalGranted` | `PurchaseOrderCreated` | `purchase_order` |
| 13 | ExecutionTrackingAgent | `procurement.execution.track` | `PurchaseOrderCreated` | `ExecutionStatusUpdated` | `execution_status` |
| 14 | OutcomeAgent | `procurement.outcome.record` | `RecordProcurementOutcome` | `OutcomeRecorded` | `procurement_outcome` |

> **Note:** CompletionTracker is not an agent but a runtime primitive subscribed to `ANY_EVENT`. It gates phase transitions by counting expected artifacts.

### Enterprise Integration Layer

```
flowchart TB
subgraph Swarm["Swarm Runtime"]
  POA["PurchaseOrderAgent"]
  ETA["ExecutionTrackingAgent"]
  IDEMP["IdempotencyGuard<br/>(decision_id + action)"]
end
subgraph Factory["Connector Factory"]
  CFG["ConnectorConfig<br/>provider / mode / endpoint / credentials"]
  BUILD["build_connector(config) → BaseConnector"]
end
subgraph Adapters["Runtime-Adaptive Adapters"]
  MOCK["MockConnector<br/>SUBMITTED → CONFIRMED → SHIPPED → DELIVERED"]
  SUP["SupplierAPIConnector"]
  SAP["SAPConnector"]
  ORA["OracleConnector"]
  COU["CoupaConnector"]
end
subgraph Audit["Audit Trail"]
  ECA["ExternalCallArtifact<br/>{system, action, request, response, idempotency_key}"]
end

POA -->|submit_order| IDEMP
ETA -->|get_order_status| IDEMP
IDEMP -->|deduplicated| Factory
Factory --> Adapters
Adapters -->|recorded| ECA
ECA -->|lineage to| Swarm
```

#### Environment-Driven Connector Selection

| Environment Variable | Default | Result |
|---|---|---|
| `PROCUREMENT_CONNECTOR_PROVIDER` unset | — | `mock` |
| `PROCUREMENT_CONNECTOR_MODE=sandbox` | — | `MockConnector` — deterministic lifecycle |
| `PROCUREMENT_CONNECTOR_MODE=sandbox` | `supplier_api` | `SupplierAPIConnector` — simulated HTTP |
| `PROCUREMENT_CONNECTOR_MODE=prod` | `sap` | `SAPConnector` — live ERP adapter with fallback simulation |
| `PROCUREMENT_CONNECTOR_MODE=prod` | `oracle` | `OracleConnector` — live ERP adapter |
| `PROCUREMENT_CONNECTOR_MODE=prod` | `coupa` | `CoupaConnector` — live ERP adapter |

### Artifact Lineage & Audit Trail

Every artifact in `SwarmState` carries `parent_ids`, creating an immutable DAG:

```
flowchart TD
REQ["requirement"] --> STR["strategy"]
STR --> SL["supplier_list"]
SL --> E1["evaluation_SupplierA"]
SL --> E2["evaluation_SupplierB"]
E1 --> Q1["quote_SupplierA"]
E2 --> Q2["quote_SupplierB"]
Q1 --> D["decision"]
Q2 --> D
D --> DE["decision_explanation"]
D --> CV["contract_validation"]
CV --> RA["risk_assessment"]
RA --> GD["governance_decision"]
GD --> EA["execution_authorization"]
EA --> PO["purchase_order"]
PO --> ES["execution_status"]
PO --> EC1["external_call: submit_order"]
ES --> EC2["external_call: get_status"]
ES --> OUT["procurement_outcome"]
OUT --> SP["supplier_performance"]
```

**Lineage Guarantees:**

- Every artifact has a UUID `id` and `parent_ids: list[str]`
- `correlation_id` ties all artifacts and events to a single logical conversation
- `created_by` records the agent name for accountability
- `version` supports artifact evolution (latest wins on `get_artifact`)

### State Machines

```mermaid
stateDiagram-v2
    [*] --> INIT
    INIT --> RFQ_BROADCAST
    RFQ_BROADCAST --> BID_COLLECTION
    BID_COLLECTION --> EVALUATION
    EVALUATION --> SHORTLIST_BARTER
    SHORTLIST_BARTER --> AWARDED
    EVALUATION --> TERMINATED
    SHORTLIST_BARTER --> TERMINATED
    AWARDED --> [*]
    TERMINATED --> [*]
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- 2GB free RAM (PostgreSQL + API + optional LLM)

### 1. Clone & Install

```bash
git clone https://github.com/aragit/autonomous-procurement-swarm.git
cd autonomous-procurement-swarm
pip install -e .
```

### 2. Start Infrastructure

```bash
docker compose up -d
# PostgreSQL 16 + pgvector on localhost:5433
```

### 3. Run the Deterministic Swarm Demo

```bash
python -m examples.procurement_swarm_demo
```

**Expected output:**

```
🤖 AUTONOMOUS PROCUREMENT SWARM — Full Lifecycle Demo
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Requirement: Source 1000 units of aluminum
Strategy: low_carbon (carbon constraint triggered)
Suppliers: 5 discovered → 5 evaluated → 5 quoted

Decision: MinerCorp_A @ $984.00/unit
Risk: LOW (financial: 0.12, delivery: 0.88, quality: 0.91, carbon: 0.95)
Governance: APPROVED
Execution: SUBMITTED → CONFIRMED → SHIPPED → DELIVERED
Outcome: recorded → supplier intelligence updated
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 4. Start API Server

```bash
uvicorn api.main:app --reload --port 8000
```

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 5. Dispatch a Swarm Requirement

```bash
curl -X POST http://localhost:8000/swarm/requirements \
  -H "Content-Type: application/json" \
  -d '{
    "material": "aluminum",
    "quantity": 1000,
    "budget": 2000000,
    "target_lead_time_days": 30,
    "max_carbon_per_unit": 800
  }'
```

### 6. Inspect the Full Timeline

```bash
curl http://localhost:8000/swarm/timeline/REQ-XXXXXX
```

---

## Installation

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Base package (MockLLM, no PyTorch)
pip install -e .

# With real LLM support (~6GB download)
pip install -e ".[llm]"

# With vLLM support
pip install -e ".[vllm]"

# With all dev tools
pip install -e ".[dev]"
```

### Docker (Production)

```bash
docker compose build
docker compose up -d
# API at http://localhost:8000
# PostgreSQL at localhost:5433
```

> The production Dockerfile intentionally excludes PyTorch. The API uses MockLLM by default. For real LLM inference in containers:
>
> ```dockerfile
> # In Dockerfile, change:
> RUN pip install --no-cache-dir -e .
> # To:
> RUN pip install --no-cache-dir -e ".[llm]"
> ```

---

## Configuration

All configuration is managed via `configs/base.yaml` and environment variables (override via pydantic-settings).

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://procurement:procurement@localhost:5433/procurement` | PostgreSQL connection string |
| `PROCUREMENT_LLM__MODEL_NAME` | `Qwen/Qwen2.5-3B-Instruct` | HuggingFace model ID |
| `PROCUREMENT_LLM__PREFER_VLLM` | `false` | Use vLLM over Transformers |
| `PROCUREMENT_LLM__TEMPERATURE` | `0.7` | LLM temperature |
| `PROCUREMENT_LLM__MAX_TOKENS` | `512` | Max generation tokens |
| `PROCUREMENT_EVALUATION__SHORTLIST_SIZE` | `2` | Suppliers entering bartering |
| `PROCUREMENT_CONNECTOR_PROVIDER` | — | Connector provider (`mock`, `supplier_api`, `sap`, `oracle`, `coupa`) |
| `PROCUREMENT_CONNECTOR_MODE` | `sandbox` | Connector mode (`sandbox` / `prod`) |

### Base Configuration (`configs/base.yaml`)

```yaml
project:
  name: "autonomous-procurement-swarm"
  version: "0.8.1"
  description: "Deterministic multi-agent procurement with enterprise integration"

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

Interactive docs: [Swagger UI](http://localhost:8000/docs) and [ReDoc](http://localhost:8000/redoc).

### Auction Engine (Legacy Core)

#### `POST /auctions`

Start a sealed-bid reverse auction with optional bilateral bartering.

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
  "scored_bids": [...],
  "shortlist": [...],
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

#### `GET /auctions/{session_id}`

Retrieve auction events with ledger chain verification.

#### `GET /ledger/stats`

Global ledger statistics.

#### `GET /suppliers/{supplier_id}/profile`

Heuristic supplier profile (concession speed, margin, reliability).

#### `GET /suppliers`

List all suppliers with accumulated memory profiles.

#### `GET /suppliers/similar?supplier_id=MinerCorp_A&n=3`

Find behaviorally similar suppliers via pgvector cosine similarity.

### Deterministic Swarm

#### `POST /swarm/requirements`

Dispatch a requirement into the deterministic swarm. Runs the full parallel flow (requirement → strategy → discovery → evaluation → negotiation → decision → contract validation → risk → governance → approval) with zero LLM involvement.

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
  "completions": {"REQ-3F9A2C1B-CONV": ["evaluation", "quote"]},
  "event_count": 31,
  "artifact_count": 12
}
```

#### `GET /swarm/timeline/{request_id}`

The crown jewel of observability. A causally ordered, read-only projection that merges all events and artifacts into a single deterministic timeline with phase markers, lineage links, and sensitive-field masking.

**Response:**

```json
{
  "request_id": "REQ-3F9A2C1B",
  "status": "incomplete",
  "timeline": [
    {
      "id": "...",
      "type": "artifact",
      "subtype": "requirement",
      "phase": "discovery",
      "agent": "requirement_agent",
      "timestamp": "2026-08-06T12:00:00Z",
      "order": 0,
      "payload": {...}
    }
  ],
  "summary": {
    "total_events": 31,
    "total_artifacts": 12,
    "external_calls": 0
  }
}
```

#### Additional Swarm Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/swarm/explanation/{request_id}` | Human-readable decision explanation |
| `GET` | `/swarm/risk/{request_id}` | Deterministic risk assessment |
| `GET` | `/swarm/governance/{request_id}` | Governance decision + policy applied |
| `POST` | `/swarm/{request_id}/approve` | Resolve pending approval |
| `GET` | `/swarm/authorization/{request_id}` | Execution authorization status |
| `POST` | `/swarm/{request_id}/execute` | Create PO + track execution (idempotent) |
| `GET` | `/swarm/order/{request_id}` | Read-only purchase order |
| `GET` | `/swarm/execution/{request_id}` | Read-only execution status |
| `GET` | `/swarm/external/{request_id}` | External call audit trail |
| `POST` | `/swarm/{request_id}/sync` | Force external-system reconciliation |
| `POST` | `/swarm/{request_id}/outcome` | Feed post-decision outcome |
| `GET` | `/swarm/supplier/{supplier_id}/performance` | Deterministic supplier performance |
| `GET` | `/swarm/trace/{request_id}` | Raw execution trace |
| `GET` | `/swarm/state/{request_id}` | Full serialized state snapshot |

### Policy & Learning

The Adaptive Intelligence Layer exposes a controlled, offline API for policy evolution. Learning and activation are **separated** — learning never triggers live behavior changes.

#### `POST /policies/learn`

Idempotent. Gathers historical traces + feedback from the event store, generates candidate policies via grid search, evaluates each via the replay engine, and returns the best-eligible policy.

**Request:**

```json
{}  // empty — uses all available traces
```

**Response:**

```json
{
  "status": "ok",
  "best": {
    "version": "6e8a51562c78",
    "strategy_type": "balanced",
    "thresholds": {...},
    "weights": {...},
    "metric": 0.847,
    "success_rate": 0.89,
    "decision_stability": 0.91
  },
  "candidates_evaluated": 128,
  "trace_count": 32
}
```

#### `POST /policies/promote`

Atomically activates a selected policy version. Only the explicitly-chosen version is activated — no automatic promotion occurs.

**Request:**

```json
{
  "version": "6e8a51562c78",
  "force": false
}
```

**Response:**

```json
{
  "status": "promoted",
  "version": "6e8a51562c78"
}
```

**Promotion rules:**

- `force: false` — fails unless the candidate's metric exceeds the active policy by a safety margin.
- `force: true` — allows rollback to a previously-promoted version (operator-initiated).

#### `GET /policies/active`

Returns the currently active policy (thresholds, weights, strategy, version).

**Response:**

```json
{
  "version": "6e8a51562c78",
  "signature": "a1b2c3d4e5f6...",
  "thresholds": {"confidence_threshold": 0.7, "stability_threshold": 0.7, "trust_threshold": 0.7},
  "weights": {"price_weight": 0.4, "score_weight": 0.4, "carbon_weight": 0.2},
  "strategy": {"type": "routing", "rules": [...], "default": "balanced"},
  "metric": 0.847,
  "active": true
}
```

**Safety-first model:** Learning and promotion are explicit, separate operations. No policy enters production without an explicit `POST /policies/promote` call.

### LLM Observability

Mounted at `/llm-obs` (v0.9). Provides read-only cognitive analysis of swarm runs.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/llm-obs/runs` | List available swarm runs with LLM artifacts |
| `GET` | `/llm-obs/runs/{correlation_id}/analysis` | Supplier comparison analysis |
| `GET` | `/llm-obs/runs/{correlation_id}/consensus` | Temporal trust scoring across strategy evaluations |

---

## CLI & Examples

#### Full Swarm Lifecycle Demo

```bash
python -m examples.procurement_swarm_demo
```

Demonstrates: requirement → strategy → discovery → evaluation → negotiation → decision → contract validation → risk → governance → approval → order → execution → outcome → intelligence update.

#### Core Runtime Primitives Demo

```bash
python -m examples.run_swarm_demo
```

Demonstrates: BaseAgent lifecycle, EventBus pub/sub, SwarmState artifacts, and event replay.

#### Sealed-Bid Auction Demo

```bash
python scripts/run_cnp_auction.py
```

Demonstrates: CNP auction lifecycle with LLM-powered supplier bidding and bilateral bartering.

#### Programmatic Swarm Usage

```python
from swarm.domain.wiring import build_procurement_swarm

swarm = build_procurement_swarm(
    request_id="REQ-002",
    goal="Source aluminum",
    supplier_memory=supplier_memory,       # shared across runs
    governance_policy=strict_policy,     # or STANDARD_POLICY
    base_connector=build_connector_from_env(),  # runtime-selected
)
await swarm.start()
await swarm.send_message(
    "CreateRequirement",
    {"material": "aluminum", "quantity": 1000, "budget": 2000000},
    correlation_id="REQ-002-CONV"
)
await swarm.shutdown()

# Read-only introspection
trace = swarm.get_execution_trace("REQ-002-CONV")
timeline = build_timeline(swarm.state)
```

---

## Testing

### Unit & Integration Tests

> **Requires PostgreSQL running on port 5433.**

```bash
# Full suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=core --cov=api --cov=swarm --cov-report=html

# Specific modules
pytest tests/unit/test_swarm_core.py tests/unit/test_coordinator.py -v
pytest tests/unit/test_contract_rejection_short_circuits_to_rejected_governance.py -v
pytest tests/integration/test_external_sync_api.py -v
```

### Test Coverage

| Module | Test Files | Focus |
|---|---|---|
| `swarm/core` | `test_agent.py`, `test_event.py`, `test_registry.py`, `test_coordinator.py`, `test_swarm_core.py`, `test_completion.py`, `test_timeline.py` | Runtime primitives, replay safety, completion tracking |
| `swarm/domain` | `test_requirement_agent.py`, `test_strategy_selection.py`, `test_supplier_agent.py`, `test_evaluation_agent.py`, `test_negotiation_agent.py`, `test_weighted_evaluation.py`, `test_decision_agent.py`, `test_decision_explanation.py`, `test_supplier_memory.py`, `test_outcome_agent.py`, `test_supplier_intelligence.py`, `test_supplier_history_evaluation.py` | Domain agent correctness, strategy selection, memory feedback |
| `swarm/integrations` | `test_governance_flow.py`, `test_external_sync_api.py` | Contract pre-gate hardening, idempotency, connector factory |
| `core` | `test_scoring.py`, `test_bilateral.py`, `test_ledger.py` | Auction engine, hash chain integrity |
| `Integration` | `test_api.py`, `test_load.py`, `test_feedback_cycle.py` | End-to-end API, load testing, feedback loop |

### Terminal Functional & Resilience Suite

```bash
chmod +x test_suite.sh
./test_suite.sh
```

Covers 20 functional checks + 13 resilience checks (concurrent auctions, bartering stress, DB persistence, chain integrity under load).

### Load Testing

```bash
python stress_test.py
```

---

## Performance

| Metric | Value | Notes |
|---|---|---|
| Auction throughput | 15–30 req/s | End-to-end sealed-bid + bartering, 5 suppliers |
| Swarm dispatch latency | ~50ms | Deterministic path, no LLM, 5 suppliers |
| p50 latency (MockLLM) | ~150ms | Core auction with mock backend |
| p99 latency | ~600ms | Under 20-concurrent load |
| Policy validation | <1ms | Deterministic Python, zero LLM |
| Ledger append | ~5ms | Asyncpg, append-only |
| Hash chain verification | ~50ms/1000 events | Cryptographic integrity |
| Timeline projection | ~10ms/100 items | Pure read, no external calls |

### Scaling Dimensions

| Bottleneck | Mitigation |
|---|---|
| LLM inference latency | Use vLLM with continuous batching; pre-compute supplier strategies |
| Bilateral bartering serialization | Shard by session_id; each auction is independent |
| Ledger read amplification | Verify chain asynchronously (background), not on every GET |
| Swarm state memory | `MAX_SWARM_STATES = 50` LRU; persistent storage is a future phase |

---

## Security

### Current Protections

1. **Input validation** — Pydantic v2 schemas reject malformed requests (HTTP 422)
2. **Material whitelist** — Only configured commodities are accepted
3. **Spend caps** — Hard budget limits enforced by PolicyEngine (no LLM override)
4. **Audit trail** — SHA-256 hash chain detects tampering
5. **Secret masking** — Timeline projection redacts `password`, `token`, `api_key`, etc.
6. **Idempotency** — `IdempotencyGuard` prevents duplicate external side effects
7. **Contract pre-gate** — Invalid/expired contracts short-circuit to REJECTED before risk assessment

### Production Hardening Checklist

- [ ] Add OAuth2 / API key authentication (FastAPI Depends + HTTPBearer)
- [ ] Add rate limiting (slowapi or nginx limit_req)
- [ ] Terminate TLS at reverse proxy
- [ ] Add buyer organization isolation (multi-tenant session_id prefixing)
- [ ] Rotate `PROCUREMENT_CONNECTOR_*` credentials via secrets manager
- [ ] Enable PostgreSQL SSL/TLS for managed instances

See `SECURITY.md` for responsible disclosure policy.

---

## Determinism vs Adaptation

This is a critical distinction for production credibility:

| Aspect | Determinism | Adaptation |
|---|---|---|
| **When** | Per-execution | Between runs |
| **Trigger** | Input + policy version | Explicit `POST /policies/promote` |
| **Storage** | In-memory state, event log | Event store (traces + feedback) |
| **Risk** | None — pure functions | Bounded — validated candidates only |
| **Audit Trail** | Timeline projection | Policy version history + promotion log |

**Execution remains deterministic for a given input + policy version.** The same requirement dispatched against the same active policy will always produce the same decision, timeline, and artifacts.

**Adaptation happens offline and between runs:**

1. The replay engine evaluates candidate policies against historical traces.
2. An operator (or automated safety-check script) reviews the results.
3. An explicit `POST /policies/promote` call activates the new version.
4. All subsequent executions use the new policy — deterministically, reproducibly.

No learning, no LLM call, and no policy mutation occurs during live procurement execution.

---

## Replay-Based Evaluation

The replay engine provides a **sandboxed, deterministic** environment for evaluating candidate policies:

- **Historical traces** (events + artifacts + feedback) are loaded from the event store.
- The **replay engine** re-executes each trace using the candidate's thresholds, weights, and routing rules — **no external calls, no state mutations**.
- The **`IdempotencyGuard`** prevents any external side effects from being re-executed.
- An **overrides-applied flag** prevents double application of contextual parameter overrides across nested replay contexts.
- **Artifact lineage** from the original run is reused for timeline projection — the replayed event stream is projected into the same causal structure.

**Evaluation metric (hybrid objective):**

```
metric = 0.5 × feedback_success_rate
       + 0.3 × feedback_outcome_score
       + 0.2 × decision_stability
```

Where `decision_stability` is the fraction of replays that reproduce the originally persisted chosen supplier.

---

## Troubleshooting

### Connection refused to PostgreSQL

```bash
docker compose ps
docker compose logs postgres
docker compose down && docker compose up -d
```

### ModuleNotFoundError: No module named 'core'

```bash
pip install -e .
# Do NOT use sys.path hacks. The package must be installed.
```

### ImportError: cannot import name 'Vector' from 'pgvector'

Ensure Docker image is `pgvector/pgvector:pg16`, not plain `postgres:16`.

### Tests fail with connection refused

```bash
docker compose up -d
pytest tests/ -v
```

### Slow Docker build with LLM

The production Dockerfile excludes PyTorch by default. Change to `pip install -e ".[llm]"` for real inference. Expect 3–5 min build time due to torch (~2GB).

---

## Changelog

### [Unreleased] — Adaptive Intelligence Layer (Steps 20–25)

- **Policy Learning**: Offline replay-based candidate generation and evaluation from historical traces + feedback
- **Adaptive Routing** (Step 24): Context-aware strategy selection via routing rules with priority-based conflict resolution
- **Contextual Parameter Adaptation** (Step 25): Runtime adjustment of thresholds and weights bounded by delta caps, clamping, and normalization
- **Policy & Learning APIs**: `POST /policies/learn`, `POST /policies/promote`, `GET /policies/active`
- **Idempotency Guard for Overrides**: Prevents double application of parameter overrides in nested execution contexts
- **Safety Validation**: Zero-delta rejection, total delta cap (0.15), MAX_PARAMS_PER_RULE=2 enforcement
- **47+ unit tests** covering all adaptive intelligence components

### [Unreleased] — Runtime Configuration & Governance Hardening

- **Connector Factory**: `ConnectorConfig` + `build_connector()` for runtime-selectable ERP adapters (mock → supplier_api → sap / oracle / coupa)
- **Contract Pre-Gate Hardening**: `ContractValidationAgent` is a hard gate — `ContractRejected` → REJECTED governance with no risk assessment, no PO, no execution
- **Timeline Projection**: `GET /swarm/timeline/{request_id}` — deterministic, read-only, causally-ordered merge of events + artifacts with phase markers and secret masking
- **Idempotency Layer**: `IdempotencyGuard` keyed by `(decision_id, action)` prevents duplicate external calls
- **External Call Artifacts**: Every outbound ERP call recorded with `idempotency_key`, `request_payload`, `response_payload`

### v0.8 — Enterprise Integration Layer

- `BaseConnector` port + `MockConnector` / `SupplierAPIConnector` / `SAPConnector` / `OracleConnector` / `CoupaConnector`
- `ExternalCallArtifact` audit trail
- `ContractValidationAgent` between decision and risk
- `POST /swarm/{id}/sync` for external reconciliation

### v0.7 — Execution & Procurement Operations

- `PurchaseOrderAgent` + `ExecutionTrackingAgent`
- `POST /swarm/{id}/execute` + `GET /swarm/order/{id}` + `GET /swarm/execution/{id}`
- Full lifecycle: requirement → ... → order → tracking → delivery

### v0.6 — Governance + Risk-Aware Procurement

- `RiskAssessmentAgent` (financial / delivery / quality / carbon)
- `GovernanceAgent` with `GovernancePolicy` (`STANDARD_POLICY`, `STRICT_POLICY`)
- `ApprovalAgent` with `ExecutionAuthorizationArtifact`
- `POST /swarm/{id}/approve`

### v0.5 — Feedback Intelligence

- `OutcomeAgent` + `SupplierIntelligenceAgent`
- `SupplierMemoryStore` with deterministic performance folding
- `POST /swarm/{id}/outcome` + `GET /swarm/supplier/{id}/performance`

### v0.4 — Strategy Intelligence

- `StrategyAgent` with `cost_optimized` / `balanced` / `low_carbon` strategies
- `DecisionExplanationArtifact` (auditable, deterministic reasoning)

### v0.3 — Parallel Multi-Agent Execution

- Per-supplier parallel evaluation
- `CompletionTracker` with `expect_artifact()` / `complete_artifact()`
- Full execution trace API

### v0.1 — Swarm Foundation

- `EventBus`, `BaseAgent`, `SwarmState`, `Artifact`, `Capability`, `AgentRegistry`
- Core auction engine: CNP sealed-bid, bilateral bartering, hash-chained ledger

---

## Contributing

Fork the repository → Create a feature branch → Install dev dependencies → Run quality checks → Submit a Pull Request.

```bash
git checkout -b feature/your-feature
pip install -e ".[dev]"
ruff check . && mypy core/ api/ swarm/ --ignore-missing-imports
pytest tests/ -v
```

### Code Standards

| Standard | Requirement |
|---|---|
| Type hints | All public functions must have return type annotations |
| Async | All I/O-bound operations use `async`/`await` |
| Pydantic | All data schemas use Pydantic v2 |
| Logging | Use `structlog`, never `print()` |
| Tests | All new features require unit tests; API changes require integration tests |
| Determinism | No random without seed; no LLM in control-path agents |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

<div align="center">

> Built with determinism, auditability, and game theory in mind.
>
> *"The swarm does not think. The swarm decides — deterministically, defensibly, and with a complete paper trail."*

</div>