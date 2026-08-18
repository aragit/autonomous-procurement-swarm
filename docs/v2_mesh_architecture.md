# V2 Mesh Architecture: Neuro-Symbolic Procurement Mesh

> **Version:** v2.0 (distributed Ray mesh runtime)
> **Status:** Active
> **Philosophy:** Neural components propose. Symbolic kernel disposes. Distributed execution scales. Audit trail persists.

---

## 1. Overview

The V2 Procurement Mesh replaces the single-node asyncio pipeline (V1) with a
distributed Ray actor system built around a **typed blackboard** with
capability-scoped ACLs.  Four elastic agent archetypes (Scout, Evaluator,
Negotiator, Buyer) coordinate through channels enforced by a singleton
SafetyKernelActor.  A Neuro-Symbolic bridge optionally augments Scout and
Negotiator with schema-constrained LLM generation, validated by the kernel
with an auto-correction retry loop and a deterministic fallback.

## 2. Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           API GATEWAY (FastAPI v2)                        │
│  POST /v2/procurement/run   GET /v2/procurement/{trace_id}/status       │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                  ┌──────────────▼──────────────┐
                  │     RayMeshRuntime          │
                  │  (api/v2/runtime.py)        │
                  │  Lazily connects to Ray     │
                  └──────────────┬──────────────┘
                                 │
    ┌────────────────────────────┼─────────────────────────────────┐
    │  Ray Head Node (:8265)     │                                 │
    │                           │                                 │
    │  ┌──────────────────────┐ │  ┌─────────────────────────────┐ │
    │  │ DistributedBlackboard│ │  │ SafetyKernelActor (singleton)│ │
    │  │  (mesh/blackboard.py)│ │  │  (mesh/actors/base.py)       │ │
    │  └──────────┬───────────┘ │  └─────────────┬───────────────┘ │
    │             │              │                │                 │
    │  ┌──────────▼──────────┐   │  ┌─────────────▼───────────────┐ │
    │  │   Channel ACLs      │   │  │  Symbolic Validation        │ │
    │  │ (mesh/channels.py)  │   │  │  - Budget clamping          │ │
    │  └─────────────────────┘   │  │  - Lead-time bounds         │ │
    │                             │  │  - ESG material whitelist   │ │
    │                             │  │  - Payment term whitelist   │ │
    │                             │  │  - Policy enforcement       │ │
    │                             │  └─────────────┬───────────────┘ │
    │                             │                │                 │
    │  ┌──────────┬──────────┬───┐ │  ┌─────────────▼───────────────┐ │
    │  │  Scout   │Evaluator │   │ │  │ Neuro-Symbolic Bridge       │ │
    │  │  Actors  │ Actors   │   │ │  │  (mesh/neuro/)              │ │
    │  │  (N,     │ (N,      │   │ │  │  - Schema-constrained LLM   │ │
    │  │  elastic)│  elastic)│   │ │  │  - Retry loop + fallback    │ │
    │  └──────────┴──────────┴───┘ │  └─────────────────────────────┘ │
    │                                  │                              │
    │  ┌──────────┬──────────┬────────┐ │                              │
    │  │Negotiator│  Buyer   │        │ │                              │
    │  │  Actors  │  Actor   │        │ │                              │
    │  │ (N,      │ (singleton)      │ │                              │
    │  │ elastic) │                 │ │                              │
    │  └──────────┴──────────┴────────┘ │                              │
    └──────────────────────────────────────────────────────────────────┘
          │          │          │
   Channel │          │          │
   Flow    ▼          ▼          ▼
  REQUIREMENT → DISCOVERY → SCORE/RISK → DEAL → DECISION

┌──────────────────────────────────────────────────────────────────────────┐
│                         Optional LLM Backend                            │
│    Ollama (11434) or vLLM (8000) — only needed when neuro enabled       │
└──────────────────────────────────────────────────────────────────────────┘
```

### Channel Flow (Typed Blackboard)

```
kernel.write(REQUIREMENT)      ──►  Kernel writes the requirement
    ├── ScoutActor reads REQUIREMENT
    │   └── writes DISCOVERY   ──►  Supplier candidates
    ├── EvaluatorActor reads DISCOVERY
    │   ├── writes SCORE       ──►  MCDA price/lead-time/ESG/reliability
    │   └── writes RISK        ──►  Risk assessment
    ├── NegotiatorActor reads SCORE
    │   └── writes DEAL        ──►  Negotiated quotes (neuro or deterministic)
    └── BuyerActor reads DEAL, RISK, SCORE
        └── writes DECISION    ──►  Deterministic MCDA winner (singleton)
```

---

## 3. The Four Agent Archetypes

| Archetype       | Count    | Neural? | Blackboard Channels                          |
| --------------- | -------- | ------- | -------------------------------------------- |
| **Scout**       | N (elastic) | optional | `read: REQUIREMENT` → `write: DISCOVERY`    |
| **Evaluator**   | N (elastic) | no      | `read: DISCOVERY` → `write: SCORE, RISK`     |
| **Negotiator**  | N (elastic) | optional | `read: SCORE` → `write: DEAL`               |
| **Buyer**       | 1 (singleton) | no   | `read: DEAL, RISK, SCORE` → `write: DECISION` |

Every agent is a **Ray actor**. The Buyer is a singleton per procurement run
that performs deterministic Multi-Criteria Decision Analysis (MCDA) over the
distributed candidate pool.

### Capability-Scoped ACLs

| Channel       | Actors that can read                         | Actors that can write  |
| ------------- | -------------------------------------------- | ---------------------- |
| `REQUIREMENT` | Scout, Evaluator, Negotiator, Buyer, Kernel  | Kernel only            |
| `DISCOVERY`   | Evaluator, Negotiator, Buyer, Kernel         | Scout only             |
| `SCORE`       | Negotiator, Buyer, Kernel                    | Evaluator only         |
| `RISK`        | Buyer, Kernel                                | Evaluator only         |
| `DEAL`        | Buyer, Kernel                                | Negotiator only        |
| `DECISION`    | Kernel                                       | Buyer only             |

These ACLs are enforced at the blackboard level in
`mesh/channels.py:channel_acl.py`. An agent attempting to read or write a
channel outside its capability raises `PermissionError`.

---

## 4. Procurement Request Lifecycle

### Phase 1 — Requirement

1. Client calls `POST /v2/procurement/run` with a requirement payload.
2. `api/v2/router.py` generates a `trace_id` and delegates to the active
   `MeshRuntime`.
3. `RayMeshRuntime.run_procurement()` creates a `ProcurementCluster` with an
   isolated blackboard named `v2_bb_{trace_id}` and kernel
   `v2_kernel_{trace_id}`.
4. The kernel writes the requirement to the `REQUIREMENT` channel.

### Phase 2 — Scout Discovery

1. A pool of `ScoutActor` instances (default 3, configurable via
   `MESH_N_SCOUTS`) each call `step()`.
2. Each Scout reads the `REQUIREMENT` channel from the blackboard.
3. If neuro is enabled (`enable_neuro: true` on the requirement), the Scout
   uses the `NeuroSymbolicBridge` with a schema-constrained LLM backend.
   The LLM generates a `ScoutProposal` (supplier list with price/ESG/reliability).
4. The proposal is sent to the `SafetyKernelActor.validate()` for symbolic
   validation. If rejected, the bridge retries up to `neuro_max_retries` (default 3).
5. On exhaustion of retries or when neuro is disabled, the Scout falls back
   to the deterministic supplier pool.
6. Validated discoveries are written to the `DISCOVERY` channel.

### Phase 3 — Evaluation

1. A pool of `EvaluatorActor` instances (default 3) each call `step()`.
2. Each Evaluator reads `DISCOVERY` entries from the blackboard.
3. For each supplier, the Evaluator computes:
   - **Score**: deterministic MCDA scoring across price, lead time, ESG,
     and reliability (using `core/evaluator/scoring.py`).
   - **Risk**: deterministic risk classification (using
     `core/risk/`).
4. Results are written to `SCORE` and `RISK` channels.

### Phase 4 — Negotiation

1. A pool of `NegotiatorActor` instances (default 2) each call `step()`.
2. Each Negotiator reads `SCORE` entries to identify winning candidates.
3. If neuro is enabled, the Negotiator generates `NegotiatorQuote` proposals
   via the LLM backend. The kernel validates each quote's price against the
   budget; if over-budget, the bridge auto-corrects and retries.
4. On neuro exhaustion, deterministic fallback quotes are written.
5. Validated quotes are written to the `DEAL` channel.

### Phase 5 — Decision (Deterministic MCDA)

1. The singleton `BuyerActor` calls `step()`.
2. It reads `DEAL`, `RISK`, and `SCORE` channels from the blackboard.
3. It builds a candidate pool and computes a composite score using weighted
   MCDA: price (40%), lead time (25%), ESG (20%), reliability (15%), with a
   risk penalty multiplier.
4. Governance policy filtering (budget ceiling, risk threshold) is applied.
5. The deterministic argmax winner is written to the `DECISION` channel.

### Phase 6 — Kernel Validation & Completion

1. The `SafetyKernelActor` writes the final decision validation to the
   blackboard.
2. The runtime returns a `ProcurementRunResponse` with the trace ID,
   correlation ID, status, and decision data.
3. The blackboard is kept alive for status queries until shutdown.

---

## 5. Neuro-Symbolic Bridge

The bridge (`mesh/neuro/bridge.py`) implements a retry loop:

```
1. LLM generates a schema-constrained proposal
2. SafetyKernelActor validates the proposal
3. If validation fails → auto-correct prompt, retry (up to max_retries)
4. If all retries fail → fall back to deterministic path
5. If validation passes → write to blackboard
```

Key components:
- `LLMConfig` — base URL, model name, timeout
- `StructuredBackend` / `OpenAICompatibleBackend` — LLM client
- `MockNeuroBackend` — deterministic test backend
- `NeuroSymbolicBridge` — retry loop orchestrator
- `kernel.symbolic_validate()` — deterministic guardrails

---

## 6. Docker Deployment

### Quick Start

```bash
# 1. Build and start the entire mesh
docker compose -f docker-compose.mesh.yml up --build

# 2. The API is available at http://localhost:8000
#    Ray dashboard at http://localhost:8265

# 3. Submit a procurement request
curl -X POST http://localhost:8000/v2/procurement/run \
  -H "Content-Type: application/json" \
  -d '{
    "material": "aluminum",
    "quantity": 1000,
    "budget": 500000,
    "target_lead_time_days": 30,
    "enable_neuro": false
  }'

# 4. Check status
curl http://localhost:8000/v2/procurement/RUN-XXXXXXXXXXXX/status
```

### Environment Variables

| Variable             | Default | Description                              |
| -------------------- | ------- | ---------------------------------------- |
| `MESH_ROLE`          | `api`   | Container role: `head`, `worker`, `api`  |
| `MESH_N_SCOUTS`      | `3`     | Number of Scout actors                   |
| `MESH_N_EVALUATORS`  | `3`     | Number of Evaluator actors              |
| `MESH_N_NEGOTIATORS` | `2`     | Number of Negotiator actors             |
| `RAY_ADDRESS`        | —       | Ray cluster address (auto for workers)   |
| `HOST`               | `0.0.0.0` | Server bind host                        |
| `PORT`               | `8000`  | Server bind port                         |
| `LOG_LEVEL`          | `info`  | Uvicorn log level                        |

### Enabling the Neural Backend

To enable the neuro-symbolic bridge, set `enable_neuro: true` on the
requirement and provide a reachable LLM endpoint:

```bash
# With Ollama (uncomment the ollama service in docker-compose.mesh.yml)
curl -X POST http://localhost:8000/v2/procurement/run \
  -H "Content-Type: application/json" \
  -d '{
    "material": "steel",
    "quantity": 5000,
    "budget": 750000,
    "enable_neuro": true,
    "neuro_llm_base_url": "http://ollama:11434/v1",
    "neuro_llm_model": "gemma-2b-it"
  }'
```

---

## 7. API Endpoints

### `POST /v2/procurement/run`

Submit a new procurement requirement to the mesh.

**Request Body:**
```json
{
  "trace_id": "RUN-ABC123" (optional, auto-generated),
  "requirement": {
    "material": "aluminum",
    "quantity": 1000,
    "budget": 500000.0,
    "target_lead_time_days": 30,
    "enable_neuro": false,
    "neuro_llm_base_url": "http://ollama:11434/v1",
    "neuro_llm_model": "gemma-2b-it"
  }
}
```

**Response:**
```json
{
  "trace_id": "RUN-ABC123",
  "correlation_id": "RUN-ABC123",
  "status": "completed",
  "decision": { ... },
  "buyer_result": { ... }
}
```

### `GET /v2/procurement/{trace_id}/status`

Get the live blackboard snapshot and stats for a procurement run.

**Response:**
```json
{
  "trace_id": "RUN-ABC123",
  "correlation_id": "RUN-ABC123",
  "status": "completed",
  "decision": { ... },
  "channels": {
    "requirement": 1,
    "discovery": 3,
    "score": 5,
    "risk": 5,
    "deal": 5,
    "decision": 1
  },
  "blackboard_stats": { ... }
}
```

---

## 8. V1 vs V2 Comparison

| Aspect                 | V1 (Asyncio)                     | V2 (Ray Mesh)                          |
| ---------------------- | -------------------------------- | -------------------------------------- |
| Runtime                | Single-process asyncio           | Distributed Ray actors                 |
| Communication          | Global EventBus (broadcast)      | Typed Blackboard (capability-scoped)   |
| Scalability            | Single node                      | Horizontal (Ray head + workers)        |
| Agent model            | 14 hardcoded domain agents       | 4 elastic archetypes + kernel services |
| Safety                 | Deterministic agents only        | Symbolic kernel validates all neural output |
| LLM integration         | Read-only cognitive analysis     | Schema-constrained generation + validation |
| Audit trail            | Artifact lineage + hash chain    | Blackboard snapshots + event log       |
| Decision making        | Central argmax                   | Distributed discovery + deterministic MCDA |
| Fault tolerance        | None (single process crashes)    | Ray actor restarts, failure traces     |
| Deployment             | `api.main:app` (uvicorn)          | `api.v2:app` (uvicorn) + Ray cluster    |

---

## 9. File Reference

| Component              | Path                             |
| ---------------------- | -------------------------------- |
| Docker entrypoint      | `mesh/entrypoint.py`              |
| Ray cluster config     | `mesh/cluster.py` (`MeshConfig`)  |
| Blackboard actor       | `mesh/blackboard.py`              |
| Channel ACLs           | `mesh/channels.py`                |
| Safety Kernel actor    | `mesh/actors/base.py`             |
| Scout actor             | `mesh/actors/scout.py`            |
| Evaluator actor        | `mesh/actors/evaluator.py`        |
| Negotiator actor       | `mesh/actors/negotiator.py`       |
| Buyer actor (MCDA)     | `mesh/actors/buyer.py`            |
| Neuro bridge           | `mesh/neuro/bridge.py`            |
| Neuro schemas          | `mesh/neuro/schemas.py`           |
| Neuro backends         | `mesh/neuro/backend.py`           |
| Neuro kernel           | `mesh/neuro/kernel.py`            |
| V2 API app             | `api/v2/__init__.py`              |
| V2 API router          | `api/v2/router.py`                |
| V2 API runtime         | `api/v2/runtime.py`               |
| V2 API models          | `api/v2/models.py`                |
