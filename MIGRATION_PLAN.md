# Migration Plan: From Deterministic Procurement Pipeline to Neuro-Symbolic Procurement Mesh

> **Version:** v1.0 | **Target:** Type-6 Neuro[Symbolic] with mandatory Type-2 safety invariants | **Estimated Duration:** 16–20 weeks

---

## Executive Summary

Your current architecture is a **production-grade Type-2 Symbolic[Neuro] event-driven multi-agent pipeline** with a closed-loop policy learner. The next evolution is **not** to make it a "classical swarm" (which would sacrifice auditability) but to transform it into a **Neuro-Symbolic Procurement Mesh** — a Type-6 architecture where:

- **Neural layers** handle perception, communication, and exploration (LLM/SLM)
- **Symbolic layers** enforce safety, governance, and execution (your existing deterministic tail)
- **Stigmergic environment** replaces the rigid pipeline with an emergent coordination substrate
- **Distributed actor model** (Ray) enables horizontal scaling beyond the single-node event bus

This plan preserves every hard-won safety invariant (hash chains, idempotency, replay safety, contract pre-gates) while adding genuine emergent behavior.

---

## 1. Honest Current-State Assessment

### What You Have (Strengths)

| Capability | Maturity | Notes |
|---|---|---|
| **Deterministic control plane** | ⭐⭐⭐⭐⭐ | 14 agents, zero LLM in governance path |
| **Closed-loop policy learning** | ⭐⭐⭐⭐☆ | Grid search + replay + hybrid objective + atomic promotion |
| **Artifact lineage + audit trail** | ⭐⭐⭐⭐⭐ | `parent_ids`, SHA-256 policy signatures, hash-chained ledger |
| **LLM observability layer** | ⭐⭐⭐⭐☆ | Trust gating, drift detection, stability monitoring, fallback chains |
| **Replay engine** | ⭐⭐⭐⭐⭐ | Isolated storage, decision stability metrics, trace reconstruction |
| **Enterprise integration** | ⭐⭐⭐⭐☆ | Connector factory, idempotency guard, external call artifacts |
| **Event-driven MAS framework** | ⭐⭐⭐⭐⭐ | `BaseAgent`, `EventBus`, `Capability`, `AgentRegistry` |

### What You Lack (Gaps vs. Target)

| Gap | Severity | Why It Blocks the Target |
|---|---|---|
| **Centralized pipeline topology** | 🔴 High | `wiring.py` hardcodes a DAG. No emergence possible. |
| **Global event bus** | 🔴 High | All agents see all events. No local interaction = no swarm dynamics. |
| **Single-node execution** | 🟡 Medium | `asyncio` event loop on one process. Cannot scale to 1000+ suppliers. |
| **No stigmergy** | 🔴 High | `SwarmState` is a key-value store, not an environmental field. |
| **Homogeneous agent design** | 🟡 Medium | 14 specialized agents cannot self-organize. |
| **No distributed consensus** | 🔴 High | `DecisionAgent` runs `argmax` centrally. No quorum, no emergence. |
| **LLM is read-only only** | 🟡 Medium | `SupplierAnalysisLLMAgent` is cognitive-only. No generative negotiation. |
| **No actor-model isolation** | 🟡 Medium | Agents share memory space. Fault in one agent kills the swarm. |

---

## 2. Target Architecture: Neuro-Symbolic Procurement Mesh (NSPM)

### 2.1 Architectural Tenets

1. **Safety-First Neuro-Symbolism**: The symbolic layer is the "kernel" — it can veto any neural output. The neural layer is the "cortex" — it suggests, explores, and communicates but never executes.
2. **Stigmergic Coordination**: Agents coordinate via a shared environmental field (the Mesh), not via direct messages or a global bus.
3. **Emergent Decision Boundaries**: No central `DecisionAgent`. Decisions emerge from distributed scoring fields and quorum consensus.
4. **Homogeneous Archetypes**: 4 agent archetypes with configurable policies, not 14 hardcoded specialists.
5. **Actor-Model Distribution**: Each agent is a Ray actor. The Mesh is a Ray distributed object store.

### 2.2 Layer Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NEURAL COGNITIVE LAYER                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Perception │  │  Generation │  │   Memory    │  │   Exploration       │ │
│  │   (SLM)     │  │   (LLM)     │  │ (Embedding) │  │   (MARL Policy)     │ │
│  │             │  │             │  │             │  │                     │ │
│  │ • Supplier  │  │ • Negotiate │  │ • Semantic  │  │ • Bid exploration   │ │
│  │   document  │  │   language  │  │   supplier  │  │ • Strategy search   │ │
│  │   parsing   │  │ • Contract  │  │   memory    │  │ • Counter-offer     │ │
│  │ • Market    │  │   drafting  │  │ • Cross-run │  │   generation        │ │
│  │   sentiment │  │ • Risk      │  │   retrieval │  │                     │ │
│  │             │  │   narrative │  │             │  │                     │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                │                    │            │
│         └────────────────┴────────────────┴────────────────────┘            │
│                                    │                                        │
│                         ┌──────────▼──────────┐                             │
│                         │  Structured Output  │                             │
│                         │  (JSON Schema /     │                             │
│                         │   Pydantic v2)      │                             │
│                         └──────────┬──────────┘                             │
└────────────────────────────────────┼────────────────────────────────────────┘
                                     │
                         ┌───────────▼────────────┐
                         │   SYMBOLIC KERNEL      │
                         │   (Deterministic)      │
                         │                        │
                         │ • Validate neural out  │
                         │ • Enforce guardrails   │
                         │ • Compute scores       │
                         │ • Apply policies       │
                         │ • Authorize execution  │
                         │ • Write audit trail    │
                         └───────────┬────────────┘
                                     │
┌────────────────────────────────────┼────────────────────────────────────────┐
│                         STIGMERGIC MESH (Ray Distributed Store)             │
│                                                                             │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐            │
│   │  Scout  │◄───►│  Field  │◄───►│  Field  │◄───►│  Buyer  │            │
│   │  Actor  │     │  Node   │     │  Node   │     │  Actor  │            │
│   └────┬────┘     └─────────┘     └─────────┘     └────┬────┘            │
│        │                                               │                   │
│        │         ┌─────────┐     ┌─────────┐          │                   │
│        └────────►│  Field  │◄───►│  Field  │◄─────────┘                   │
│                  │  Node   │     │  Node   │                              │
│                  └─────────┘     └─────────┘                              │
│                                                                             │
│   Each "Field Node" is a typed pheromone trace:                            │
│   • `discovery_pheromone` (supplier_id, strength, decay_rate)              │
│   • `score_pheromone` (supplier_id, composite_score, confidence)           │
│   • `deal_pheromone` (supplier_id, proposed_price, expiry)                 │
│   • `risk_pheromone` (supplier_id, risk_level, timestamp)                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 The Four Agent Archetypes

| Archetype | Role | Neural? | Symbolic? | Count |
|---|---|---|---|---|
| **Scout** | Discovers suppliers, deposits `discovery_pheromone` | SLM for web research | Deterministic filtering | N (elastic) |
| **Evaluator** | Senses `discovery_pheromone`, evaluates, deposits `score_pheromone` | SLM for supplier doc parsing | Deterministic MCDA scoring | N (elastic) |
| **Negotiator** | Senses `score_pheromone`, engages pairwise, deposits `deal_pheromone` | LLM for language generation | Deterministic floor/ceiling guardrails | N (elastic) |
| **Buyer** | Senses `deal_pheromone`, follows strongest trail, reaches quorum | None | Deterministic quorum consensus + policy enforcement | 1 per procurement |

**Governance, Risk, Approval, Execution** are **not agents** in the target architecture. They are **kernel services** that every archetype calls via a mandatory `safety_check()` hook before any `act()`.

---

## 3. Migration Phases

### Phase 0: Foundation Hardening (Week 1–2)
**Goal:** Prepare the codebase for structural surgery. No behavioral changes.

#### 3.0.1 Repository Restructure
```
autonomous-procurement-swarm/
├── kernel/              # NEW: symbolic safety layer (extracted from swarm/domain)
│   ├── governance/
│   ├── risk/
│   ├── approval/
│   ├── execution/
│   └── contracts/
├── mesh/                # NEW: stigmergic environment + Ray actors
│   ├── environment.py
│   ├── fields/
│   ├── actors/
│   └── consensus/
├── cognitive/           # NEW: neural layer (moved from swarm/utils/llm_*)
│   ├── perception/
│   ├── generation/
│   ├── memory/
│   └── exploration/
├── learning/            # EXPAND: existing swarm/learning → full MARL
│   ├── policy/          # existing
│   ├── replay/          # moved from swarm/simulation
│   ├── rewards/         # NEW
│   └── marl/            # NEW
├── legacy/              # TEMP: old swarm/ during migration
│   ├── core/
│   ├── domain/
│   └── integrations/
└── api/                 # REFACTOR: mount both legacy and new routes
```

**Files to create:**
- `kernel/__init__.py` — safety kernel facade
- `mesh/__init__.py` — mesh environment facade
- `cognitive/__init__.py` — cognitive layer facade
- `legacy/` — git mv swarm/ legacy/ (preserves history)

**Files to modify:**
- `pyproject.toml` — add `ray[default]`, `vllm`, `transformers`
- `docker-compose.yml` — add Ray head + worker services
- `.github/workflows/ci.yml` — add Ray cluster spin-up for integration tests

#### 3.0.2 Interface Contracts
Define the **Neural-Symbolic Bridge Protocol (NSBP)**:

```python
# mesh/protocol.py
from pydantic import BaseModel

class NeuralProposal(BaseModel):
    """Any output from the cognitive layer MUST pass through this schema."""
    proposal_id: str
    archetype: Literal["scout", "evaluator", "negotiator"]
    payload: dict
    confidence: float  # 0.0–1.0
    structured: bool   # True if generated via constrained decoding

class SymbolicVerdict(BaseModel):
    """The kernel's response to every NeuralProposal."""
    proposal_id: str
    approved: bool
    reason: str
    clamped_payload: dict | None  # If approved but modified
    audit_hash: str  # SHA-256 of the full verdict
```

**Every neural output is validated by the kernel before it becomes a `MeshField` deposit.**

---

### Phase 1: The Mesh Environment (Week 3–5)
**Goal:** Replace `EventBus` + `SwarmState` with a stigmergic distributed environment.

#### 3.1.1 Ray-Based Field Store

```python
# mesh/environment.py
import ray
from ray.util.queue import Queue

@ray.remote
class MeshField:
    """A typed pheromone field. Agents deposit traces; agents sense locally."""

    def __init__(self, field_type: str, decay_rate: float = 0.05):
        self.field_type = field_type
        self.traces: list[dict] = []
        self.decay_rate = decay_rate

    def deposit(self, trace: dict) -> None:
        trace["timestamp"] = time.time()
        trace["strength"] = trace.get("strength", 1.0)
        self.traces.append(trace)
        self._decay()

    def sense_local(self, position: str, radius: int = 3) -> list[dict]:
        """Agents only see traces near their position."""
        self._decay()
        return [t for t in self.traces 
                if self._distance(position, t.get("position", "")) <= radius]

    def _decay(self) -> None:
        now = time.time()
        self.traces = [
            {**t, "strength": t["strength"] * (1 - self.decay_rate) ** (now - t["timestamp"])}
            for t in self.traces
            if t["strength"] > 0.01
        ]
```

**Key difference from `SwarmState`:**
- `SwarmState` is a global dictionary. Any agent can `find_artifacts(kind="quote")` and see everything.
- `MeshField` is local sensing. A `Negotiator` at position `"supplier_A"` cannot see deals at `"supplier_B"` unless it moves there.

#### 3.1.2 Migration of Existing Artifacts to Fields

| Old Artifact | New Field | Deposited By | Sensed By |
|---|---|---|---|
| `supplier_list` | `discovery_pheromone` | Scout | Evaluator |
| `evaluation` | `score_pheromone` | Evaluator | Negotiator |
| `quote` | `deal_pheromone` | Negotiator | Buyer |
| `risk_assessment` | `risk_pheromone` | Evaluator (mandatory) | Buyer |
| `decision` | **Removed** — emerges from quorum | — | — |
| `governance_decision` | `governance_pheromone` | Buyer (post-quorum) | Execution service |

#### 3.1.3 Position Model

Agents move through a **discrete supplier graph**:

```python
# mesh/topology.py
class SupplierGraph:
    """The procurement topology: nodes are suppliers, edges are material categories."""

    def __init__(self, suppliers: list[str], materials: list[str]):
        self.nodes = suppliers
        self.edges = self._build_material_edges(suppliers, materials)

    def neighbors(self, supplier_id: str, radius: int = 1) -> list[str]:
        """BFS neighbors within radius hops."""
        ...
```

A `Scout` starts at `"market"` and deposits `discovery_pheromone` on supplier nodes. An `Evaluator` must move to a supplier node to sense its discovery trace and deposit a score trace.

---

### Phase 2: Homogenize Agents into Archetypes (Week 6–8)
**Goal:** Collapse 14 specialized agents into 4 archetypes + kernel services.

#### 3.2.1 Archetype Base Class

```python
# mesh/actors/base.py
import ray

@ray.remote
class MeshActor:
    """Base class for all four archetypes."""

    def __init__(self, actor_id: str, archetype: str, 
                 cognitive_engine: str | None = None):
        self.actor_id = actor_id
        self.archetype = archetype
        self.position = "market"
        self.energy = 1.0
        self.cognitive_engine = cognitive_engine  # SLM/LLM model name or None

    async def step(self, mesh: MeshHandle) -> None:
        """The universal agent loop."""
        perception = await self.perceive(mesh)

        # MANDATORY: Safety kernel check before any action
        proposal = await self.reason(perception)
        verdict = await self.safety_check(proposal)
        if not verdict.approved:
            await self.log_rejection(verdict)
            return

        await self.act(mesh, verdict.clamped_payload or proposal.payload)

    async def safety_check(self, proposal: NeuralProposal) -> SymbolicVerdict:
        """RPC to the symbolic kernel. NEVER overridden."""
        kernel = ray.get_actor("safety_kernel")
        return await kernel.validate.remote(proposal)
```

#### 3.2.2 Mapping Old Agents to New Archetypes

| Old Agent(s) | New Archetype | Cognitive Role | Symbolic Role |
|---|---|---|---|
| `RequirementAgent` | **Buyer** (initialization) | Parses natural language requirements via SLM | Validates constraints against policy |
| `StrategyAgent` | **Buyer** (strategy selection) | SLM suggests strategy given market context | Symbolic rule selects from approved strategy set |
| `SupplierDiscoveryAgent` | **Scout** | SLM researches/web-scrapes suppliers | Deterministic filtering by compliance/certification |
| `EvaluationAgent` | **Evaluator** | SLM parses supplier documents/ESG reports | Deterministic MCDA scoring with strategy weights |
| `NegotiationAgent` | **Negotiator** | LLM generates counter-offer language | Deterministic floor/ceiling enforcement |
| `DecisionAgent` | **Removed** → `Buyer` quorum | None | Distributed consensus via `deal_pheromone` concentration |
| `RiskAssessmentAgent` | **Evaluator** (risk deposit) | None | Deterministic risk scoring, deposited as `risk_pheromone` |
| `GovernanceAgent` | **Kernel Service** | None | Policy rule engine |
| `ApprovalAgent` | **Kernel Service** | None | State-machine authorization |
| `PurchaseOrderAgent` | **Kernel Service** | None | Connector dispatch + idempotency |
| `ExecutionTrackingAgent` | **Kernel Service** | None | Lifecycle tracking |
| `OutcomeAgent` | **Kernel Service** | None | Payload normalization |
| `SupplierIntelligenceAgent` | **Evaluator** (memory update) | Embedding model for supplier similarity | Rolling average updates |
| `ContractValidationAgent` | **Kernel Service** | SLM for contract clause extraction | Boolean compliance validation |

#### 3.2.3 The Buyer Quorum Consensus (Replacing DecisionAgent)

```python
# mesh/consensus/quorum.py
class QuorumConsensus:
    """Emergent decision: the buyer follows the strongest deal pheromone trail."""

    def __init__(self, min_quorum: int = 3, confidence_threshold: float = 0.7):
        self.min_quorum = min_quorum
        self.confidence_threshold = confidence_threshold

    def resolve(self, deal_field: MeshField) -> dict | None:
        """Returns the winning supplier when quorum is reached."""
        traces = deal_field.sense_local("market", radius=999)  # Buyer sees all

        # Group by supplier, sum pheromone strength
        concentrations = defaultdict(float)
        for trace in traces:
            concentrations[trace["supplier_id"]] += trace["strength"]

        if len(concentrations) < self.min_quorum:
            return None  # Not enough distinct offers

        winner = max(concentrations, key=concentrations.get)
        total_strength = sum(concentrations.values())
        confidence = concentrations[winner] / total_strength if total_strength > 0 else 0

        if confidence < self.confidence_threshold:
            return None  # Too close to call — extend negotiation

        return {
            "supplier_id": winner,
            "confidence": confidence,
            "quorum_size": len(concentrations),
            "method": "stigmergic_quorum"
        }
```

**Why this is better than `argmax`:**
- The decision emerges from the collective behavior of all negotiators, not a central agent.
- If two suppliers are close, the quorum returns `None` and the system naturally extends negotiation (more `deal_pheromone` deposits).
- The confidence metric is observable and auditable.

---

### Phase 3: Neural-Symbolic Bridge (Week 9–11)
**Goal:** Make the cognitive layer production-safe with structured generation and deterministic validation.

#### 3.3.1 Constrained LLM Generation

Replace open-ended LLM prompts with **JSON Schema-constrained generation**:

```python
# cognitive/generation/negotiation.py
from outlines import models, generate

class NegotiationGenerator:
    """LLM counter-offer generation with 100% schema compliance."""

    schema = {
        "type": "object",
        "properties": {
            "proposed_price": {"type": "number", "minimum": 0, "maximum": 1000000},
            "proposed_lead_time_days": {"type": "integer", "minimum": 1, "maximum": 365},
            "payment_terms": {"enum": ["net_30", "net_60", "cod", "letter_of_credit"]},
            "justification": {"type": "string", "maxLength": 500},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["proposed_price", "payment_terms", "confidence"]
    }

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        self.model = models.transformers(model_name)
        self.generator = generate.json(self.model, self.schema)

    async def generate_counter(self, context: dict) -> dict:
        prompt = self._build_prompt(context)
        return self.generator(prompt)
```

> **Tool:** Use [Outlines](https://github.com/dottxt-ai/outlines) or [lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer) for guaranteed schema compliance. This eliminates hallucinated payment terms or negative prices.

#### 3.3.2 Deterministic Validation Layer

Every neural output runs through the **Symbolic Validator** before becoming a proposal:

```python
# kernel/validator.py
class SymbolicValidator:
    """Deterministic guardrail: rejects or clamps any neural output."""

    def validate(self, neural_output: dict, archetype: str) -> SymbolicVerdict:
        checks = [
            self._check_price_bounds(neural_output, archetype),
            self._check_lead_time_bounds(neural_output),
            self._check_payment_terms_whitelist(neural_output),
            self._check_confidence_minimum(neural_output, archetype),
            self._check_material_whitelist(neural_output),
        ]

        failures = [c for c in checks if not c.passed]
        if failures:
            return SymbolicVerdict(
                approved=False,
                reason=f"VALIDATION_FAILED: {'; '.join(f.reason for f in failures)}",
                clamped_payload=None,
                audit_hash=self._hash(neural_output, failures)
            )

        clamped = self._clamp(neural_output)
        return SymbolicVerdict(
            approved=True,
            reason="VALIDATED",
            clamped_payload=clamped,
            audit_hash=self._hash(neural_output, clamped)
        )
```

#### 3.3.3 LLM Trust Gating Integration

Your existing `swarm/utils/llm_trust_gating.py` becomes a **kernel service**:

```python
# kernel/llm_gate.py
class LLMTrustGate:
    """Runtime trust scoring for every neural inference."""

    def __init__(self):
        self.drift_detector = DriftDetector()  # from swarm/utils/llm_drift.py
        self.stability_monitor = StabilityMonitor()  # from swarm/utils/llm_stability.py

    def score_inference(self, history: list[dict], current: dict) -> TrustScore:
        drift_detected, drift_reasons = self.drift_detector.detect(history)
        stability = self.stability_monitor.measure(history)

        trust = TrustScore(
            confidence=current.get("confidence", 0.0),
            stability=stability,
            drift_detected=drift_detected,
            overall=min(current.get("confidence", 0.0), stability) * (0.5 if drift_detected else 1.0)
        )

        if trust.overall < TRUST_THRESHOLD:
            raise NeuralOutputRejected(f"Trust too low: {trust.overall}")

        return trust
```

---

### Phase 4: Multi-Agent Reinforcement Learning (Week 12–14)
**Goal:** Evolve the existing grid-search policy learner into a full MARL training loop.

#### 3.4.1 MARL Formulation

| MARL Concept | Procurement Mapping |
|---|---|
| **Agent** | Scout / Evaluator / Negotiator archetype |
| **Observation** | Local mesh field traces + position + energy |
| **Action** | Move to neighbor / Deposit pheromone / Negotiate / Rest |
| **Reward** | `+1` if procurement successful, `-1` if failed, `+0.1` per good deal pheromone |
| **State** | Full mesh field snapshot (centralized training) |
| **Policy** | PPO / MAPPO with shared critic |

#### 3.4.2 Reward Shaping (Preserving Safety)

```python
# learning/marl/rewards.py
def compute_reward(
    archetype: str,
    action: dict,
    mesh_state: dict,
    outcome: dict | None,
) -> float:
    """Reward function with safety invariants."""

    base_reward = 0.0

    if archetype == "negotiator":
        # Reward for depositing deal pheromone that wins
        if action["type"] == "deposit_deal":
            deal = action["payload"]
            if deal["price"] > deal.get("budget", float("inf")):
                return -10.0  # Hard penalty for over-budget (symbolic guard)
            base_reward += 0.1 * (1 - deal["price"] / deal.get("market_price", deal["price"]))

    if archetype == "scout":
        # Reward for discovering compliant suppliers
        if action["type"] == "deposit_discovery":
            if action["payload"].get("compliant", False):
                base_reward += 0.2

    if outcome is not None:
        if outcome["success"]:
            base_reward += 10.0
        else:
            base_reward -= 5.0

        # Bonus for matching the symbolic kernel's decision
        if outcome.get("emergent_winner") == outcome.get("kernel_winner"):
            base_reward += 2.0  # Alignment bonus

    return base_reward
```

#### 3.4.3 Training Loop

```python
# learning/marl/trainer.py
import ray
from ray.rllib.algorithms.mappo import MAPPOConfig

class ProcurementMARLTrainer:
    """Trains the mesh agents using MAPPO."""

    def __init__(self, env_class: type):
        self.config = (
            MAPPOConfig()
            .environment(env=env_class, env_config={"suppliers": 50, "materials": 6})
            .framework("torch")
            .resources(num_gpus=1)
            .rollouts(num_rollout_workers=4)
            .training(
                model={"fcnet_hiddens": [256, 256]},
                vf_loss_coeff=0.5,
                entropy_coeff=0.01,
            )
        )
        self.algo = self.config.build()

    def train(self, iterations: int = 1000):
        for i in range(iterations):
            result = self.algo.train()
            if i % 100 == 0:
                self._evaluate_policy_stability()
                self._checkpoint()
```

#### 3.4.4 Preserve Existing Learning Infrastructure

Your existing `swarm/learning/` is **not replaced** — it is **elevated**:

| Existing Component | New Role |
|---|---|
| `learner.py` + `policy.py` | **Baseline policy generator** — seeds MARL with grid-search winners |
| `replay_engine.py` | **MARL environment reset** — reconstructs states from traces |
| `event_store.py` | **Experience replay buffer** — feeds historical trajectories to MARL |
| `adaptive_policy.py` | **Safety policy constraints** — clamps MARL actions to safe regions |
| `routing.py` | **Action space pruning** — reduces valid actions per context |

---

### Phase 5: Distributed Execution with Ray (Week 15–16)
**Goal:** Scale from single-node `asyncio` to multi-node Ray cluster.

#### 3.5.1 Actor Pool Architecture

```python
# mesh/cluster.py
import ray

class ProcurementCluster:
    """Manages the Ray actor pool for procurement runs."""

    def __init__(self, num_scouts: int = 5, num_evaluators: int = 10, 
                 num_negotiators: int = 5):
        ray.init(address="auto")

        self.scouts = [ScoutActor.remote(f"scout_{i}") for i in range(num_scouts)]
        self.evaluators = [EvaluatorActor.remote(f"eval_{i}") for i in range(num_evaluators)]
        self.negotiators = [NegotiatorActor.remote(f"neg_{i}") for i in range(num_negotiators)]
        self.buyer = BuyerActor.remote("buyer_0")
        self.kernel = SafetyKernelActor.remote("kernel")

        # Register kernel as named actor
        ray.util.register_actor("safety_kernel", self.kernel)

    async def run_procurement(self, requirement: dict) -> dict:
        mesh = MeshEnvironment.remote(requirement)

        # Phase 1: Scatter scouts
        scout_tasks = [s.step.remote(mesh) for s in self.scouts]
        await ray.get(scout_tasks)

        # Phase 2: Scatter evaluators (sense local discoveries)
        eval_tasks = [e.step.remote(mesh) for e in self.evaluators]
        await ray.get(eval_tasks)

        # Phase 3: Scatter negotiators (sense local scores)
        neg_tasks = [n.step.remote(mesh) for n in self.negotiators]
        await ray.get(neg_tasks)

        # Phase 4: Buyer quorum consensus
        result = await self.buyer.resolve.remote(mesh)
        return result
```

#### 3.5.2 Fault Tolerance

```python
# mesh/resilience.py
@ray.remote(max_restarts=3, max_task_retries=3)
class ResilientScoutActor(MeshActor):
    """Scout with automatic restart on failure."""

    async def step(self, mesh):
        try:
            return await super().step(mesh)
        except Exception as e:
            # Log failure, deposit "failure_pheromone" so other scouts avoid this path
            await mesh.deposit.remote({
                "field_type": "failure_pheromone",
                "position": self.position,
                "error": str(e),
                "actor_id": self.actor_id
            })
            raise
```

---

### Phase 6: API & Observability Migration (Week 17–18)
**Goal:** Preserve all existing API contracts while serving the new mesh runtime.

#### 3.6.1 Dual-Runtime API

```python
# api/main.py (refactored)
from fastapi import FastAPI

app = FastAPI()

# Legacy endpoints — unchanged, served by legacy swarm
legacy_router = LegacySwarmRouter()
app.include_router(legacy_router, prefix="/v1")

# New mesh endpoints — served by Ray cluster
mesh_router = MeshRouter()
app.include_router(mesh_router, prefix="/v2")

# Unified observability
app.include_router(ObservabilityRouter(), prefix="/obs")
```

#### 3.6.2 Timeline Projection for Mesh

```python
# mesh/timeline.py
def build_mesh_timeline(mesh_state: dict) -> TimelineResponse:
    """Read-only projection of the mesh environment."""

    # Convert pheromone traces to timeline events
    events = []
    for field_type, traces in mesh_state["fields"].items():
        for trace in traces:
            events.append(TimelineEvent(
                timestamp=trace["timestamp"],
                phase=_field_type_to_phase(field_type),
                agent=trace.get("actor_id", "unknown"),
                type="pheromone_deposit",
                payload={k: v for k, v in trace.items() 
                        if k not in {"password", "token", "api_key"}}
            ))

    # Add quorum decision as synthetic event
    if mesh_state.get("quorum_result"):
        events.append(TimelineEvent(
            timestamp=mesh_state["quorum_result"]["timestamp"],
            phase="decision",
            agent="buyer_quorum",
            type="emergent_decision",
            payload=mesh_state["quorum_result"]
        ))

    return TimelineResponse(events=sorted(events, key=lambda e: e.timestamp))
```

---

### Phase 7: Hardening & Production Cutover (Week 19–20)
**Goal:** A/B test legacy vs. mesh, then cut over.

#### 3.7.1 A/B Testing Framework

```python
# testing/ab_test.py
class ProcurementABTest:
    """Runs both legacy and mesh on the same requirement, compares outcomes."""

    async def compare(self, requirement: dict) -> ABResult:
        legacy_result = await self.legacy_swarm.run(requirement)
        mesh_result = await self.mesh_cluster.run(requirement)

        return ABResult(
            same_supplier=legacy_result["supplier"] == mesh_result["supplier"],
            price_delta=mesh_result["price"] - legacy_result["price"],
            latency_delta=mesh_result["latency_ms"] - legacy_result["latency_ms"],
            legacy_trace=legacy_result["trace"],
            mesh_trace=mesh_result["trace"]
        )
```

#### 3.7.2 Gradual Rollout

| Stage | Traffic | Criteria to Advance |
|---|---|---|
| **Canary** | 1% | Mesh decisions match legacy > 95% of the time |
| **10%** | 10% | No mesh decision exceeds budget; all governance gates pass |
| **50%** | 50% | MARL policies show > 5% cost savings vs. legacy |
| **100%** | 100% | 30-day stability with zero safety violations |

---

## 4. File-Level Migration Map

### New Files (Create)

| File | Phase | Purpose |
|---|---|---|
| `mesh/environment.py` | 1 | Ray-based stigmergic field store |
| `mesh/topology.py` | 1 | Supplier graph for agent positioning |
| `mesh/actors/base.py` | 2 | `MeshActor` archetype base class |
| `mesh/actors/scout.py` | 2 | Scout archetype |
| `mesh/actors/evaluator.py` | 2 | Evaluator archetype |
| `mesh/actors/negotiator.py` | 2 | Negotiator archetype |
| `mesh/actors/buyer.py` | 2 | Buyer archetype with quorum consensus |
| `mesh/consensus/quorum.py` | 2 | Emergent decision algorithm |
| `kernel/__init__.py` | 3 | Safety kernel facade |
| `kernel/validator.py` | 3 | Symbolic validation of neural outputs |
| `kernel/llm_gate.py` | 3 | Trust gating as kernel service |
| `kernel/governance_service.py` | 3 | Extracted from `GovernanceAgent` |
| `kernel/risk_service.py` | 3 | Extracted from `RiskAssessmentAgent` |
| `kernel/approval_service.py` | 3 | Extracted from `ApprovalAgent` |
| `kernel/execution_service.py` | 3 | Extracted from `PurchaseOrderAgent` + `ExecutionTrackingAgent` |
| `cognitive/generation/negotiation.py` | 3 | Schema-constrained LLM counter-offer generation |
| `cognitive/perception/supplier_doc.py` | 3 | SLM-based supplier document parsing |
| `cognitive/memory/embedding_store.py` | 3 | pgvector-backed semantic memory |
| `learning/marl/rewards.py` | 4 | MARL reward shaping with safety invariants |
| `learning/marl/trainer.py` | 4 | MAPPO training loop |
| `learning/marl/env.py` | 4 | Ray RLlib environment wrapper |
| `mesh/cluster.py` | 5 | Ray actor pool manager |
| `mesh/resilience.py` | 5 | Fault-tolerant actor wrappers |
| `api/v2/router.py` | 6 | New mesh API endpoints |
| `testing/ab_test.py` | 7 | A/B testing framework |

### Modified Files (Refactor)

| File | Phase | Change |
|---|---|---|
| `pyproject.toml` | 0 | Add `ray[default]`, `ray[rllib]`, `outlines`, `vllm` |
| `docker-compose.yml` | 0 | Add `ray-head`, `ray-worker` services |
| `swarm/learning/learner.py` | 4 | Integrate with MARL trainer as baseline seeder |
| `swarm/learning/policy.py` | 4 | Add `to_marl_policy()` conversion method |
| `swarm/learning/adaptive_policy.py` | 4 | Expose as MARL action space constraints |
| `swarm/storage/event_store.py` | 4 | Add `load_trajectory()` for MARL experience replay |
| `swarm/simulation/replay_engine.py` | 4 | Use as MARL environment reset function |
| `api/main.py` | 6 | Mount both `/v1` (legacy) and `/v2` (mesh) routers |

### Deprecated Files (Move to `legacy/`)

| File | Replacement | Migration Path |
|---|---|---|
| `swarm/domain/wiring.py` | `mesh/cluster.py` | Wiring logic becomes actor pool initialization |
| `swarm/domain/agents/decision_agent.py` | `mesh/consensus/quorum.py` | Central argmax → distributed quorum |
| `swarm/core/event.py` + `event_bus.py` | `mesh/environment.py` | Pub/sub → stigmergic fields |
| `swarm/core/completion.py` | `mesh/consensus/quorum.py` | Completion tracker → quorum sensing |

---

## 5. Risk Mitigation

| Risk | Mitigation |
|---|---|
| **MARL policies violate budget** | Hard symbolic clamp in `kernel/validator.py` — any action exceeding budget is rejected before execution |
| **Quorum deadlock (no winner)** | Timeout fallback to legacy `DecisionAgent` argmax; log as `emergent_fallback` |
| **Ray actor crash cascades** | `max_restarts=3`, `failure_pheromone` deposition, circuit breaker on kernel RPC |
| **LLM drift in production** | Existing `llm_drift.py` + `llm_stability.py` promoted to kernel gate; auto-rollback to last known good policy |
| **Mesh decisions non-reproducible** | Deterministic seeding of Ray actors, deterministic pheromone decay, full mesh state snapshotting |
| **Performance regression** | A/B test latency; mesh parallelizes scout/evaluator/negotiator phases, should be faster |

---

## 6. Testing Strategy

| Test Layer | Scope | Target |
|---|---|---|
| **Unit** | `kernel/validator.py`, `mesh/consensus/quorum.py` | 100% coverage |
| **Integration** | `mesh/cluster.py` with `MockConnector` | All safety invariants pass |
| **MARL** | `learning/marl/env.py` | Win rate > 90% vs. random policy |
| **A/B** | `testing/ab_test.py` | > 95% decision alignment with legacy |
| **Chaos** | Kill random Ray actors mid-procurement | Recovery within 5s, no duplicate orders |
| **Load** | 100 concurrent procurement runs | p99 < 2s, zero safety violations |

---

## 7. Recommended Repository Name

| Option | Rationale |
|---|---|
| **`neuro-symbolic-procurement-mesh`** | Accurate, academic credibility, signals Type-6 architecture |
| **`cognitive-procurement-mesh`** | Shorter, emphasizes the neural layer without losing the structural metaphor |
| **`axiomis-procurement-runtime`** | Brand continuity with your AXIOMIS architecture work |

**My recommendation:** `cognitive-procurement-mesh` for the repo, with the description:

> *"A neuro-symbolic, stigmergic multi-agent procurement architecture. Deterministic safety kernel with LLM-augmented cognitive agents, closed-loop MARL policy learning, and emergent quorum consensus."*

---

## 8. Summary: What Changes, What Stays

| Aspect | Current | Target |
|---|---|---|
| **Topology** | Hardcoded 14-agent pipeline | 4 archetypes + emergent coordination |
| **Communication** | Global `EventBus` | Local stigmergic sensing on `MeshField` |
| **Decision** | Central `DecisionAgent.argmax()` | Distributed quorum consensus |
| **Scaling** | Single-node `asyncio` | Ray actor cluster |
| **Learning** | Grid-search policy optimization | MAPPO with safety constraints |
| **LLM Role** | Read-only cognitive analysis | Generative negotiation + perception, validated by kernel |
| **Safety** | Deterministic agents | Symbolic kernel veto over all neural outputs |
| **Auditability** | Artifact lineage + hash chain | Preserved + mesh state snapshots |
| **Replay** | Isolated SQLite replay | Deterministic Ray actor replay |

The migration is **evolutionary, not revolutionary.** Every safety invariant you have built — the hash chains, the idempotency guards, the contract pre-gates, the replay engine — is **preserved and elevated** into the kernel. The innovation is in the coordination layer: from pipeline to mesh, from broadcast to stigmergy, from central decision to emergent consensus.
