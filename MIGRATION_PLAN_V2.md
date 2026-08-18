# Migration Plan V2: From Deterministic Pipeline to Neuro-Symbolic Procurement Mesh

> **Version:** v2.0 | **Status:** Consolidated after Gemini architectural review | **Duration:** 16–20 weeks  
> **Philosophy:** Neural components propose. Symbolic kernel disposes. Distributed execution scales. Audit trail persists.

---

## 0. Honest Acknowledgment of Corrections

This plan incorporates critical corrections from an independent architectural review (Gemini):

| Original Flaw | Correction | Rationale |
|---|---|---|
| Spatial stigmergy (pheromone fields, graph movement, decay rates) | **Ray Distributed Blackboard with typed channels** | Procurement is an information graph, not a physical space. Spatial routing adds synthetic latency and non-determinism without business value. |
| Quorum consensus (summing pheromone concentrations) | **Deterministic MCDA Buyer** over distributed candidate pool | Financial decisioning requires auditable multi-criteria trade-offs. "Emergent voting" is un-justifiable under SOX. |
| MAPPO MARL training | **Contextual Bandits + Heuristic Policy Engine** | MAPPO requires millions of steps, suffers non-stationarity, and converges on adversarial strategies. Your existing grid-search learner with ContextVar overrides is the correct foundation. |

The core insight preserved: **distributed candidate generation + centralized deterministic selection** is how real-world procurement actually works.

---

## 1. Current-State Snapshot (As-Is)

Your repo is a **Type-2 Symbolic[Neuro] closed-loop procurement operating system**:

| System | File | Maturity |
|---|---|---|
| Deterministic 14-agent pipeline | `swarm/domain/wiring.py` | Production |
| Closed-loop policy learner | `swarm/learning/learner.py` (32KB) | Production |
| Deterministic policy versioning | `swarm/learning/policy.py` (17KB) | Production |
| Context-aware routing | `swarm/learning/routing.py` (16KB) | Production |
| Isolated replay engine | `swarm/simulation/replay_engine.py` (12KB) | Production |
| Persistent event store | `swarm/storage/event_store.py` (20KB) | Production |
| LLM trust layer (12 files) | `swarm/utils/llm_*.py` | Production |
| Adaptive policy resolution | `swarm/learning/adaptive_policy.py` | Production |
| Enterprise connector factory | `swarm/integrations/` | Production |
| Hash-chained PostgreSQL ledger | `core/ledger/repository.py` | Production |

**Explicit design constraint (from your docs):**  
> *"LLM integration remains explicitly out of scope by design. The runtime is built so a single, guarded adapter could later assist with negotiation language, supplier communication, contract analysis and market intelligence — never with policy enforcement, compliance, approval or financial controls, which stay deterministic."*

This constraint is **preserved and elevated** in this plan.

---

## 2. Target Architecture: Neuro-Symbolic Procurement Mesh (NSPM)

### 2.1 Architectural Tenets

1. **Safety-First Neuro-Symbolism**: The symbolic kernel is the non-overridable "root of trust." Neural layers generate proposals; the kernel validates, clamps, and authorizes.
2. **Distributed Generation, Centralized Selection**: Many agents explore the supplier space in parallel. One deterministic MCDA agent selects the winner. This mirrors real procurement committees.
3. **Typed Blackboard Coordination**: Agents communicate via a distributed blackboard with capability-scoped channels — not spatial pheromones, not a global broadcast bus.
4. **Homogeneous Archetypes, Heterogeneous Policies**: 4 agent archetypes with configurable policies, not 14 hardcoded specialists.
5. **Actor-Model Distribution**: Each archetype instance is a Ray actor. The blackboard is a Ray distributed object store.

### 2.2 Layer Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COGNITIVE PERCEPTION LAYER                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐│
│  │  Perception │  │  Generation │  │   Memory    │  │   Exploration       ││
│  │   (SLM)     │  │   (LLM)     │  │ (Embedding) │  │  (Contextual Bandit)││
│  │             │  │             │  │             │  │                     ││
│  │ • Supplier  │  │ • Negotiate │  │ • Semantic  │  │ • Strategy selection││
│  │   document  │  │   language  │  │   supplier  │  │ • Scout allocation  ││
│  │   parsing   │  │ • Contract  │  │   memory    │  │ • Stop criteria     ││
│  │ • Market    │  │   drafting  │  │ • Cross-run │  │                     ││
│  │   sentiment │  │ • Risk      │  │   retrieval │  │                     ││
│  │             │  │   narrative │  │             │  │                     ││
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘│
│         │                │                │                    │           │
│         └────────────────┴────────────────┴────────────────────┘           │
│                                    │                                       │
│                         ┌──────────▼──────────┐                            │
│                         │  Structured Output  │                            │
│                         │  (JSON Schema /     │                            │
│                         │   Pydantic v2)      │                            │
│                         └──────────┬──────────┘                            │
└────────────────────────────────────┼───────────────────────────────────────┘
                                     │
                         ┌───────────▼────────────┐
                         │   DETERMINISTIC SAFETY │
                         │        KERNEL          │
                         │                        │
                         │ • Policy Enforcement   │
                         │ • Budget Clamping      │
                         │ • Pre-gate Validation  │
                         │ • Hash-Chained Ledger  │
                         │ • Replay Engine        │
                         │ • Idempotency Guards   │
                         │ • MCDA Scoring         │
                         └───────────┬────────────┘
                                     │
┌────────────────────────────────────┼────────────────────────────────────────┐
│              RAY DISTRIBUTED MESH BUS (Typed Blackboard)                    │
│                                                                             │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐            │
│   │  Scout  │────►│ Channel │────►│ Channel │────►│Evaluator│            │
│   │  Actor  │     │discovery│     │  score  │     │  Actor  │            │
│   └────┬────┘     └─────────┘     └─────────┘     └────┬────┘            │
│        │                                               │                   │
│        │         ┌─────────┐     ┌─────────┐          │                   │
│        └────────►│ Channel │────►│ Channel │◄─────────┘                   │
│                  │  deal   │     │  risk   │                              │
│                  └─────────┘     └─────────┘                              │
│                        │                                                  │
│                        ▼                                                  │
│                   ┌─────────┐                                             │
│                   │  Buyer  │  ← Deterministic MCDA over candidate pool   │
│                   │  Actor  │                                             │
│                   └─────────┘                                             │
│                                                                             │
│   Channel ACLs (capability-scoped):                                       │
│   • Scout:    read[requirement] → write[discovery]                        │
│   • Evaluator: read[discovery] → write[score, risk]                       │
│   • Negotiator: read[score] → write[deal]                                 │
│   • Buyer:     read[deal, risk, score] → write[decision]                  │
│   • Kernel:    read[ALL] → write[decision_validation, execution_auth]     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 The Four Agent Archetypes

| Archetype | Role | Neural? | Symbolic? | Count | Blackboard Channels |
|---|---|---|---|---|---|
| **Scout** | Discovers suppliers, deposits discovery traces | SLM for web/doc research | Deterministic compliance filtering | N (elastic) | `read: requirement` → `write: discovery` |
| **Evaluator** | Reads discoveries, scores suppliers, deposits score + risk traces | SLM for document parsing | Deterministic MCDA scoring | N (elastic) | `read: discovery` → `write: score, risk` |
| **Negotiator** | Reads scores, engages pairwise, deposits deal traces | LLM for language generation | Deterministic floor/ceiling guardrails | N (elastic) | `read: score` → `write: deal` |
| **Buyer** | Reads deal + risk + score traces, runs deterministic MCDA, deposits decision | None | Deterministic MCDA argmax + policy enforcement | 1 per procurement | `read: deal, risk, score` → `write: decision` |

**Governance, Risk, Approval, Execution** are **kernel services**, not agents. Every archetype calls `kernel.validate()` before any `write` operation.

---

## 3. Phase-by-Phase Migration

### Phase 0: Foundation Hardening (Week 1–2)
**Goal:** Structural prep. Extract kernel. Add Ray deps. No behavioral changes.

#### 3.0.1 Repository Restructure

```
autonomous-procurement-swarm/
├── kernel/              # NEW: symbolic safety layer (extracted from swarm/domain/agents)
│   ├── __init__.py
│   ├── validator.py     # Neural output validation
│   ├── governance.py    # Extracted from GovernanceAgent
│   ├── risk.py          # Extracted from RiskAssessmentAgent
│   ├── approval.py      # Extracted from ApprovalAgent
│   ├── execution.py     # Extracted from PurchaseOrderAgent + ExecutionTrackingAgent
│   ├── contracts.py     # Extracted from ContractValidationAgent
│   ├── ledger.py        # Wraps core/ledger/repository.py
│   └── llm_gate.py      # Promoted from swarm/utils/llm_*.py
├── mesh/                # NEW: distributed blackboard + Ray actors
│   ├── __init__.py
│   ├── blackboard.py    # Typed channel store
│   ├── channels.py      # Channel definitions + ACLs
│   ├── topology.py      # Supplier graph (information topology, NOT spatial)
│   ├── actors/
│   │   ├── base.py      # MeshActor base class
│   │   ├── scout.py
│   │   ├── evaluator.py
│   │   ├── negotiator.py
│   │   └── buyer.py     # Deterministic MCDA
│   └── consensus/       # REMOVED: no quorum voting
├── cognitive/           # NEW: neural layer
│   ├── __init__.py
│   ├── perception/
│   │   └── supplier_doc.py
│   ├── generation/
│   │   └── negotiation.py  # Schema-constrained LLM
│   ├── memory/
│   │   └── embedding_store.py
│   └── exploration/
│       └── contextual_bandit.py
├── learning/            # EXPAND: existing swarm/learning/
│   ├── policy/          # existing
│   ├── replay/          # moved from swarm/simulation
│   ├── rewards/         # existing
│   ├── bandit/          # NEW: contextual bandit engine
│   └── heuristic/       # NEW: evolved from learner.py
├── legacy/              # TEMP: old swarm/ during migration
│   ├── core/
│   ├── domain/
│   ├── integrations/
│   └── utils/
├── api/                 # REFACTOR
│   ├── v1/              # legacy routes (mounted at /v1)
│   ├── v2/              # NEW: mesh routes (mounted at /v2)
│   └── observability/   # unified
└── core/                # UNCHANGED
    └── ledger/
```

#### 3.0.2 Dependencies

```toml
# pyproject.toml additions
[project.optional-dependencies]
ray = ["ray[default]>=2.9", "ray[rllib]>=2.9"]
cognitive = ["outlines>=0.0.40", "vllm>=0.3.0", "transformers>=4.38"]
bandit = ["scikit-learn>=1.4", "numpy>=1.26"]
```

#### 3.0.3 Interface Contracts

```python
# mesh/channels.py
from pydantic import BaseModel
from enum import StrEnum

class ChannelType(StrEnum):
    REQUIREMENT = "requirement"
    DISCOVERY = "discovery"
    SCORE = "score"
    RISK = "risk"
    DEAL = "deal"
    DECISION = "decision"

class ChannelACL(BaseModel):
    """Capability-based read/write permissions per channel."""
    channel: ChannelType
    read: list[str]   # archetypes that can read
    write: list[str]  # archetypes that can write

CHANNEL_ACLS: list[ChannelACL] = [
    ChannelACL(channel=ChannelType.REQUIREMENT, read=["scout", "evaluator", "negotiator", "buyer", "kernel"], write=["kernel"]),
    ChannelACL(channel=ChannelType.DISCOVERY, read=["evaluator", "kernel"], write=["scout"]),
    ChannelACL(channel=ChannelType.SCORE, read=["negotiator", "buyer", "kernel"], write=["evaluator"]),
    ChannelACL(channel=ChannelType.RISK, read=["buyer", "kernel"], write=["evaluator"]),
    ChannelACL(channel=ChannelType.DEAL, read=["buyer", "kernel"], write=["negotiator"]),
    ChannelACL(channel=ChannelType.DECISION, read=["kernel"], write=["buyer"]),
]
```

---

### Phase 1: Ray Distributed Blackboard (Week 3–5)
**Goal:** Replace global `EventBus` with typed blackboard on Ray Object Store.

#### 3.1.1 Why Not Spatial Stigmergy

Gemini correctly identified that spatial pheromone fields (decay rates, graph movement, local radius sensing) are inappropriate for procurement. Procurement data is:
- **Non-spatial**: Suppliers are not points in Euclidean space.
- **Static during a run**: A supplier's location doesn't change mid-procurement.
- **Audit-critical**: "The agent moved to node X and sensed pheromone Y" is not a valid audit trail.

The correct abstraction is a **typed blackboard** — a distributed key-value store where each key is a channel, and agents have capability-scoped read/write access.

#### 3.1.2 Ray-Based Blackboard

```python
# mesh/blackboard.py
import ray
from ray.util.queue import Queue
from pydantic import BaseModel
from typing import Any

@ray.remote
class TypedBlackboard:
    """Distributed blackboard with capability-scoped channels.

    Unlike SwarmState (global dict) or EventBus (global broadcast),
    this enforces that agents only see what their capabilities allow.
    """

    def __init__(self):
        self._channels: dict[ChannelType, list[dict]] = {
            ct: [] for ct in ChannelType
        }
        self._acls = {acl.channel: acl for acl in CHANNEL_ACLS}

    def write(self, archetype: str, channel: ChannelType, payload: dict) -> str:
        """Write to a channel. Enforced by ACL."""
        acl = self._acls[channel]
        if archetype not in acl.write:
            raise PermissionError(
                f"Archetype '{archetype}' cannot write to '{channel}'. "
                f"Allowed writers: {acl.write}"
            )

        trace = {
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "archetype": archetype,
            "payload": payload,
            "parent_ids": payload.get("parent_ids", []),
        }
        self._channels[channel].append(trace)
        return trace["id"]

    def read(self, archetype: str, channel: ChannelType, 
             limit: int = 100) -> list[dict]:
        """Read from a channel. Enforced by ACL."""
        acl = self._acls[channel]
        if archetype not in acl.read:
            raise PermissionError(
                f"Archetype '{archetype}' cannot read from '{channel}'. "
                f"Allowed readers: {acl.read}"
            )
        return self._channels[channel][-limit:]

    def read_multi(self, archetype: str, 
                   channels: list[ChannelType]) -> dict[ChannelType, list[dict]]:
        """Read multiple channels atomically."""
        return {ch: self.read(archetype, ch) for ch in channels}

    def snapshot(self) -> dict:
        """Full snapshot for replay / audit."""
        return {
            "channels": {k: v.copy() for k, v in self._channels.items()},
            "timestamp": time.time(),
        }
```

**Key differences from `SwarmState`:**
- `SwarmState`: Any agent can `find_artifacts(kind="quote")` and see everything.
- `TypedBlackboard`: A Scout cannot read `deal` channel. A Negotiator cannot read `risk` channel. This is **encapsulation**, not spatial locality.

#### 3.1.3 Migration of Existing Artifacts to Channels

| Old Artifact | New Channel | Written By | Read By |
|---|---|---|---|
| `requirement` | `requirement` | Kernel (from API input) | Scout, Evaluator, Negotiator, Buyer |
| `supplier_list` | `discovery` | Scout | Evaluator |
| `evaluation` | `score` | Evaluator | Negotiator, Buyer |
| `risk_assessment` | `risk` | Evaluator | Buyer |
| `quote` | `deal` | Negotiator | Buyer |
| `decision` | `decision` | Buyer (MCDA) | Kernel |
| `governance_decision` | N/A — kernel service | Kernel | N/A |
| `execution_authorization` | N/A — kernel service | Kernel | N/A |

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
        self.cognitive_engine = cognitive_engine
        self.kernel = ray.get_actor("safety_kernel")

    async def step(self, blackboard: ray.ObjectRef) -> None:
        """The universal agent loop."""
        # 1. Read what I'm allowed to see
        perception = await self.perceive(blackboard)

        # 2. Generate proposal (may use LLM/SLM)
        proposal = await self.reason(perception)

        # 3. MANDATORY: Kernel validation before any write
        verdict = await self.kernel.validate.remote(
            NeuralProposal(
                proposal_id=str(uuid.uuid4()),
                archetype=self.archetype,
                payload=proposal,
                confidence=proposal.get("confidence", 0.0),
                structured=True,
            )
        )

        if not verdict.approved:
            await self.log_rejection(verdict)
            return

        # 4. Write to blackboard (ACL-enforced)
        await self.act(blackboard, verdict.clamped_payload or proposal)
```

#### 3.2.2 Mapping Old Agents to New Archetypes

| Old Agent(s) | New Archetype | Cognitive Role | Symbolic Role |
|---|---|---|---|
| `RequirementAgent` | **Buyer** init | SLM parses free-text RFQs | Kernel validates constraints |
| `StrategyAgent` | **Buyer** strategy | Contextual bandit suggests strategy | Kernel selects from approved set |
| `SupplierDiscoveryAgent` | **Scout** | SLM researches suppliers | Deterministic compliance filter |
| `EvaluationAgent` | **Evaluator** | SLM parses supplier docs/ESG | Deterministic MCDA scoring |
| `NegotiationAgent` | **Negotiator** | LLM generates counter-offer language | Floor/ceiling clamping by kernel |
| `DecisionAgent` | **Buyer** MCDA | None | Deterministic argmax over candidate pool |
| `RiskAssessmentAgent` | **Evaluator** risk | None | Deterministic risk scoring, writes to `risk` channel |
| `GovernanceAgent` | **Kernel Service** | None | Policy rule engine |
| `ApprovalAgent` | **Kernel Service** | None | Authorization state machine |
| `PurchaseOrderAgent` | **Kernel Service** | None | Connector dispatch + idempotency |
| `ExecutionTrackingAgent` | **Kernel Service** | None | Lifecycle tracking |
| `OutcomeAgent` | **Kernel Service** | None | Payload normalization |
| `SupplierIntelligenceAgent` | **Evaluator** memory | Embedding model for similarity | Rolling average updates |
| `ContractValidationAgent` | **Kernel Service** | SLM for clause extraction | Boolean compliance validation |

#### 3.2.3 The Buyer: Deterministic MCDA over Distributed Candidate Pool

**Gemini's correction implemented:** The Buyer does NOT use quorum voting. It uses deterministic Multi-Criteria Decision Analysis.

```python
# mesh/actors/buyer.py
@ray.remote
class BuyerActor(MeshActor):
    """The Buyer archetype performs deterministic MCDA over the distributed candidate pool.

    This replaces the central DecisionAgent with the same deterministic logic,
    but operating over a richer candidate pool generated by distributed scouts/evaluators/negotiators.
    """

    def __init__(self, actor_id: str, policy: GovernancePolicy):
        super().__init__(actor_id, "buyer", cognitive_engine=None)
        self.policy = policy
        self.scoring_weights = {
            "price": 0.40,
            "lead_time": 0.25,
            "esg": 0.20,
            "reliability": 0.15,
        }

    async def perceive(self, blackboard: ray.ObjectRef) -> dict:
        """Read deal, risk, and score channels."""
        bb = ray.get(blackboard)
        return bb.read_multi.remote("buyer", [ChannelType.DEAL, ChannelType.RISK, ChannelType.SCORE])

    async def reason(self, perception: dict) -> dict:
        """Deterministic MCDA over all candidates."""
        deals = perception[ChannelType.DEAL]
        risks = perception[ChannelType.RISK]
        scores = perception[ChannelType.SCORE]

        # Build candidate pool
        candidates = []
        for deal in deals:
            supplier_id = deal["payload"]["supplier_id"]

            # Look up risk and score for this supplier
            risk = next((r for r in risks if r["payload"]["supplier_id"] == supplier_id), None)
            score = next((s for s in scores if s["payload"]["supplier_id"] == supplier_id), None)

            if not score:
                continue  # Cannot evaluate without score

            composite = self._compute_composite_score(deal, score, risk)
            candidates.append({
                "supplier_id": supplier_id,
                "composite_score": composite,
                "deal": deal["payload"],
                "risk": risk["payload"] if risk else None,
                "score": score["payload"],
            })

        if not candidates:
            return {"action": "insufficient_candidates", "reason": "No scored suppliers available"}

        # Policy filter (from GovernancePolicy)
        valid_candidates = [
            c for c in candidates 
            if self._passes_policy_filter(c)
        ]

        if not valid_candidates:
            return {"action": "policy_rejection", "reason": "No candidates pass governance policy"}

        # Deterministic argmax
        winner = max(valid_candidates, key=lambda c: c["composite_score"])

        return {
            "action": "decision",
            "selected_supplier": winner["supplier_id"],
            "composite_score": winner["composite_score"],
            "ranked": sorted(candidates, key=lambda c: c["composite_score"], reverse=True),
            "method": "deterministic_mcda",
            "policy_applied": self.policy.name,
        }

    def _compute_composite_score(self, deal: dict, score: dict, risk: dict | None) -> float:
        """Deterministic weighted scoring — identical to core/evaluator/scoring.py."""
        price_score = 1.0 - (deal["payload"]["price"] / deal["payload"].get("budget", deal["payload"]["price"]))
        lead_time_score = score["payload"].get("lead_time_score", 0.5)
        esg_score = score["payload"].get("esg_score", 0.5)
        reliability_score = score["payload"].get("reliability_score", 0.5)

        composite = (
            self.scoring_weights["price"] * price_score +
            self.scoring_weights["lead_time"] * lead_time_score +
            self.scoring_weights["esg"] * esg_score +
            self.scoring_weights["reliability"] * reliability_score
        )

        # Risk adjustment (deterministic)
        if risk:
            risk_penalty = risk["payload"].get("overall_risk_score", 0.0) * 0.3
            composite *= (1.0 - risk_penalty)

        return round(composite, 6)  # Explicit rounding for determinism
```

**Why this is correct:**
- The CFO gets: "We selected MinerCorp_A because it scored 0.923 on our weighted criteria matrix."
- Not: "The swarm converged on MinerCorp_A because its pheromone concentration was highest."
- The exploration (scouts, evaluators, negotiators) is distributed and dynamic.
- The selection is centralized, deterministic, and auditable.

---

### Phase 3: Neural-Symbolic Bridge (Week 9–11)
**Goal:** Make the cognitive layer production-safe.

#### 3.3.1 Schema-Constrained LLM Generation

```python
# cognitive/generation/negotiation.py
from outlines import models, generate
from pydantic import BaseModel, Field

class CounterOfferSchema(BaseModel):
    """Every LLM-generated counter-offer MUST conform to this schema."""
    proposed_price: float = Field(..., ge=0, le=1_000_000, description="Must be non-negative")
    proposed_lead_time_days: int = Field(..., ge=1, le=365)
    payment_terms: str = Field(..., pattern="^(net_30|net_60|cod|letter_of_credit)$")
    justification: str = Field(..., max_length=500)
    confidence: float = Field(..., ge=0.0, le=1.0)

class NegotiationGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        self.model = models.transformers(model_name)
        self.generator = generate.json(self.model, CounterOfferSchema)

    async def generate(self, context: dict) -> CounterOfferSchema:
        prompt = self._build_prompt(context)
        return self.generator(prompt)
```

#### 3.3.2 Symbolic Validator (Kernel Service)

```python
# kernel/validator.py
class SymbolicValidator:
    """Deterministic guardrail: rejects or clamps any neural output."""

    def validate(self, proposal: NeuralProposal) -> SymbolicVerdict:
        payload = proposal.payload
        checks = [
            self._check_price_bounds(payload, proposal.archetype),
            self._check_lead_time_bounds(payload),
            self._check_payment_terms_whitelist(payload),
            self._check_material_whitelist(payload),
            self._check_budget_compliance(payload),
        ]

        failures = [c for c in checks if not c.passed]
        if failures:
            return SymbolicVerdict(
                approved=False,
                reason=f"VALIDATION_FAILED: {'; '.join(f.reason for f in failures)}",
                clamped_payload=None,
                audit_hash=self._hash(proposal, failures)
            )

        clamped = self._clamp(payload)
        return SymbolicVerdict(
            approved=True,
            reason="VALIDATED",
            clamped_payload=clamped,
            audit_hash=self._hash(proposal, clamped)
        )
```

#### 3.3.3 LLM Trust Gating (Promoted to Kernel)

Your existing `swarm/utils/llm_drift.py`, `llm_stability.py`, `llm_trust_gating.py` become kernel services:

```python
# kernel/llm_gate.py
class LLMTrustGate:
    def __init__(self):
        self.drift = DriftDetector()      # from swarm/utils/llm_drift.py
        self.stability = StabilityMonitor()  # from swarm/utils/llm_stability.py

    def score(self, history: list[dict], current: dict) -> TrustScore:
        drift_detected, reasons = self.drift.detect(history)
        stability = self.stability.measure(history)

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

### Phase 4: Contextual Bandits + Heuristic Policy Engine (Week 12–14)
**Gemini's correction implemented:** No MAPPO. No unconstrained neural MARL.

#### 3.4.1 Why Contextual Bandits

| Property | MAPPO | Contextual Bandits | Your Grid-Search |
|---|---|---|---|
| Sample efficiency | Millions of steps | Hundreds of steps | N/A (offline) |
| Non-stationarity | Catastrophic | Handle gracefully | Deterministic |
| Action space | Continuous/discrete | Discrete (strategies) | Discrete (parameters) |
| Interpretability | Black box | Linear/GLM weights | Fully interpretable |
| Safety | Hard to constrain | Easy to clamp | Fully constrained |

Contextual bandits are the right abstraction for **strategy selection** (which of 3 strategies to use given market context) and **scout allocation** (how many scouts to deploy given supplier diversity).

#### 3.4.2 LinUCB for Strategy Selection

```python
# learning/bandit/strategy_bandit.py
import numpy as np
from sklearn.linear_model import Ridge

class StrategyBandit:
    """LinUCB for selecting procurement strategy given market context.

    Arms: cost_optimized, balanced, low_carbon
    Context: [market_volatility, carbon_constraint_present, budget_tightness, supplier_count]
    """

    def __init__(self, n_arms: int = 3, n_features: int = 4, alpha: float = 1.0):
        self.n_arms = n_arms
        self.alpha = alpha
        self.A = [np.eye(n_features) for _ in range(n_arms)]  # Design matrix
        self.b = [np.zeros(n_features) for _ in range(n_arms)]  # Response vector
        self.theta = [np.zeros(n_features) for _ in range(n_arms)]  # Estimated weights

    def select(self, context: np.ndarray) -> int:
        """Select arm (strategy) with highest UCB."""
        ucb_values = []
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            self.theta[a] = A_inv @ self.b[a]

            # UCB = predicted reward + exploration bonus
            pred = self.theta[a] @ context
            uncertainty = self.alpha * np.sqrt(context @ A_inv @ context)
            ucb_values.append(pred + uncertainty)

        return int(np.argmax(ucb_values))

    def update(self, arm: int, context: np.ndarray, reward: float):
        """Update after observing reward."""
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
```

#### 3.4.3 Integration with Existing Learner

Your existing `learner.py` becomes the **offline evaluator** for the bandit:

```python
# learning/heuristic/evolved_learner.py
class EvolvedPolicyLearner:
    """Combines grid-search baseline with contextual bandit exploration."""

    def __init__(self):
        self.baseline = PolicyLearner()  # existing grid-search
        self.bandit = StrategyBandit()
        self.exploration_rate = 0.1

    def select_strategy(self, context: dict) -> str:
        """Hybrid: mostly bandit, occasionally grid-search winner."""
        if random.random() < self.exploration_rate:
            # Fallback to proven grid-search winner
            return self.baseline.best_strategy()

        context_vec = self._encode_context(context)
        arm = self.bandit.select(context_vec)
        return ["cost_optimized", "balanced", "low_carbon"][arm]

    def feedback(self, strategy: str, context: dict, outcome: dict):
        """Feed outcome back to bandit."""
        reward = self._compute_reward(outcome)
        context_vec = self._encode_context(context)
        arm = ["cost_optimized", "balanced", "low_carbon"].index(strategy)
        self.bandit.update(arm, context_vec, reward)
```

#### 3.4.4 Multi-Agent Coordination Without MARL

Instead of MAPPO, use **mean-field approximation** for scout allocation:

```python
# learning/heuristic/scout_allocator.py
class ScoutAllocator:
    """Determines how many Scout actors to spawn based on market complexity.

    No neural network. Pure heuristic with bandit-tuned parameters.
    """

    def allocate(self, requirement: dict, market_state: dict) -> int:
        base = 3

        # Heuristic factors
        if market_state["supplier_diversity"] > 0.7:
            base += 2
        if requirement.get("max_carbon_per_unit"):
            base += 1  # Carbon-constrained needs more scouting
        if market_state["volatility"] > 0.5:
            base += 2

        # Clamp to safe range
        return min(base, 10)  # Never spawn more than 10 scouts
```

---

### Phase 5: Ray Actor Distribution (Week 15–16)
**Goal:** Scale execution to thousands of dynamic evaluations.

#### 3.5.1 Actor Pool

```python
# mesh/cluster.py
import ray

class ProcurementCluster:
    def __init__(self, config: MeshConfig):
        ray.init(address="auto")

        self.scouts = [ScoutActor.remote(f"scout_{i}") for i in range(config.n_scouts)]
        self.evaluators = [EvaluatorActor.remote(f"eval_{i}") for i in range(config.n_evaluators)]
        self.negotiators = [NegotiatorActor.remote(f"neg_{i}") for i in range(config.n_negotiators)]
        self.buyer = BuyerActor.remote("buyer_0", policy=config.policy)
        self.kernel = SafetyKernelActor.remote("kernel")
        ray.util.register_actor("safety_kernel", self.kernel)

    async def run(self, requirement: dict) -> dict:
        blackboard = TypedBlackboard.remote()

        # Write requirement to blackboard
        bb = ray.get(blackboard)
        await bb.write.remote("kernel", ChannelType.REQUIREMENT, requirement)

        # Phase 1: Parallel scout deployment
        scout_tasks = [s.step.remote(blackboard) for s in self.scouts]
        await asyncio.gather(*[asyncio.wrap_future(t) for t in scout_tasks])

        # Phase 2: Parallel evaluation
        eval_tasks = [e.step.remote(blackboard) for e in self.evaluators]
        await asyncio.gather(*[asyncio.wrap_future(t) for t in eval_tasks])

        # Phase 3: Parallel negotiation
        neg_tasks = [n.step.remote(blackboard) for n in self.negotiators]
        await asyncio.gather(*[asyncio.wrap_future(t) for t in neg_tasks])

        # Phase 4: Deterministic MCDA
        decision = await self.buyer.step.remote(blackboard)
        return ray.get(decision)
```

#### 3.5.2 Fault Tolerance

```python
# mesh/resilience.py
@ray.remote(max_restarts=3, max_task_retries=3)
class ResilientScoutActor(ScoutActor):
    async def step(self, blackboard):
        try:
            return await super().step(blackboard)
        except Exception as e:
            # Deposit failure trace for observability
            bb = ray.get(blackboard)
            await bb.write.remote("scout", ChannelType.DISCOVERY, {
                "type": "failure",
                "error": str(e),
                "actor_id": self.actor_id,
            })
            raise
```

---

### Phase 6: API Migration (Week 17–18)
**Goal:** Serve both runtimes without breaking existing clients.

```python
# api/main.py
from fastapi import FastAPI

app = FastAPI()

# Legacy runtime — unchanged
from api.v1 import legacy_router
app.include_router(legacy_router, prefix="/v1")

# New mesh runtime
from api.v2 import mesh_router
app.include_router(mesh_router, prefix="/v2")

# Unified observability
from api.observability import obs_router
app.include_router(obs_router, prefix="/obs")
```

**New V2 endpoints mirror V1 but return richer traces:**
- `POST /v2/procurement` — dispatch to mesh
- `GET /v2/procurement/{id}/blackboard` — read-only blackboard snapshot
- `GET /v2/procurement/{id}/timeline` — mesh-aware timeline projection

---

### Phase 7: A/B Testing & Cutover (Week 19–20)

```python
# testing/ab_test.py
class ProcurementABTest:
    async def compare(self, requirement: dict) -> ABResult:
        legacy = await self.legacy_swarm.run(requirement)
        mesh = await self.mesh_cluster.run(requirement)

        return ABResult(
            same_supplier=legacy["supplier"] == mesh["supplier"],
            price_delta=mesh["price"] - legacy["price"],
            legacy_trace=legacy["trace"],
            mesh_trace=mesh["trace"],
            mesh_scout_count=mesh.get("n_scouts", 0),
            mesh_eval_count=mesh.get("n_evaluators", 0),
        )
```

| Stage | Traffic | Advance Criteria |
|---|---|---|
| Canary | 1% | > 95% decision alignment with legacy |
| 10% | 10% | Zero budget violations; all governance gates pass |
| 50% | 50% | Bandit shows > 3% cost savings vs. legacy |
| 100% | 100% | 30-day stability, zero safety violations |

---

## 4. Risk Mitigation

| Risk | Mitigation |
|---|---|
| Bandit explores unsafe strategy | Kernel policy whitelist — bandit can only select from pre-approved strategies |
| Buyer MCDA ties (two equal scores) | Deterministic tie-breaker: lower `supplier_id` lexicographically; logged explicitly |
| Ray actor crash | `max_restarts=3`, failure trace deposition, circuit breaker on kernel RPC |
| Blackboard ACL bypass | Kernel validates every write; invalid ACL attempts are rejected and logged |
| LLM drift in production | Existing `llm_drift.py` + `llm_stability.py` promoted to kernel gate |
| Performance regression | A/B test latency; mesh parallelizes exploration phases, should be faster |

---

## 5. Testing Strategy

| Layer | Scope | Target |
|---|---|---|
| Unit | `kernel/validator.py`, `mesh/actors/buyer.py` | 100% coverage |
| Integration | `mesh/cluster.py` with `MockConnector` | All safety invariants pass |
| Bandit | `learning/bandit/strategy_bandit.py` | Regret bounded, converges in < 500 steps |
| A/B | `testing/ab_test.py` | > 95% decision alignment with legacy |
| Chaos | Kill random Ray actors mid-run | Recovery < 5s, no duplicate orders |
| Load | 100 concurrent runs | p99 < 2s, zero safety violations |

---

## 6. What Changes, What Stays

| Aspect | Current | Target |
|---|---|---|
| **Topology** | Hardcoded 14-agent pipeline | 4 archetypes + distributed blackboard |
| **Communication** | Global `EventBus` broadcast | Typed blackboard with capability ACLs |
| **Decision** | Central `DecisionAgent.argmax()` | Deterministic MCDA Buyer over distributed pool |
| **Scaling** | Single-node `asyncio` | Ray actor cluster |
| **Learning** | Grid-search policy optimization | Grid-search baseline + contextual bandits |
| **LLM Role** | Read-only cognitive analysis | Generative negotiation + perception, validated by kernel |
| **Safety** | Deterministic agents | Symbolic kernel veto over all neural outputs |
| **Auditability** | Artifact lineage + hash chain | Preserved + blackboard snapshots |
| **Replay** | Isolated SQLite replay | Deterministic Ray actor replay |

---

## 7. Final Recommendation

**Repo name:** `cognitive-procurement-mesh`  
**Description:** *"A neuro-symbolic, distributed multi-agent procurement architecture. Deterministic safety kernel with LLM-augmented cognitive agents, contextual bandit policy learning, and typed blackboard coordination."*

**Execution priority:**
1. **Phase 0** (extract kernel) — immediate security gain
2. **Phase 3** (schema-constrained LLM) — immediate stability gain
3. **Phase 1** (Ray blackboard) — scalability gain
4. **Phase 2** (archetype collapse) — maintainability gain
5. **Phase 4** (contextual bandits) — optimization gain

The neural components propose. The symbolic kernel disposes. The distributed mesh scales. The audit trail persists.
