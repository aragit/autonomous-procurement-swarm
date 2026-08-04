# Swarm Runtime Architecture

The swarm runtime (`swarm/`) is the architectural spine for autonomous
multi-agent procurement. This document describes its layers, the data flow of a
single request, and where it is heading.

> **Status:** Phase 7 — an end-to-end deterministic **Autonomous Procurement
> Operating System**: parallel, per-supplier execution, strategy-aware
> evaluation, an auditable supplier-intelligence feedback loop, a governance +
> risk layer that gates whether a decision is safe and authorized to execute, and
> a deterministic execution tier that turns an authorized decision into a traced
> purchase order and tracks it to delivery. The runtime is fully deterministic —
> no planning, no LLM integration, no autonomous learning — and a planner (and any
> LLM usage) remains explicitly out of scope. Governance, risk, approval and
> execution are static, policy-driven artifacts (no autonomous approval).

## Layer overview

```mermaid
flowchart TB
    subgraph Agents["Agents (Phase 7)"]
        RA[RequirementAgent]
        SA[StrategyAgent]
        SDA[SupplierDiscoveryAgent]
        EA[EvaluationAgent]
        NA[NegotiationAgent]
        DA[DecisionAgent]
        RSA[RiskAssessmentAgent]
        GA[GovernanceAgent]
        APA[ApprovalAgent]
        POA[PurchaseOrderAgent]
        ETA[ExecutionTrackingAgent]
        OA[OutcomeAgent]
        SIA[SupplierIntelligenceAgent]
    end

    subgraph Orchestration["Orchestration"]
        C[SwarmCoordinator]
        B[EventBus]
        R[AgentRegistry]
        L[Structured Logs]
        CT[CompletionTracker]
    end

    subgraph State["Shared State"]
        A[(SwarmState)]
        AR[Artifacts: requirement / strategy / supplier_list / evaluation / quote / decision / decision_explanation / risk_assessment / governance_decision / execution_authorization / purchase_order / execution_status / procurement_outcome / supplier_performance]
        EV[Event history]
        RS[Results + completions]
    end

    C -->|register / route / replay| B
    C --> R
    C --> A
    B --> L
    Agents -->|publish events| B
    Agents -->|read / write| A
    A --> AR
    A --> EV
    A --> RS
    B -->|completion events| CT
    CT -->|is_group_completed| A
```

## Components

| Component | Module | Responsibility |
|-----------|--------|----------------|
| **BaseAgent** | `swarm/core/agent.py` | Abstract agent with `perceive → reason → act` lifecycle, status, a declarative `Capability` schema and a `tags` dict; `step(event)` protocol driven by `drive_on_event` / `route_on_event` |
| **Message** | `swarm/core/message.py` | Sender / receiver / intent / payload / metadata / `correlation_id` |
| **Event + EventBus** | `swarm/core/event.py` | Typed events on an async pub/sub bus; audit log with **replay** |
| **Artifact** | `swarm/core/artifact.py` | Typed, versioned working data produced by agents; supports `tags` for selective queries and `parent_ids` for lineage |
| **SwarmState** | `swarm/core/state.py` | Shared, serializable context over artifacts, events and results; `find_artifacts()` queries by kind / name / tags / `correlation_id`; `expect_artifact` / `complete_artifact` / `get_execution_trace` |
| **CompletionTracker** | `swarm/core/completion.py` | Subscribes to every event, closes a phase once the expected artifact count exists, publishes `EvaluationCompleted` / `QuotesCompleted` (idempotent) |
| **Capability** | `swarm/core/capability.py` | Declarative schema for what an agent can do (name / description / parameters / priority) |
| **AgentRegistry** | `swarm/core/registry.py` | Name-indexed registry with priority-ranked capability discovery and `best_for_capability(capability, **tags)` routing |
| **SwarmCoordinator** | `swarm/orchestration/coordinator.py` | Internal engine: registers agents, routes events, maintains state, seeds `correlation_id` |
| **Swarm** | `swarm/core/__init__.py` | Public facade: start / stop / send / replay / `get_execution_trace` / `expect_artifact` / `complete_artifact`. No business logic |
| **Logging** | `swarm/core/logging.py` | `SWARM_LOG_LEVEL` control (`DEBUG` full event detail, `INFO` lifecycle only) |
| **Wiring** | `swarm/domain/wiring.py` | `build_procurement_swarm(..., supplier_memory=..., governance_policy=..., supplier_connector=...)` — registers all agents + tracker with the Phase 7 subscriptions and strategy-weighted routing (Requirement→Strategy→Discovery→Evaluation→Negotiation→Decision→Risk→Governance→Approval→**Order→Execution**→Outcome) |
| **Domain events** | `swarm/domain/events.py` | `ProcurementEventType` (`RequirementCreated`, `StrategySelected`, per-supplier `SupplierDiscovered` / `SupplierEvaluated` / `QuoteGenerated`, completion `EvaluationCompleted` / `QuotesCompleted`, `DecisionMade`, `ContractValidated` / `ContractRejected`, `RiskAssessmentCompleted`, `GovernanceDecisionMade`, `ApprovalGranted` / `ApprovalRequired` / `ApprovalRejected`, `PurchaseOrderCreated`, `ExecutionStatusUpdated`, `ExternalCallRecorded`) and the `CreateRequirement` / `RecordProcurementOutcome` / `ApproveDecision` / `Execute` / `SyncExternal` intents |
| **Domain artifacts** | `swarm/domain/artifacts.py` | Requirement / strategy / supplier list / evaluation / quote / decision / decision-explanation / `contract_validation` / `risk_assessment` / `governance_decision` / `execution_authorization` / `purchase_order` / `execution_status` / procurement-outcome / supplier-performance / `external_call` artifacts with documented data contracts and `parent_ids` lineage |
| **Domain agents** | `swarm/domain/agents/` | `RequirementAgent`, `StrategyAgent`, `SupplierDiscoveryAgent`, `EvaluationAgent`, `NegotiationAgent`, `DecisionAgent`, `ContractValidationAgent`, `RiskAssessmentAgent`, `GovernanceAgent`, `ApprovalAgent`, `PurchaseOrderAgent`, `ExecutionTrackingAgent`, `OutcomeAgent`, `SupplierIntelligenceAgent` — pure adapters over the existing `core/` logic |
| **Order model** | `swarm/domain/order.py` | Deterministic `PurchaseOrder` (`PurchaseStatus` lifecycle), `OrderLine` and the `SupplierConnector` protocol with a deterministic `MockSupplierConnector` |
| **Contracts model** | `swarm/domain/contracts.py` | Deterministic `Contract` (supplier coverage, allowed items, pricing caps, expiry/compliance) with a pure `validate(...)` → `(valid, reason)` |
| **Integration layer** | `swarm/integrations/` | `BaseConnector` port (`swarm/integrations/base.py` + `ExternalResponse`/`ExternalStatus`), deterministic `MockConnector`, stateless `SupplierAPIConnector`, and ERP adapters (`SAPConnector`, `OracleConnector`, `CoupaConnector` with `ConnectorConfig`) |
| **Idempotency** | `swarm/utils/idempotency.py` | `IdempotencyGuard` — deterministic `(decision_id, action)` deduplication of external side effects |
| **Pricing helpers** | `swarm/domain/pricing.py` | Deterministic floor price, lead time, carbon, bid bond (mirrors `core.agents.supplier` rules) |
| **Risk model** | `swarm/domain/risk.py` | Deterministic `RiskAssessment` with financial / delivery / quality / carbon sub-scores, `RiskLevel`, and `compute_risk_scores` / `classify_risk` |
| **Governance model** | `swarm/domain/governance.py` | `GovernancePolicy` (`standard_policy`, `strict_policy`) and deterministic `GovernanceDecision.from_risk` (`APPROVED` / `APPROVAL_REQUIRED` / `REJECTED`) |
| **Supplier memory** | `swarm/memory/supplier.py` | Deterministic, in-memory `SupplierMemoryStore` + module-level `default_store`; `update_from_outcome` maintains running averages, `history_adjustment` maps reliability to a score delta |
| **Strategies** | `swarm/domain/strategy.py` | `Strategy` (price / score / carbon weights summing to 1.0) with `DEFAULT_STRATEGIES` (`cost_optimized`, `balanced`, `low_carbon`) and deterministic `select_strategy(constraints)` |

## Request flow

```mermaid
sequenceDiagram
    participant U as User
    participant S as Swarm (facade)
    participant B as EventBus
    participant RA as RequirementAgent
    participant ST as StrategyAgent
    participant SDA as SupplierDiscoveryAgent
    participant EA as EvaluationAgent
    participant NA as NegotiationAgent
    participant CT as CompletionTracker
    participant DA as DecisionAgent
    participant RSA as RiskAssessmentAgent
    participant GA as GovernanceAgent
    participant APA as ApprovalAgent
    participant POA as PurchaseOrderAgent
    participant ETA as ExecutionTrackingAgent
    participant STT as SwarmState

    U->>S: send_message("CreateRequirement", payload, correlation_id)
    S->>B: Event.from_message (message + correlation_id)
    B->>STT: record event
    B->>RA: step() → put_artifact(kind=requirement)
    RA->>B: publish(RequirementCreated, correlation_id)
    B->>ST: step() → put_artifact(kind=strategy)
    ST->>B: publish(StrategySelected, correlation_id)
    B->>SDA: step() → put_artifact(kind=supplier_list)
    SDA->>STT: expect_artifact(evaluation, count=5) / expect_artifact(quote, count=5)
    SDA->>B: publish(SupplierDiscovered x5, correlation_id)

    loop For each supplier s
        B->>EA: step() → put_artifact(kind=evaluation, tags={"supplier": s}, parent_ids=[...])
        EA->>B: publish(SupplierEvaluated, correlation_id)
        B->>NA: step() → put_artifact(kind=quote, tags={"supplier": s}, parent_ids=[evaluation_s])
        NA->>B: publish(QuoteGenerated, correlation_id)
    end

    B->>CT: handler() counts artifacts per group
    CT->>B: publish(EvaluationCompleted, correlation_id) [once]
    CT->>B: publish(QuotesCompleted, correlation_id) [once]
    B->>DA: step() → reads all quotes → put_artifact(kind=decision, parent_ids=[...])
    DA->>DA: put_artifact(kind=decision_explanation, parent_ids=[decision])
    DA->>B: publish(DecisionMade, correlation_id)
    B->>RSA: step() → put_artifact(kind=risk_assessment, parent_ids=[decision])
    RSA->>B: publish(RiskAssessmentCompleted, correlation_id)
    B->>GA: step() → put_artifact(kind=governance_decision, parent_ids=[risk])
    GA->>B: publish(GovernanceDecisionMade, correlation_id)
    B->>APA: step() → put_artifact(kind=execution_authorization, parent_ids=[governance])
    APA->>B: publish(ApprovalGranted | ApprovalRequired, correlation_id)
    B->>POA: step() → put_artifact(kind=purchase_order, parent_ids=[authorization])
    POA->>B: publish(PurchaseOrderCreated, correlation_id)
    B->>ETA: step() → put_artifact(kind=execution_status, parent_ids=[order])
    ETA->>B: publish(ExecutionStatusUpdated, correlation_id)
    U->>S: RecordProcurementOutcome (message)
    S->>B: Event.from_message
    B->>OA: step() → put_artifact(kind=procurement_outcome, parent_ids=[decision])
    OA->>B: publish(OutcomeRecorded, correlation_id)
    B->>SIA: step() → put_artifact(kind=supplier_performance, parent_ids=[outcome])
    ```

Every message/event in one logical conversation carries the same
`correlation_id`, so the full exchange can be traced end to end.

## Phase 3: parallel per-supplier flow

The Phase 2 linear pipeline evolved into a parallel, multi-agent flow while
staying fully deterministic and LLM-free:

- **Per-supplier granularity** — `SupplierDiscoveryAgent` announces one
  `SupplierDiscovered` event per supplier; `EvaluationAgent` and
  `NegotiationAgent` run a single dedicated step per supplier
  (`SupplierEvaluated`, `QuoteGenerated`). Suppliers no longer wait on each
  other, and the events are independently traceable.
- **Completion tracking** — the discovery agent declares its expectations with
  `expect_artifact("evaluation", count=5)` / `expect_artifact("quote", count=5)`.
  A `CompletionTracker` subscribed to every event closes a group once the
  expected number of artifacts exists (`is_group_completed`), publishing
  `EvaluationCompleted` then `QuotesCompleted` — exactly once, idempotently.
- **Gate, not broadcast** — `DecisionAgent` subscribes only to
  `QuotesCompleted` and reads all quotes back from shared state, so it cannot
  decide before every supplier has quoted.
- **Capability routing** — `SupplierDiscovered` is delivered through
  `route_on_event(registry.best_for_capability("supplier.evaluate", **tags))`,
  letting a single generalist (or future specialists tagged by region/material)
  win the event without every evaluator reacting to everything.
- **Artifact lineage** — each evaluation artifact lists its requirement
  `parent_ids` and each quote lists its evaluation `parent_ids`, so the
  derivation chain is auditable from `decision` back to `requirement`.
- **Execution trace** — `Swarm.get_execution_trace(correlation_id)` returns an
  ordered `{events, artifacts, agent_actions}` audit trail (agent actions merge
  artifact creation and event publication, excluding runtime-sourced events).
  The `step(event)` protocol — `state` assigned at registration / routing —
  means agents read state without a `step(state, event)` coupling.

## Phase 2 data flow

The domain agents are pure adapters over the existing procurement logic — they
reuse it, they do not reimplement it:

| Domain agent | Reused logic |
|--------------|--------------|
| `RequirementAgent` | `core.protocol.schema.RFQPayload` + the delivery-window / payment-terms defaults from `BuyerOrchestrator.create_rfq`; the market-derived price cap (`spot * 1.2`) from the API layer |
| `SupplierDiscoveryAgent` | `core.market_simulator.MarketSimulator(seed=42)` spot reference + the CostModel supplier templates from `api/main._create_suppliers` |
| `EvaluationAgent` | `core.evaluator.scoring.MultiCriteriaEvaluator` with `configs.settings` weights and ESG baselines |
| `NegotiationAgent` | `swarm.domain.pricing` (same floor-price rule as `core.agents.supplier.SupplierAgent`) |
| `DecisionAgent` | `core.protocol.policy_engine.PolicyEngine` for compliance, then ranks by score and price |

Agents never call each other directly. They only read artifacts from shared
state, produce new artifacts, and announce domain events — so any stage can be
replaced, reordered, or replayed without coupling.

## Hardening guarantees (Phase 1.5)

- **Traceability** — `correlation_id` on both `Message` and `Event`, propagated
  across the message/event boundary and seeded by the coordinator (and by the
  `Swarm` facade's `send_message`).
- **Typed shared state** — arbitrary dict growth replaced by versioned
  `Artifact` objects (who created it, when, and at what version). Artifacts may
  carry `tags` (e.g. `{"supplier": "abc", "region": "EU"}`) and be queried
  selectively via `SwarmState.find_artifacts(kind=..., name=..., tags=...)`.
- **Declarative capabilities** — agents advertise `Capability` schemas with an
  optional `priority`; `by_capability()` returns candidates ranked by priority
  (ties keep registration order) and `best_for_capability()` picks the top one.
- **Replay is read-only** — `EventBus.replay()` (and
  `SwarmCoordinator.replay()`) re-delivers the audit log as `replayed` copies
  that state recorders ignore. Replay is a debugging aid and **never mutates
  canonical state**; use `mode="deliver"` only when rebuilding a fresh runtime.
- **Log level control** — `SWARM_LOG_LEVEL` (default `INFO`) selects the detail:
  `DEBUG` for full event-level logs, `INFO` for lifecycle and high-level routing
  only. Standalone tools call `configure_logging()`; host applications (e.g. the
  FastAPI service) control this through their own structlog config.
- **Role contract** — `Swarm` is the public interface (hold events, agents, and
  state; start / stop / send / replay) and contains no business logic;
  `SwarmCoordinator` is the internal engine and is not called directly by
  external code.
- **Automatic stepping** — registering an agent with a `Swarm` (or
  `SwarmCoordinator`) assigns it the bus and subscribes it via
  `drive_on_event`, so a delivered event runs its whole
  `perceive → reason → act` cycle. Agents therefore never hold a reference to
  the coordinator or to each other.
- **`step(event)` protocol** — agents implement `step(event)` instead of
  `step(state, event=None)`; `BaseAgent.state` is assigned by registration /
  routing / `drive_on_event` / `route_on_event`, and `step` raises
  `RuntimeError` if it was never set. Routing uses `route_on_event(route, state)`
  so a route-selected agent also sees the shared state.
 - **Deterministic domain layer** — the Phase 6 agents are pure, seed-based
   adapters: the market, the supplier pool, the evaluation scores, the quotes,
   strategy selection, risk assessment, governance and approval are all
   deterministic, so the same `CreateRequirement` message always yields the same
   strategy, decision, risk and authorization. Domain agents skip *replayed*
   events in `perceive`, so audit-log replay never re-runs the flow.
 - **Completion tracking** — `SwarmState.expect_artifact(kind, count=..., correlation_id=...)`
   declares group sizes; `CompletionTracker` (subscribed to every event, ignoring
   replayed ones) closes a group once the expected artifact count exists and
   publishes the completion event exactly once. `complete_artifact` /
   `is_group_completed` are idempotent and per-`correlation_id`.
 - **Execution trace** — `Swarm.get_execution_trace(correlation_id)` is read-only:
   it derives `events`, `artifacts` and a chronological `agent_actions` audit
   trail (artifact creation + event publication per agent, runtime-sourced events
   filtered) from canonical state and never mutates it.

## Phase 4: strategy-based evaluation and auditable decisions

The Phase 3 flow gained two deterministic, LLM-free concerns — what to
optimize, and why a particular supplier won:

- **Execution strategy** — `StrategyAgent` reacts to `RequirementCreated` and
  picks a `Strategy` from the requirement's constraints via the pure
  `select_strategy(constraints)` rule: a hard carbon constraint
  (`max_carbon_per_unit`) selects `low_carbon`; a tight budget (below half of
  `quantity * max_unit_price`) selects `cost_optimized`; otherwise `balanced`.
  The strategy is written as a `StrategyArtifact` (its three weights, which sum to
  1.0) and announced with `StrategySelected`.
- **Strategy-gated discovery** — `SupplierDiscoveryAgent` now subscribes to
  `StrategySelected` rather than `RequirementCreated`, so the supplier pool and
  its completion expectations are only created *after* the strategy artifact
  exists. Because the event bus delivers events concurrently, this gate removes
  the race where evaluation could run before the weights it depends on. In the
  `balanced` strategy the weights reproduce the Phase 3 composite scores
  exactly; when no strategy artifact is present an evaluation falls back to
  `balanced`, so existing score assertions are preserved.
- **Weighted evaluation** — `EvaluationAgent` blends the price, quality and
  carbon sub-scores from the existing `MultiCriteriaEvaluator` with the active
  strategy's weights, recording the strategy used in each `EvaluationArtifact`.
- **Auditable decision reasoning** — `DecisionAgent` produces, alongside the
  `DecisionArtifact`, a `DecisionExplanationArtifact` (`selected_supplier`,
  `strategy_used`, `top_factors`, and a `rejected_suppliers` list with a
  deterministic textual reason for every non-selected supplier derived from the
  policy engine's rejection reason and score/price ordering).

This keeps the runtime deterministic and LLM-free while making *what* is
optimized and *why* it was chosen explicit and traceable through the same
artifact/event lineage.

## Phase 6: deterministic governance, risk and approval

Phase 5 answered *"who should we buy from?"* and fed the result back into
supplier memory. Phase 6 adds the control layer that answers *"is this decision
safe and authorized to execute?"* — a deterministic governance tail that runs
after the decision and never blocks the existing Phase 1–5 flow:

- **Risk assessment** — `RiskAssessmentAgent` reacts to `DecisionMade`, loads the
  `DecisionArtifact`, the requirement, the selected supplier's `QuoteArtifact`,
  its `SupplierPerformance` history and the active `GovernancePolicy`, and
  computes four deterministic sub-scores (financial / delivery / quality /
  carbon) blended into an `overall_risk_score` and an ordinal `risk_level`
  (`LOW` / `MEDIUM` / `HIGH` / `CRITICAL`). It writes a `RiskAssessmentArtifact`
  (parented to `DecisionArtifact.id`) and publishes `RiskAssessmentCompleted`.
  No history yields neutral defaults, so a clean decision stays `LOW`.
- **Governance** — `GovernanceAgent` applies the static `GovernancePolicy`
  (`standard_policy` / `strict_policy`) to the risk assessment via the pure
  `GovernanceDecision.from_risk`, producing `APPROVED` / `APPROVAL_REQUIRED` /
  `REJECTED`. It writes a `GovernanceDecisionArtifact` (parented to
  `RiskAssessmentArtifact.id`) and publishes `GovernanceDecisionMade`.
- **Approval** — `ApprovalAgent` closes the gate: `APPROVED` grants an
  `ExecutionAuthorizationArtifact` (`authorized`) immediately; `APPROVAL_REQUIRED`
  records a `pending` authorization resolved by `POST /swarm/{request_id}/approve`
  (deterministic simulated approval — no autonomous approval); `REJECTED` blocks
  execution and publishes `ApprovalRejected`. The authorization is parented to
  `GovernanceDecisionArtifact.id`.

Full lineage: `RequirementArtifact → StrategyArtifact → SupplierListArtifact →
EvaluationArtifact → QuoteArtifact → DecisionArtifact → RiskAssessmentArtifact →
GovernanceDecisionArtifact → ExecutionAuthorizationArtifact`. Each link is by
`Artifact.id`, and all three new agents skip *replayed* events so audit-log
replay stays read-only and `test_full_procurement_flow_is_replay_safe` keeps
passing.

## Phase 7: execution & procurement operations (control → action)

Phase 6 answered *"is this decision safe and authorized to execute?"* Phase 7
adds the action tier that answers *"how do we safely execute the approved
decision?"* — a deterministic execution tail that runs **only** after
authorization is granted and never bypasses governance:

- **Purchase order** — `PurchaseOrderAgent` (capability `procurement.order.create`)
  reacts to `ApprovalGranted` (and the `Execute` intent). It loads the
  `ExecutionAuthorizationArtifact` — whose status must be `authorized`; a pending
  or rejected decision produces **no** order — builds a deterministic
  `PurchaseOrder` from the decision, authorization, requirement and winning
  `QuoteArtifact`, and submits it through a `SupplierConnector`. The
  `MockSupplierConnector` (replaceable later by `SAPConnector`, `OracleConnector`,
  `CoupaConnector`) records the deterministic `SUBMITTED` status. The
  `PurchaseOrderArtifact` is parented to `ExecutionAuthorizationArtifact.id`.
- **Execution tracking** — `ExecutionTrackingAgent` (capability `procurement.execution.track`)
  reacts to `PurchaseOrderCreated`, asks the connector for the order's realized
  status and lifecycle, and writes an `ExecutionStatusArtifact` (parented to
  `PurchaseOrderArtifact.id`) publishing `ExecutionStatusUpdated`. The same order
  always maps to the same status record, keeping the trace replay-safe.
- **Explicit execution** — `POST /swarm/{request_id}/execute` resolves a *pending*
  authorization (deterministic simulated approval) and then creates the order and
  tracks it on the remembered state; the call is idempotent, so an already-executed
  run returns the existing order status.

Execution is gated end-to-end: `authorized` → order created → status tracked;
`pending` → no order until approval + `/execute`; `rejected` (which never
produces an authorization) → no order. The execution tier only *records* what the
deterministic connector reports — there is no LLM and no autonomous action.

Full lifecycle lineage: `... → ExecutionAuthorizationArtifact →
PurchaseOrderArtifact → ExecutionStatusArtifact → ProcurementOutcomeArtifact →
SupplierPerformanceArtifact`.

## Phase 8: Enterprise Integration Layer (in progress)

Phase 7 turned an authorized decision into a deterministic order and tracked it.
Phase 8 is the boundary crossing into *real external systems* — **without**
relinquishing the swarm's determinism, replay safety, idempotency or source-of-truth
guarantees, and **without** altering governance/approval logic:

- **Contract pre-gate** — `ContractValidationAgent` consumes `DecisionMade` and
  applies the active `Contract` set (supplier coverage, allowed items, pricing
  caps, active/expiry/compliance). A valid decision publishes
  `ContractValidated`; an invalid/expired/missing-required contract publishes
  `ContractRejected` which short-circuits straight to a `REJECTED`
  `GovernanceDecision` (risk is skipped entirely). With no contract on file and
  `require_contract=False` (the default) the decision is treated as valid, so the
  gate is opt-in. The post-decision chain is now **Decision → Contract → Risk →
  Governance**; `RiskAssessmentAgent` triggers on `ContractValidated`.
- **BaseConnector port** — `swarm/integrations/base.py` defines the deterministic
  adapter interface (`submit_order -> ExternalResponse`,
  `get_order_status -> ExternalStatus`, `validate_supplier -> bool`). Adapters are
  *translators only*: they map `PurchaseOrder` to an external schema and back to the
  normalized response; they hold no swarm state and never mutate the artifact graph.
  `MockConnector`, `SupplierAPIConnector` and the `SAPConnector` / `OracleConnector`
  / `CoupaConnector` adapters all implement it.
- **Audited side effects** — every outbound call produces an
  `ExternalCallArtifact` (kind `external_call`) carrying `{system, action,
  request_payload, response_payload, status, idempotency_key, timestamp}` and
  lineage to the originating decision.
- **Idempotency** — `swarm/utils/idempotency.py` (`IdempotencyGuard`) deduplicates
  every operation by a deterministic `(decision_id, action)` key, so re-running an
  order submission or a reconciliation pass never duplicates an external side
  effect.
- **Replay safety** — the runtime skips `act` for replayed events, so connectors are
  never re-invoked on replay; the guard is a second defence for the API-driven
  `/execute` and `/sync` path that calls agents directly.

New API: `GET /swarm/external/{request_id}` (external-call audit trail) and
`POST /swarm/{request_id}/sync` (idempotent external reconciliation); `POST
/swarm/{request_id}/execute` now routes through the base connector and reports its
external calls. `build_procurement_swarm` gains `base_connector`, `contracts` and
`require_contract` parameters.

## Phase 5: deterministic supplier intelligence and outcome feedback

Phase 5 makes the swarm learn from the outcome **deterministically** — closing the loop from
`DecisionArtifact` → `ProcurementOutcome` → `SupplierPerformance`, without any
LLM and without breaking Phase 4's scores.

- **Outcome capture** — `OutcomeAgent` handles `RecordProcurementOutcome`,
  validates the payload against the remembered decision, and writes an
  `OutcomeArtifact` parented to that `DecisionArtifact.id` (lineage:
  `DecisionArtifact.id` → `OutcomeArtifact.parent_ids` →
  `SupplierPerformanceArtifact.parent_ids`).
- **Supplier memory** — `SupplierIntelligenceAgent` reacts to `OutcomeRecorded`,
  updates the shared `SupplierMemoryStore` via `update_from_outcome(...)`, and
  writes a `SupplierPerformanceArtifact` (running-averages for delivery, quality,
  price-competitiveness and carbon — no free parameters, no embeddings).
- **History-weighted evaluation** — `EvaluationAgent` now accepts a `memory`
  store; after the strategy-weighted composite, it applies
  `history_adjustment(perf)` to the reliability term, clamped to `[0, 1]` and
  rounded to 4dp. When no memory is supplied the adjustment is `0.0`, so the
  Phase 4 balanced scores are reproduced exactly and existing assertions are
  preserved.
- **API + wiring** — `build_procurement_swarm(..., supplier_memory=...)` shares a
  module-level `default_store` across requests; `POST /swarm/{request_id}/outcome`
  and `GET /swarm/supplier/{supplier_id}/performance` expose the feedback loop,
  and replay skips `RecordProcurementOutcome` so re-runs stay deterministic.

Determinism caveat: the metrics are running means, so they are deterministic
given the order of recorded outcomes; only `last_updated` timestamps vary with
wall-clock time and tests avoid asserting timestamp equality.

## Outlook

Phase 7 is complete: the runtime is now a deterministic **Autonomous Procurement
Operating System** — parallel, per-supplier execution, strategy-aware
evaluation, an auditable supplier-intelligence feedback loop, a governance + risk
layer that gates whether a decision is safe and authorized to execute, and a
deterministic execution tier that turns an authorized decision into a traced
purchase order and tracks it to delivery. The maturity is:

```
Intelligence:   Strategy  +  Feedback
Decision:       Governance
Control:        Risk  +  Approval
Action:         Order  +  Execution  +  Monitoring
```

What is deliberately left for later:

- **Live integration** — replace the seeded `MarketSimulator` reference with the
  live market feed and wire `DecisionMade` into the ledger / API, so the swarm
  becomes the auction engine's orchestration path. The read-only swarm trace
  endpoints in `api/main.py` already expose dispatch (`POST /swarm/requirements`),
  the governance tail (`GET /swarm/risk/{request_id}`,
  `GET /swarm/governance/{request_id}`, `POST /swarm/{request_id}/approve`), the
  execution tier (`POST /swarm/{request_id}/execute`,
  `GET /swarm/order/{request_id}`, `GET /swarm/execution/{request_id}`), and trace
  reads (`GET /swarm/trace/{request_id}`, `/completions`, `/swarm/state/{request_id}`).
- **Planning** — an explicit planner (task decomposition, capability routing via
  `best_for_capability()`) that assembles the agent chain from intent rather than
  a hard-coded registration order; specialist evaluators tagged by region or
  material would then be selected by `route_on_event`.
- **Robustness** — retries, sagas/compensation for failed stages, and durable
  event sourcing in place of the in-memory audit log.
- **LLM integration** — remains explicitly out of scope by design. The runtime
  is built so a single, guarded adapter could later assist with *negotiation
  language, supplier communication, contract analysis and market intelligence* —
  never with policy enforcement, compliance, approval or financial controls,
  which stay deterministic.

The next architectural phase is **v0.8 — Enterprise Integration Layer** (enterprise
connectors, ERP/supplier APIs, contracts, audited + idempotent external side
effects), now in progress; only after that **v0.9 — Cognitive Procurement Agents** (LLM-assisted negotiation and contract reasoning).
The architecture is now mature enough that adding LLMs later will make it stronger
instead of making it fragile.

