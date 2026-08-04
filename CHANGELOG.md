# Changelog

Release history for the Autonomous Procurement Swarm. Tags follow a phase-driven
`v0.x` scheme; `v0.2` was intentionally skipped — its concerns (Swarm Foundation)
and (Procurement Agent Architecture) were delivered together in `v0.1`.

## v0.8 — Enterprise Integration Layer (commit TBD)

This phase crosses the deterministic boundary: the swarm still owns source of
truth and approval/governance logic is unchanged, but outbound calls to external
systems are now audited, idempotent, and replay-safe.

Added:
- `BaseConnector` port (`swarm/integrations/base.py`): a pure, deterministic
  interface (`submit_order -> ExternalResponse`, `get_order_status ->
  ExternalStatus`, `validate_supplier -> bool`) plus `ExternalResponse` /
  `ExternalStatus` normalized dataclasses. Every call returns a deterministic
  shape and simulates a response when no live credentials are configured.
- `MockConnector` (`swarm/integrations/mock.py`): the canonical in-memory adapter
  implementing `BaseConnector` (deterministic lifecycle SUBMITTED -> CONFIRMED ->
  SHIPPED -> DELIVERED); supersedes the role of `MockSupplierConnector` for the
  new integration layer.
- `SupplierAPIConnector` (`swarm/integrations/supplier_api.py`): stateless
  supplier-order adapter (simulated HTTP surface, deterministic per order id,
  `validate_supplier` rejects an `invalid_` prefix).
- ERP adapters (`swarm/integrations/erp/`): `SAPConnector`, `OracleConnector`,
  `CoupaConnector` with a `ConnectorConfig` (provider/endpoint/credentials/
  environment); deterministic simulation when unconfigured, live-API shape when
  configured.
- `ExternalCallArtifact` (kind `external_call`) on the artifact graph — every
  external invocation records {system, action, request_payload, response_payload,
  status, idempotency_key, timestamp} with lineage to the originating decision.
- Idempotency layer (`swarm/utils/idempotency.py`): `IdempotencyGuard` keyed by
  deterministic `(decision_id, action)` so the same operation never performs a
  duplicate external side effect.
- `ContractValidationAgent` + `Contract` model: a contract pre-gate between
  decision and risk. `DecisionMade -> ContractValidated` (valid or no-contract)
  proceeds to risk; `ContractRejected` (invalid/expired/requires-contract)
  short-circuits straight to a `REJECTED` governance decision, skipping risk.
- API: `GET /swarm/external/{request_id}` (external-call audit trail) and
  `POST /swarm/{request_id}/sync` (idempotent external reconciliation);
  `POST /swarm/{request_id}/execute` now uses the base connector and reports
  its external calls.

Changed:
- `RiskAssessmentAgent` now triggers on `ContractValidated` (was `DecisionMade`),
  reordering the post-decision chain to
  `Decision -> Contract Validation -> Risk -> Governance`.
- `GovernanceAgent` additionally handles `ContractRejected`, producing a `REJECTED`
  decision directly (the pure `GovernanceDecision.from_risk` / approval logic is
  untouched).
- `PurchaseOrderAgent` / `ExecutionTrackingAgent` accept an optional
  `base_connector: BaseConnector`; when set they use it (deduplicated via the
  idempotency guard, audited via `ExternalCallArtifact`); otherwise the legacy
  `SupplierConnector` path is unchanged. `build_procurement_swarm` gained
  `base_connector` and `contracts`/`require_contract` parameters.
- Replay safety: replayed events never reach `act`, so connectors are never
  re-invoked on replay; the idempotency guard is a second defence for the
  API-driven execution path.

## v0.7 — Execution & Procurement Operations (a7ea64f)

Added:
- Deterministic purchase-order domain (`PurchaseOrder`, `PurchaseStatus`,
  `OrderLine`) and a `SupplierConnector` protocol with a deterministic
  `MockSupplierConnector` (submit → tracking lifecycle SUBMITTED →
  CONFIRMED → SHIPPED → DELIVERED).
- `PurchaseOrderAgent` (capability `procurement.order.create`): consumes
  `ApprovalGranted`, writes a `PurchaseOrderArtifact` lineaged to the
  `ExecutionAuthorizationArtifact` — only when the decision is `authorized`
  (pending / rejected decisions produce no order).
- `ExecutionTrackingAgent` (capability `procurement.execution.track`): consumes
  `PurchaseOrderCreated`, records an `ExecutionStatusArtifact` (terminal status +
  deterministic lifecycle) via the connector.
- API: `POST /swarm/{request_id}/execute` (resolve a pending authorization into an
  order + tracking status, idempotent), `GET /swarm/order/{request_id}`,
  `GET /swarm/execution/{request_id}`.

The swarm now runs the complete procurement lifecycle deterministically —
requirement → strategy → discovery → evaluation → negotiation → decision →
risk → governance → approval → authorization → **order → execution → completion** —
with no LLM and no autonomous approval. The full artifact lineage is now:
`Decision` → `RiskAssessment` → `GovernanceDecision` → `ExecutionAuthorization`
→ `PurchaseOrder` → `ExecutionStatus` → `Outcome`.

## v0.6 — Governance + Risk-Aware Procurement (73ce35c)

Added:
- Deterministic risk assessment layer (`RiskAssessmentAgent`, `RiskAssessmentArtifact`):
  financial / delivery / quality / carbon sub-scores blended into an
  `overall_risk_score` and `RiskLevel` (LOW / MEDIUM / HIGH / CRITICAL).
- Governance policy model (`GovernancePolicy`, `standard_policy`, `strict_policy`)
  and `GovernanceAgent` producing `GovernanceDecisionArtifact`
  (APPROVED / APPROVAL_REQUIRED / REJECTED).
- Approval workflow (`ApprovalAgent`, `ExecutionAuthorizationArtifact`):
  immediate authorization, pending approval, and explicit rejection;
  deterministic simulated approval via `POST /swarm/{request_id}/approve`.
- API: `GET /swarm/risk/{request_id}`, `GET /swarm/governance/{request_id}`,
  `GET /swarm/authorization/{request_id}`, `POST /swarm/{request_id}/approve`.
- Complete artifact lineage through the control layer:
  `DecisionArtifact` → `RiskAssessmentArtifact` → `GovernanceDecisionArtifact`
  → `ExecutionAuthorizationArtifact`.

The swarm now answers "Who should we buy from?" **and** "Is this decision safe
and authorized to execute?" — deterministically, with no LLM and no autonomous
approval.

## v0.5 — Feedback Intelligence (e5e9ff4)

Added:
- Deterministic supplier performance memory (`SupplierMemoryStore`).
- Outcome feedback loop: `OutcomeAgent` + `SupplierIntelligenceAgent`.
- `OutcomeArtifact` and `SupplierPerformanceArtifact` (lineage:
  `DecisionArtifact.id` → `OutcomeArtifact` → `SupplierPerformanceArtifact`).
- `EvaluationAgent` history adjustment (0.0 with no memory ⇒ prior scores unchanged).
- API: `POST /swarm/{request_id}/outcome`, `GET /swarm/supplier/{supplier_id}/performance`.

## v0.4 — Strategy Intelligence (fb1eb83)

Added:
- `StrategyAgent` + pure `select_strategy(constraints)`.
- `StrategyArtifact` (price/score/carbon weights summing to 1.0).
- `DecisionExplanationArtifact` (auditable, deterministic decision reasoning).
- Strategy-gated supplier discovery.

## v0.3 — Parallel Multi-Agent Execution (ee646a7)

Added:
- Per-supplier parallel evaluation steps.
- Completion tracking (`CompletionTracker`, `expect_artifact`).
- Read-only execution trace API + event/artifact lineage.

## v0.1 — Swarm Foundation (a6131ae) — _no v0.2_

Added:
- Event-driven swarm runtime (`Swarm`, `SwarmCoordinator`, `SwarmState`).
- Artifact + agent model (`BaseAgent.step(event)` protocol, `route_on_event`).
- Procurement agent foundation (Requirement, SupplierDiscovery, Evaluation,
  Quote, Negotiation, Decision).

> The next architectural phase is `v0.6 — Governance + Risk-Aware Procurement`.
