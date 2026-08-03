# Changelog

Release history for the Autonomous Procurement Swarm. Tags follow a phase-driven
`v0.x` scheme; `v0.2` was intentionally skipped — its concerns (Swarm Foundation)
and (Procurement Agent Architecture) were delivered together in `v0.1`.

## v0.6 — Governance + Risk-Aware Procurement (commit TBD)

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
