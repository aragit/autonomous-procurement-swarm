# Changelog

Release history for the Autonomous Procurement Swarm. Tags follow a phase-driven
`v0.x` scheme; `v0.2` was intentionally skipped — its concerns (Swarm Foundation)
and (Procurement Agent Architecture) were delivered together in `v0.1`.

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
