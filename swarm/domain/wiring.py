"""Assembly of the full Phase 7 procurement swarm.

``build_procurement_swarm`` wires the deterministic agents together with a
:class:`CompletionTracker` so the linear Phase 2 pipeline becomes a parallel,
per-supplier multi-agent flow with a governance + execution tail:

- ``RequirementAgent``  — listens for ``CreateRequirement`` messages
- ``StrategyAgent``     — picks the execution strategy from the requirement and
  publishes ``StrategySelected`` before any supplier is discovered
- ``SupplierDiscoveryAgent`` — publishes one ``SupplierDiscovered`` per supplier
  and declares the evaluation/quote completion expectations
- ``EvaluationAgent``   — evaluates each discovered supplier (routed by the
  ``supplier.evaluate`` capability, so specialized evaluators can compete)
- ``NegotiationAgent``  — quotes each evaluated supplier
- ``DecisionAgent``     — decides only after ``QuotesCompleted`` fires
- ``RiskAssessmentAgent`` — assesses the selected decision (subscribes to ``ContractValidated``)
- ``GovernanceAgent``   — applies governance policy to the risk assessment
  (also short-circuits a ``ContractRejected`` decision to REJECTED)
- ``ApprovalAgent``     — closes the gate into an execution authorization
 - ``PurchaseOrderAgent`` — creates a purchase order once a decision is ``authorized``
 - ``ExecutionTrackingAgent`` — tracks the order's lifecycle to delivery
 - ``OutcomeAgent``      — records the realized procurement outcome
 - ``SupplierIntelligenceAgent`` — folds the outcome into deterministic supplier history
 - ``ContractValidationAgent`` — applies supplier contracts to a decision
   (subscribes to ``DecisionMade``; emits ``ContractValidated``/``ContractRejected``)
 - ``CompletionTracker`` — closes a phase once every expected artifact exists and
   publishes ``EvaluationCompleted`` / ``QuotesCompleted``

The only public entry point is the returned :class:`Swarm` facade; callers drive
it with ``send_message(CREATE_REQUIREMENT_INTENT, payload)`` and read the result
through ``swarm.state``. ``build_procurement_swarm(..., supplier_memory=...)``
shares deterministic supplier history across the evaluation and risk stages;
``governance_policy`` selects the policy applied to each decision;
``supplier_connector`` supplies the deterministic ERP/supplier adapter used to
submit and track purchase orders.

``contracts`` is an optional mapping of ``{supplier_id: Contract}`` applied to
every decision (contract compliance gate between decision and risk). When
``require_contract`` is False (default) a supplier without a contract is
treated as valid; when True a missing contract short-circuits the decision to
REJECTED.
"""

from collections.abc import Callable

from swarm import Swarm
from swarm.core.agent import BaseAgent
from swarm.core.completion import CompletionTracker
from swarm.core.event import ANY_EVENT, Event, SwarmEventType
from swarm.domain.agents import (
    ApprovalAgent,
    ContractValidationAgent,
    DecisionAgent,
    EvaluationAgent,
    ExecutionTrackingAgent,
    GovernanceAgent,
    NegotiationAgent,
    OutcomeAgent,
    PurchaseOrderAgent,
    RequirementAgent,
    RiskAssessmentAgent,
    StrategyAgent,
    SupplierDiscoveryAgent,
    SupplierIntelligenceAgent,
)
from swarm.domain.contracts import Contract
from swarm.domain.events import ProcurementEventType
from swarm.domain.governance import STANDARD_POLICY, GovernancePolicy
from swarm.domain.order import SupplierConnector, default_connector
from swarm.integrations.base import BaseConnector
from swarm.integrations.mock import MockConnector
from swarm.memory import SupplierMemoryStore

COMPLETION_EVENTS = {
    "evaluation": ProcurementEventType.EVALUATION_COMPLETED,
    "quote": ProcurementEventType.QUOTES_COMPLETED,
}

COMPLETION_SOURCE = "completion_tracker"


def _select_evaluator(swarm: Swarm) -> Callable[[Event], BaseAgent | None]:
    """Route ``SupplierDiscovered`` events to the best evaluation agent.

    Uses capability + tag matching on the shared registry, so an EU-specialized
    evaluator would only win for EU-supplier events while generalists cover the
    rest. With a single generalist evaluator this always picks it.
    """

    def select(event: Event) -> BaseAgent | None:
        return swarm.registry.best_for_capability("supplier.evaluate")

    return select


def build_procurement_swarm(
    *,
    request_id: str = "",
    goal: str = "",
    supplier_memory: SupplierMemoryStore | None = None,
    governance_policy: GovernancePolicy | None = None,
    supplier_connector: SupplierConnector | None = None,
    base_connector: BaseConnector | None = None,
    contracts: dict[str, Contract] | None = None,
    require_contract: bool = False,
) -> Swarm:
    """Create a wired procurement swarm with completion tracking enabled."""
    swarm = Swarm(request_id=request_id, goal=goal)
    tracker = CompletionTracker(
        swarm.state,
        swarm.bus,
        completion_events=COMPLETION_EVENTS,
    )
    swarm.bus.subscribe(ANY_EVENT, tracker.handler)
    swarm.tracker = tracker  # type: ignore[attr-defined]

    memory = supplier_memory if supplier_memory is not None else SupplierMemoryStore()
    swarm.supplier_memory = memory  # type: ignore[attr-defined]
    policy = governance_policy if governance_policy is not None else STANDARD_POLICY
    swarm.governance_policy = policy  # type: ignore[attr-defined]
    connector = supplier_connector if supplier_connector is not None else default_connector
    swarm.supplier_connector = connector  # type: ignore[attr-defined]
    base = base_connector if base_connector is not None else MockConnector()
    swarm.base_connector = base  # type: ignore[attr-defined]
    swarm.contracts = dict(contracts or {})  # type: ignore[attr-defined]
    swarm.require_contract = require_contract  # type: ignore[attr-defined]

    swarm.register(RequirementAgent(), event_types=[SwarmEventType.MESSAGE])
    swarm.register(
        StrategyAgent(),
        event_types=[ProcurementEventType.REQUIREMENT_CREATED],
    )
    swarm.register(
        SupplierDiscoveryAgent(),
        event_types=[ProcurementEventType.STRATEGY_SELECTED],
    )
    evaluation_agent = EvaluationAgent()
    evaluation_agent.memory = memory
    swarm.register(
        evaluation_agent,
        event_types=[ProcurementEventType.SUPPLIER_DISCOVERED],
        route=_select_evaluator(swarm),
    )
    swarm.register(
        NegotiationAgent(),
        event_types=[ProcurementEventType.SUPPLIER_EVALUATED],
    )
    swarm.register(
        DecisionAgent(),
        event_types=[ProcurementEventType.QUOTES_COMPLETED],
    )
    swarm.register(
        ContractValidationAgent(
            contracts=dict(contracts or {}),
            require_contract=require_contract,
        ),
        event_types=[ProcurementEventType.DECISION_MADE],
    )
    swarm.register(
        RiskAssessmentAgent(memory=memory, policy=policy),
        event_types=[ProcurementEventType.CONTRACT_VALIDATED],
    )
    swarm.register(
        GovernanceAgent(policy=policy),
        event_types=[
            ProcurementEventType.RISK_ASSESSMENT_COMPLETED,
            ProcurementEventType.CONTRACT_REJECTED,
        ],
    )
    swarm.register(
        ApprovalAgent(),
        event_types=[ProcurementEventType.GOVERNANCE_DECISION_MADE, SwarmEventType.MESSAGE],
    )
    swarm.register(
        PurchaseOrderAgent(connector=connector, base_connector=base),
        event_types=[ProcurementEventType.APPROVAL_GRANTED, SwarmEventType.MESSAGE],
    )
    swarm.register(
        ExecutionTrackingAgent(connector=connector, base_connector=base),
        event_types=[ProcurementEventType.PURCHASE_ORDER_CREATED],
    )
    swarm.register(
        OutcomeAgent(),
        event_types=[SwarmEventType.MESSAGE],
    )
    intelligence_agent = SupplierIntelligenceAgent(memory=memory)
    swarm.register(
        intelligence_agent,
        event_types=[ProcurementEventType.OUTCOME_RECORDED],
    )
    return swarm
