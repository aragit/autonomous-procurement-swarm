"""Assembly of the full Phase 4 procurement swarm.

``build_procurement_swarm`` wires the deterministic agents together with a
:class:`CompletionTracker` so the linear Phase 2 pipeline becomes a parallel,
per-supplier multi-agent flow:

- ``RequirementAgent``  — listens for ``CreateRequirement`` messages
- ``StrategyAgent``     — picks the execution strategy from the requirement and
  publishes ``StrategySelected`` before any supplier is discovered
- ``SupplierDiscoveryAgent`` — publishes one ``SupplierDiscovered`` per supplier
  and declares the evaluation/quote completion expectations
- ``EvaluationAgent``   — evaluates each discovered supplier (routed by the
  ``supplier.evaluate`` capability, so specialized evaluators can compete)
- ``NegotiationAgent``  — quotes each evaluated supplier
- ``DecisionAgent``     — decides only after ``QuotesCompleted`` fires
- ``CompletionTracker`` — closes a phase once every expected artifact exists and
  publishes ``EvaluationCompleted`` / ``QuotesCompleted``

The only public entry point is the returned :class:`Swarm` facade; callers
drive it with ``send_message(CREATE_REQUIREMENT_INTENT, payload)`` and read the
result through ``swarm.state``.
"""

from collections.abc import Callable

from swarm import Swarm
from swarm.core.agent import BaseAgent
from swarm.core.completion import CompletionTracker
from swarm.core.event import ANY_EVENT, Event, SwarmEventType
from swarm.domain.agents import (
    DecisionAgent,
    EvaluationAgent,
    NegotiationAgent,
    OutcomeAgent,
    RequirementAgent,
    StrategyAgent,
    SupplierDiscoveryAgent,
    SupplierIntelligenceAgent,
)
from swarm.domain.events import ProcurementEventType
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
        OutcomeAgent(),
        event_types=[SwarmEventType.MESSAGE],
    )
    intelligence_agent = SupplierIntelligenceAgent(memory=memory)
    swarm.register(
        intelligence_agent,
        event_types=[ProcurementEventType.OUTCOME_RECORDED],
    )
    return swarm
