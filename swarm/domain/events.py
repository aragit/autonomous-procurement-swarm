"""Procurement domain event types for the Phase 3 adapter swarm.

The runtime's :class:`SwarmEventType` covers lifecycle events; the events below
are the domain facts the five procurement agents exchange. Every such event
carries a ``correlation_id`` (propagated from the originating request) and
references the artifact(s) it produced, so each stage is fully traceable.

Phase 3 splits the batch events of Phase 2 into per-supplier events
(``SupplierDiscovered`` / ``SupplierEvaluated`` / ``QuoteGenerated``) plus
phase-gate completion events (``EvaluationCompleted`` / ``QuotesCompleted``)
published by the :class:`CompletionTracker` once every expected artifact for a
request exists. Downstream agents subscribe to the per-supplier events to run
in parallel and to the completion events to know when a phase is fully done.
"""

from enum import StrEnum

CREATE_REQUIREMENT_INTENT = "CreateRequirement"


class ProcurementEventType(StrEnum):
    """Domain events published by the procurement agents."""

    REQUIREMENT_CREATED = "RequirementCreated"
    STRATEGY_SELECTED = "StrategySelected"
    SUPPLIER_DISCOVERED = "SupplierDiscovered"
    SUPPLIER_EVALUATED = "SupplierEvaluated"
    QUOTE_GENERATED = "QuoteGenerated"
    EVALUATION_COMPLETED = "EvaluationCompleted"
    QUOTES_COMPLETED = "QuotesCompleted"
    DECISION_MADE = "DecisionMade"
