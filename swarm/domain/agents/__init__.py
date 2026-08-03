"""Phase 2 adapter agents that wrap the existing ``core/`` procurement logic.

Each agent is deterministic and communicates exclusively through the swarm
runtime: it reads artifacts from shared state and announces domain events —
never holding a direct reference to another agent.
"""

from swarm.domain.agents.decision_agent import DecisionAgent
from swarm.domain.agents.evaluation_agent import EvaluationAgent
from swarm.domain.agents.negotiation_agent import NegotiationAgent
from swarm.domain.agents.requirement_agent import RequirementAgent
from swarm.domain.agents.supplier_discovery_agent import SupplierDiscoveryAgent

__all__ = [
    "DecisionAgent",
    "EvaluationAgent",
    "NegotiationAgent",
    "RequirementAgent",
    "SupplierDiscoveryAgent",
]
