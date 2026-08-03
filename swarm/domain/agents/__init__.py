"""Phase 2 adapter agents that wrap the existing ``core/`` procurement logic.

Each agent is deterministic and communicates exclusively through the swarm
runtime: it reads artifacts from shared state and announces domain events —
never holding a direct reference to another agent.
"""

from swarm.domain.agents.approval_agent import ApprovalAgent
from swarm.domain.agents.decision_agent import DecisionAgent
from swarm.domain.agents.evaluation_agent import EvaluationAgent
from swarm.domain.agents.execution_agent import ExecutionTrackingAgent
from swarm.domain.agents.governance_agent import GovernanceAgent
from swarm.domain.agents.negotiation_agent import NegotiationAgent
from swarm.domain.agents.outcome_agent import OutcomeAgent
from swarm.domain.agents.purchase_order_agent import PurchaseOrderAgent
from swarm.domain.agents.requirement_agent import RequirementAgent
from swarm.domain.agents.risk_agent import RiskAssessmentAgent
from swarm.domain.agents.strategy_agent import StrategyAgent
from swarm.domain.agents.supplier_discovery_agent import SupplierDiscoveryAgent
from swarm.domain.agents.supplier_intelligence_agent import SupplierIntelligenceAgent

__all__ = [
    "ApprovalAgent",
    "DecisionAgent",
    "EvaluationAgent",
    "ExecutionTrackingAgent",
    "GovernanceAgent",
    "NegotiationAgent",
    "OutcomeAgent",
    "PurchaseOrderAgent",
    "RequirementAgent",
    "RiskAssessmentAgent",
    "StrategyAgent",
    "SupplierDiscoveryAgent",
    "SupplierIntelligenceAgent",
]
