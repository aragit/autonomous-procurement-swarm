"""
Agent definitions for the procurement swarm.

The buyer and supplier agents for the CNP auction protocol.
"""

from core.agents.buyer import BuyerOrchestrator
from core.agents.supplier import CostModel, SupplierAgent

__all__ = ["BuyerOrchestrator", "SupplierAgent", "CostModel"]
