"""Mesh archetype actors for the Neuro-Symbolic Procurement Mesh.

This package contains the four Ray actor archetypes:
- ScoutActor: Discovers suppliers (writes DISCOVERY)
- EvaluatorActor: Scores suppliers + risk (writes SCORE, RISK)
- NegotiatorActor: Generates quotes (writes DEAL)
- BuyerActor: Deterministic MCDA decision (writes DECISION) — SINGLETON
"""

from mesh.actors.base import MeshActor, NeuralProposal, SafetyKernelActor, SymbolicVerdict
from mesh.actors.buyer import BuyerActor
from mesh.actors.evaluator import EvaluatorActor
from mesh.actors.negotiator import NegotiatorActor
from mesh.actors.scout import ScoutActor

__all__ = [
    "MeshActor",
    "NeuralProposal",
    "SymbolicVerdict",
    "SafetyKernelActor",
    "ScoutActor",
    "EvaluatorActor",
    "NegotiatorActor",
    "BuyerActor",
]
