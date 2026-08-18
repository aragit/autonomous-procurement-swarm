"""Neuro-Symbolic bridge package (ray-free, unit-testable).

Public API:
    NeuroSymbolicBridge   structured generation + SafetyKernel retry loop
    NeuroResult           outcome of a (retried) proposal
    ProtocolViolation     raised on exhausted retries (opt-in)
    LLMConfig             OpenAI-compatible backend configuration
    OpenAICompatibleBackend   vLLM / Ollama / llama.cpp structured backend
    MockNeuroBackend      deterministic backend for tests
    StructuredBackend     abstract backend contract
    NeuralProposal / SymbolicVerdict   shared neuro-symbolic contract types
    ScoutProposal / NegotiatorProposal   schema-constrained output models
"""

from mesh.neuro.backend import (
    LLMConfig,
    MockNeuroBackend,
    OpenAICompatibleBackend,
    StructuredBackend,
)
from mesh.neuro.bridge import (
    KernelValidator,
    NeuroResult,
    NeuroSymbolicBridge,
    ProtocolViolation,
)
from mesh.neuro.kernel import symbolic_validate
from mesh.neuro.schemas import (
    NegotiatorProposal,
    NegotiatorQuote,
    QuoteMetadata,
    ScoutProposal,
    SupplierDiscoveryItem,
)
from mesh.neuro.types import NeuralProposal, SymbolicVerdict

__all__ = [
    "NeuroSymbolicBridge",
    "NeuroResult",
    "ProtocolViolation",
    "KernelValidator",
    "symbolic_validate",
    "LLMConfig",
    "OpenAICompatibleBackend",
    "MockNeuroBackend",
    "StructuredBackend",
    "NeuralProposal",
    "SymbolicVerdict",
    "SupplierDiscoveryItem",
    "ScoutProposal",
    "NegotiatorProposal",
    "NegotiatorQuote",
    "QuoteMetadata",
]
