"""Unit tests for the Neuro-Symbolic retry / auto-correction loop.

These exercise :class:`NeuroSymbolicBridge` against the **real**
:func:`mesh.neuro.kernel.symbolic_validate` policy and a :class:`MockNeuroBackend`
that simulates an LLM — all without Ray or a network.  They cover:

* single-shot success
* policy-violating first attempt -> auto-correction on retry
* the re-prompt message carries the kernel's failure reason
* LLM parse errors are retried and corrected
* exhaustion after ``max_retries`` (no raise by default; raise on opt-in)
* the caller's message list is never mutated
"""

from __future__ import annotations

import pytest

from mesh.neuro import (
    MockNeuroBackend,
    NegotiatorProposal,
    NegotiatorQuote,
    NeuroSymbolicBridge,
    ProtocolViolation,
    ScoutProposal,
    SupplierDiscoveryItem,
    symbolic_validate,
)
from mesh.neuro.types import NeuralProposal, SymbolicVerdict

# ─── helpers ──────────────────────────────────────────────────────────────────


async def _real_kernel(proposal: NeuralProposal) -> SymbolicVerdict:
    """Validator backed by the production policy logic."""
    return symbolic_validate(proposal)


def _budget_quote(price: float, budget: float = 500_000.0) -> NegotiatorProposal:
    """A schema-valid quote whose economics may violate the policy kernel."""
    return NegotiatorProposal(
        correlation_id="REQ-001",
        supplier_id="S1",
        eval_trace_id="eval-1",
        pool_trace_id="pool-1",
        confidence=1.0,
        quote=NegotiatorQuote(
            supplier_id="S1",
            price=price,
            terms="net_30",
            metadata={
                "quantity": 1000,
                "lead_time_days": 30,
                "carbon_footprint_kg": 4000.0,
                "reliability_score": 0.9,
            },
        ),
    )


def _negotiator_payload_builder(m: NegotiatorProposal) -> dict:
    return m.to_kernel_payload(material="steel", quantity=1000, budget=500_000.0)


def _base_messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a procurement quote agent."},
        {"role": "user", "content": "Quote supplier S1 for steel, qty 1000, budget 500000."},
    ]


# ─── single-shot success ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_single_attempt_success():
    backend = MockNeuroBackend(_budget_quote(price=135.0))
    bridge = NeuroSymbolicBridge(backend=backend, validator=_real_kernel, max_retries=3)

    result = await bridge.safe_propose(
        archetype="negotiator",
        response_model=NegotiatorProposal,
        messages=_base_messages(),
        payload_builder=_negotiator_payload_builder,
        correlation_id="REQ-001",
    )

    assert result.verdict.approved
    assert result.attempts == 1
    assert result.exhausted is False
    assert result.model_instance is not None
    assert backend.call_count == 1


# ─── auto-correction: policy violation on attempt 1, fixed on attempt 2 ───


@pytest.mark.asyncio
async def test_auto_correction_after_policy_rejection():
    """The bridge must re-prompt and correct a schema-valid but policy-violating
    payload (600 * 1000 = 600000 exceeds the 500000 budget)."""
    bad = _budget_quote(price=600.0)
    good = _budget_quote(price=135.0)
    backend = MockNeuroBackend([bad, good])
    bridge = NeuroSymbolicBridge(backend=backend, validator=_real_kernel, max_retries=3)

    messages = _base_messages()
    result = await bridge.safe_propose(
        archetype="negotiator",
        response_model=NegotiatorProposal,
        messages=messages,
        payload_builder=_negotiator_payload_builder,
        correlation_id="REQ-001",
    )

    assert result.verdict.approved
    assert result.attempts == 2
    assert result.exhausted is False
    # Two LLM calls: original + corrected after re-prompt.
    assert backend.call_count == 2
    # History records the rejection then the acceptance.
    assert result.history[0]["approved"] is False
    assert "exceeds_budget" in result.history[0]["violations"][0]
    assert result.history[1]["approved"] is True
    # Caller's message list untouched.
    assert len(messages) == 2


@pytest.mark.asyncio
async def test_re_prompt_message_carries_failure_reason():
    """The auto-correction re-prompt must contain the kernel's violation reason."""
    bad = _budget_quote(price=600.0)
    good = _budget_quote(price=135.0)
    backend = MockNeuroBackend([bad, good])
    bridge = NeuroSymbolicBridge(backend=backend, validator=_real_kernel, max_retries=3)

    await bridge.safe_propose(
        archetype="negotiator",
        response_model=NegotiatorProposal,
        messages=_base_messages(),
        payload_builder=_negotiator_payload_builder,
        correlation_id="REQ-001",
    )

    # The second LLM call received a re-prompt containing the failure reason.
    second_call = backend._seq.calls[1]
    joined = "\n".join(m["content"] for m in second_call)
    assert "exceeds_budget" in joined
    assert "REJECTED" in joined


# ─── LLM parse error is retried ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_backend_parse_error_is_retried_and_corrects():
    """When the backend raises (unparseable JSON), the bridge re-prompts and
    recovers on the next attempt."""
    good = _budget_quote(price=135.0)
    backend = MockNeuroBackend([ValueError("unparseable JSON"), good])
    bridge = NeuroSymbolicBridge(backend=backend, validator=_real_kernel, max_retries=3)

    result = await bridge.safe_propose(
        archetype="negotiator",
        response_model=NegotiatorProposal,
        messages=_base_messages(),
        payload_builder=_negotiator_payload_builder,
        correlation_id="REQ-001",
    )

    assert result.verdict.approved
    assert result.attempts == 2
    assert result.history[0]["phase"] == "generation"
    assert result.history[1]["phase"] == "kernel_validation"
    assert result.history[1]["approved"] is True


# ─── exhaustion ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_exhaustion_returns_rejected_verdict_without_raising():
    """When every attempt is rejected and raise_on_exhaustion is False, the
    bridge returns the last verdict with exhausted=True instead of raising."""
    always_bad = _budget_quote(price=600.0)
    backend = MockNeuroBackend([always_bad, always_bad, always_bad])
    bridge = NeuroSymbolicBridge(backend=backend, validator=_real_kernel, max_retries=3)

    result = await bridge.safe_propose(
        archetype="negotiator",
        response_model=NegotiatorProposal,
        messages=_base_messages(),
        payload_builder=_negotiator_payload_builder,
        correlation_id="REQ-001",
    )

    assert result.verdict.approved is False
    assert result.exhausted is True
    assert result.attempts == 3
    assert backend.call_count == 3
    assert all(h["approved"] is False for h in result.history)


@pytest.mark.asyncio
async def test_raise_on_exhaustion_raises_protocol_violation():
    always_bad = _budget_quote(price=600.0)
    backend = MockNeuroBackend([always_bad, always_bad, always_bad])
    bridge = NeuroSymbolicBridge(
        backend=backend, validator=_real_kernel, max_retries=3, raise_on_exhaustion=True
    )

    with pytest.raises(ProtocolViolation) as exc_info:
        await bridge.safe_propose(
            archetype="negotiator",
            response_model=NegotiatorProposal,
            messages=_base_messages(),
            payload_builder=_negotiator_payload_builder,
            correlation_id="REQ-001",
        )
    assert "exceeds_budget" in str(exc_info.value)


# ─── caller messages are not mutated ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_caller_message_list_is_not_mutated():
    messages = _base_messages()
    original_len = len(messages)
    original_first = messages[0]["content"]

    backend = MockNeuroBackend([ValueError("err"), _budget_quote(price=135.0)])
    bridge = NeuroSymbolicBridge(backend=backend, validator=_real_kernel, max_retries=3)

    await bridge.safe_propose(
        archetype="negotiator",
        response_model=NegotiatorProposal,
        messages=messages,
        payload_builder=_negotiator_payload_builder,
        correlation_id="REQ-001",
    )

    assert len(messages) == original_len
    assert messages[0]["content"] == original_first


# ─── max_retries=1 means no correction loop ──────────────────────────────────


@pytest.mark.asyncio
async def test_max_retries_one_means_no_correction():
    backend = MockNeuroBackend(_budget_quote(price=600.0))
    bridge = NeuroSymbolicBridge(backend=backend, validator=_real_kernel, max_retries=1)

    result = await bridge.safe_propose(
        archetype="negotiator",
        response_model=NegotiatorProposal,
        messages=_base_messages(),
        payload_builder=_negotiator_payload_builder,
        correlation_id="REQ-001",
    )

    assert result.verdict.approved is False
    assert result.attempts == 1
    assert result.exhausted is True
    assert backend.call_count == 1


# ─── scout path with a custom (always-accept) validator ───────────────────


@pytest.mark.asyncio
async def test_scout_neuro_path_accepts_valid_output():
    supplier = SupplierDiscoveryItem(
        supplier_id="S1",
        material="steel",
        base_cost_per_unit=100.0,
        logistics_premium_per_unit=5.0,
        capacity_units=5000,
        current_utilization=0.3,
        min_margin_pct=0.2,
        reliability_score=0.9,
        esg_carbon_per_unit=400.0,
    )
    proposal = ScoutProposal(
        correlation_id="REQ-001",
        material="steel",
        quantity=1000,
        target_lead_time_days=30,
        spot_price=450.0,
        confidence=1.0,
        suppliers=[supplier],
    )
    backend = MockNeuroBackend([proposal])
    bridge = NeuroSymbolicBridge(backend=backend, validator=_real_kernel, max_retries=3)

    result = await bridge.safe_propose(
        archetype="scout",
        response_model=ScoutProposal,
        messages=_base_messages(),
        payload_builder=lambda m: m.to_kernel_payload(),
        correlation_id="REQ-001",
        confidence=1.0,
    )

    assert result.verdict.approved
    assert result.attempts == 1
    assert result.model_instance.to_pool_dict()["material"] == "steel"


# ─── a custom validator that rejects then approves ──────────────────────────


@pytest.mark.asyncio
async def test_custom_validator_drives_retry_to_acception():
    calls = {"n": 0}

    async def _flaky_validator(proposal: NeuralProposal) -> SymbolicVerdict:
        calls["n"] += 1
        if calls["n"] < 2:
            return SymbolicVerdict.rejected(
                reason="VALIDATION_FAILED: confidence_below_threshold",
                violations=["confidence_below_threshold: 0.1"],
            )
        return SymbolicVerdict.approved_verdict(clamped_payload=dict(proposal.payload))

    backend = MockNeuroBackend(
        [
            _budget_quote(price=135.0),
            _budget_quote(price=135.0),
        ]
    )
    bridge = NeuroSymbolicBridge(backend=backend, validator=_flaky_validator, max_retries=3)

    result = await bridge.safe_propose(
        archetype="negotiator",
        response_model=NegotiatorProposal,
        messages=_base_messages(),
        payload_builder=_negotiator_payload_builder,
        correlation_id="REQ-001",
    )

    assert result.verdict.approved
    assert result.attempts == 2
    assert calls["n"] == 2
