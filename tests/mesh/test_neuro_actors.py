"""Ray-gated integration tests for the neuro-bridge wiring in ScoutActor /
NegotiatorActor.

These exercise the actors end-to-end against a real Blackboard + SafetyKernelActor
with a :class:`MockNeuroBackend` injected, so no LLM or network is needed.  They
are skipped when Ray is not installed (the core retry-loop logic itself is
covered without Ray in :mod:`tests.mesh.test_neuro_bridge`).
"""

from __future__ import annotations

import pytest

from mesh.channels import ChannelType

ray = pytest.importorskip("ray")


def _valid_supplier_dict(supplier_id: str = "LLM_S1") -> dict:
    return {
        "supplier_id": supplier_id,
        "material": "steel",
        "base_cost_per_unit": 100.0,
        "logistics_premium_per_unit": 5.0,
        "capacity_units": 5000,
        "current_utilization": 0.3,
        "min_margin_pct": 0.2,
        "reliability_score": 0.9,
        "esg_carbon_per_unit": 400.0,
    }


def _write_requirement(bb, correlation_id: str) -> None:
    ray.get(
        bb.write.remote(
            "kernel",
            ChannelType.REQUIREMENT,
            {
                "requirement": {
                    "constraints": {
                        "material": "steel",
                        "quantity": 1000,
                        "target_lead_time_days": 30,
                        "budget": 500_000.0,
                    }
                },
                "correlation_id": correlation_id,
            },
        )
    )


def _async_validator(kernel):
    """Build an async validator wrapping the SafetyKernelActor."""

    async def _validate(proposal):
        return await kernel.validate.remote(proposal)

    return _validate


@pytest.mark.asyncio
async def test_scout_uses_neuro_bridge_and_writes_llm_pool():
    """ScoutActor with an injected neuro bridge writes the LLM-generated pool."""
    from mesh.actors import SafetyKernelActor, ScoutActor
    from mesh.blackboard import create_blackboard
    from mesh.neuro import MockNeuroBackend, NeuroSymbolicBridge, ScoutProposal

    ray.init(ignore_reinit_error=True, num_cpus=4, include_dashboard=False)
    try:
        bb = await create_blackboard("v2_bb_scout_neuro")
        kernel = SafetyKernelActor.options(name="v2_kernel_scout_neuro").remote()
        _write_requirement(bb, correlation_id="neuro-scout-1")

        proposal = ScoutProposal(
            correlation_id="neuro-scout-1",
            material="steel",
            quantity=1000,
            target_lead_time_days=30,
            spot_price=450.0,
            confidence=1.0,
            suppliers=[_valid_supplier_dict("LLM_Scout_S1")],
        )
        backend = MockNeuroBackend([proposal])
        bridge = NeuroSymbolicBridge(
            backend=backend, validator=_async_validator(kernel), max_retries=3
        )

        scout = ScoutActor.options(name="v2_scout_neuro").remote(
            "v2_scout_neuro", bb, kernel, neuro_bridge=bridge
        )
        result = await scout.step.remote()
        assert result["status"] == "success"

        discoveries = await bb.read.remote("evaluator", ChannelType.DISCOVERY, limit=20)
        lists = [d for d in discoveries if d.get("payload", {}).get("type") == "supplier_list"]
        assert len(lists) == 1
        pool_suppliers = lists[0]["payload"]["data"]["suppliers"]
        # The LLM-supplied supplier id is present (neuro path was used).
        assert any(s["supplier_id"] == "LLM_Scout_S1" for s in pool_suppliers)

        ray.kill(scout, no_restart=True)
    finally:
        ray.kill(kernel, no_restart=True)
        ray.shutdown()


@pytest.mark.asyncio
async def test_scout_falls_back_when_neuro_exhausted():
    """When the LLM output is kernel-rejected every attempt, the scout falls
    back to the deterministic pool."""
    from mesh.actors import SafetyKernelActor, ScoutActor
    from mesh.blackboard import create_blackboard
    from mesh.neuro import MockNeuroBackend, NeuroSymbolicBridge, ScoutProposal

    ray.init(ignore_reinit_error=True, num_cpus=4, include_dashboard=False)
    try:
        bb = await create_blackboard("v2_bb_scout_fb")
        kernel = SafetyKernelActor.options(name="v2_kernel_scout_fb").remote()
        _write_requirement(bb, correlation_id="fb-scout-1")

        # spot_price=20_000_000 -> kernel rejects price_out_of_bounds every attempt.
        bad_proposal = ScoutProposal(
            correlation_id="fb-scout-1",
            material="steel",
            quantity=1000,
            target_lead_time_days=30,
            spot_price=20_000_000.0,
            confidence=1.0,
            suppliers=[_valid_supplier_dict("BAD_SUPPLIER")],
        )
        backend = MockNeuroBackend([bad_proposal])  # repeated for all retries
        bridge = NeuroSymbolicBridge(
            backend=backend, validator=_async_validator(kernel), max_retries=3
        )

        scout = ScoutActor.options(name="v2_scout_fb").remote(
            "v2_scout_fb", bb, kernel, neuro_bridge=bridge
        )
        await scout.step.remote()

        discoveries = await bb.read.remote("evaluator", ChannelType.DISCOVERY, limit=30)
        lists = [d for d in discoveries if d.get("payload", {}).get("type") == "supplier_list"]
        # Deterministic fallback pool must be present.
        pool = lists[-1]["payload"]["data"]["suppliers"]
        assert {s["supplier_id"] for s in pool} == {
            "MinerCorp_A",
            "DistribCorp_B",
            "RecycleCorp_C",
            "TraderCorp_D",
            "PremiumSteel_E",
        }

        ray.kill(scout, no_restart=True)
    finally:
        ray.kill(kernel, no_restart=True)
        ray.shutdown()


@pytest.mark.asyncio
async def test_negotiator_uses_neuro_quote_with_correction():
    """NegotiatorActor auto-corrects an over-budget quote and writes the fixed one."""
    from mesh.actors import EvaluatorActor, NegotiatorActor, SafetyKernelActor, ScoutActor
    from mesh.blackboard import create_blackboard
    from mesh.neuro import (
        MockNeuroBackend,
        NegotiatorProposal,
        NegotiatorQuote,
        NeuroSymbolicBridge,
        QuoteMetadata,
    )
    from swarm.memory import SupplierMemoryStore

    ray.init(ignore_reinit_error=True, num_cpus=4, include_dashboard=False)
    try:
        bb = await create_blackboard("v2_bb_neg_neuro")
        kernel = SafetyKernelActor.options(name="v2_kernel_neg_neuro").remote()
        _write_requirement(bb, correlation_id="neuro-neg-1")

        # Run scout + evaluator to seed DISCOVERY / SCORE.
        await (
            ScoutActor.options(name="v2_scout_neg").remote("v2_scout_neg", bb, kernel).step.remote()
        )
        memory = SupplierMemoryStore()
        await (
            EvaluatorActor.options(name="v2_eval_neg")
            .remote("v2_eval_neg", bb, kernel, memory)
            .step.remote()
        )

        def _quote(price: float) -> NegotiatorProposal:
            return NegotiatorProposal(
                correlation_id="neuro-neg-1",
                supplier_id="MinerCorp_A",
                quote=NegotiatorQuote(
                    supplier_id="MinerCorp_A",
                    price=price,
                    terms="net_30",
                    metadata=QuoteMetadata(
                        quantity=1000,
                        lead_time_days=30,
                        carbon_footprint_kg=4000.0,
                        reliability_score=0.85,
                    ),
                ),
                confidence=1.0,
            )

        backend = MockNeuroBackend([_quote(600.0), _quote(135.0)])  # over-budget -> fixed
        bridge = NeuroSymbolicBridge(
            backend=backend, validator=_async_validator(kernel), max_retries=3
        )

        negotiator = NegotiatorActor.options(name="v2_neg_neuro").remote(
            "v2_neg_neuro", bb, kernel, neuro_bridge=bridge
        )
        result = await negotiator.step.remote()
        assert result["status"] == "success"

        deals = await bb.read.remote("buyer", ChannelType.DEAL, limit=20)
        quote_entries = [d for d in deals if d.get("payload", {}).get("type") == "quote"]
        assert len(quote_entries) == 5
        # The written quote must be the corrected price (the second backend
        # response).  Since the backend lives inside the Ray actor process we
        # cannot inspect ``backend.call_count`` directly; instead we verify the
        # retry happened by confirming the corrected (in-budget) quote was
        # actually written to the DEAL channel.
        written = {
            d["payload"]["data"]["supplier_id"]: d["payload"]["data"]["price"]
            for d in quote_entries
        }
        assert written["MinerCorp_A"] == 135.0

        ray.kill(negotiator, no_restart=True)
    finally:
        ray.kill(kernel, no_restart=True)
        ray.shutdown()
