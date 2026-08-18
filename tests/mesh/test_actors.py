"""Integration tests for mesh actors via DistributedBlackboard."""

import pytest


@pytest.mark.asyncio
async def test_scout_writes_discovery_channel():
    """ScoutActor reads REQUIREMENT and writes DISCOVERY channel."""
    import ray

    from mesh.actors import SafetyKernelActor, ScoutActor
    from mesh.blackboard import create_blackboard
    from mesh.channels import ChannelType

    ray.init(ignore_reinit_error=True, num_cpus=4, include_dashboard=False)

    try:
        # Setup
        bb = await create_blackboard("test_bb")
        kernel = SafetyKernelActor.options(name="safety_kernel").remote()
        # ray.util.register_actor("safety_kernel", kernel)

        # Write a requirement
        await bb.write.remote(
            "kernel",
            ChannelType.REQUIREMENT,
            {
                "requirement": {
                    "constraints": {
                        "material": "steel",
                        "quantity": 1000,
                        "target_lead_time_days": 30,
                    }
                }
            },
        )

        # Create and run scout
        scout = ScoutActor.options(name="scout_test").remote("scout_test", bb, kernel)
        result = await scout.step.remote()

        assert result["status"] == "success"

        # Verify DISCOVERY channel has supplier list
        discoveries = await bb.read.remote("evaluator", ChannelType.DISCOVERY, limit=20)
        assert len(discoveries) > 0

        # Should have supplier_list and supplier_discovered entries
        types = {d.get("payload", {}).get("type") for d in discoveries}
        assert "supplier_list" in types
        assert "supplier_discovered" in types

        # Should have 5 suppliers (from templates)
        supplier_discovered = [
            d for d in discoveries if d.get("payload", {}).get("type") == "supplier_discovered"
        ]
        assert len(supplier_discovered) == 5

        # Cleanup
        ray.kill(scout)
        ray.kill(kernel)

    finally:
        ray.shutdown()


@pytest.mark.asyncio
async def test_evaluator_reads_discovery_writes_score_risk():
    """EvaluatorActor reads DISCOVERY, writes SCORE and RISK channels."""
    import ray

    from mesh.actors import EvaluatorActor, SafetyKernelActor, ScoutActor
    from mesh.blackboard import create_blackboard
    from mesh.channels import ChannelType
    from swarm.memory import SupplierMemoryStore

    ray.init(ignore_reinit_error=True, num_cpus=4, include_dashboard=False)

    try:
        # Setup
        bb = await create_blackboard("test_bb2")
        kernel = SafetyKernelActor.options(name="safety_kernel2").remote()
        # ray.util.register_actor("safety_kernel2", kernel)

        # Write requirement first
        await bb.write.remote(
            "kernel",
            ChannelType.REQUIREMENT,
            {
                "requirement": {
                    "constraints": {
                        "material": "steel",
                        "quantity": 1000,
                        "target_lead_time_days": 30,
                    }
                }
            },
        )

        # First run scout to populate DISCOVERY
        scout = ScoutActor.options(name="scout_for_eval").remote("scout_for_eval", bb, kernel)
        await scout.step.remote()

        # Create and run evaluator
        memory = SupplierMemoryStore()
        evaluator = EvaluatorActor.options(name="eval_test").remote("eval_test", bb, kernel, memory)
        result = await evaluator.step.remote()

        assert result["status"] == "success"

        # Verify SCORE channel has evaluations
        scores = await bb.read.remote("negotiator", ChannelType.SCORE, limit=20)
        assert len(scores) > 0
        eval_entries = [s for s in scores if s.get("payload", {}).get("type") == "evaluation"]
        assert len(eval_entries) == 5

        # Verify RISK channel has risk assessments
        risks = await bb.read.remote("buyer", ChannelType.RISK, limit=20)
        assert len(risks) > 0
        risk_entries = [r for r in risks if r.get("payload", {}).get("type") == "risk_assessment"]
        assert len(risk_entries) == 5

        # Each evaluation should have score, breakdown, strategy
        for eval_trace in eval_entries:
            data = eval_trace.get("payload", {}).get("data", {})
            assert "supplier_id" in data
            assert "score" in data
            assert "breakdown" in data
            assert "strategy" in data

        # Each risk should have risk_level, overall_risk_score
        for risk_trace in risk_entries:
            data = risk_trace.get("payload", {}).get("data", {})
            assert "supplier_id" in data
            assert "risk_level" in data
            assert "risk_scores" in data
            assert "overall_risk_score" in data["risk_scores"]

        # Cleanup
        ray.kill(scout)
        ray.kill(evaluator)
        ray.kill(kernel)

    finally:
        ray.shutdown()


@pytest.mark.asyncio
async def test_negotiator_reads_score_writes_deal():
    """NegotiatorActor reads SCORE, writes DEAL channel."""
    import ray

    from mesh.actors import EvaluatorActor, NegotiatorActor, SafetyKernelActor, ScoutActor
    from mesh.blackboard import create_blackboard
    from mesh.channels import ChannelType
    from swarm.memory import SupplierMemoryStore

    ray.init(ignore_reinit_error=True, num_cpus=4, include_dashboard=False)

    try:
        # Setup
        bb = await create_blackboard("test_bb3")
        kernel = SafetyKernelActor.options(name="safety_kernel3").remote()
        # ray.util.register_actor("safety_kernel3", kernel)

        # Write requirement first
        await bb.write.remote(
            "kernel",
            ChannelType.REQUIREMENT,
            {
                "requirement": {
                    "constraints": {
                        "material": "steel",
                        "quantity": 1000,
                        "target_lead_time_days": 30,
                    }
                }
            },
        )

        # Run pipeline up to evaluation
        scout = ScoutActor.options(name="scout_for_neg").remote("scout_for_neg", bb, kernel)
        await scout.step.remote()

        memory = SupplierMemoryStore()
        evaluator = EvaluatorActor.options(name="eval_for_neg").remote(
            "eval_for_neg", bb, kernel, memory
        )
        await evaluator.step.remote()

        # Create and run negotiator
        negotiator = NegotiatorActor.options(name="neg_test").remote("neg_test", bb, kernel)
        result = await negotiator.step.remote()

        assert result["status"] == "success"

        # Verify DEAL channel has quotes
        deals = await bb.read.remote("buyer", ChannelType.DEAL, limit=20)
        assert len(deals) > 0
        quote_entries = [d for d in deals if d.get("payload", {}).get("type") == "quote"]
        assert len(quote_entries) == 5

        # Each quote should have supplier_id, price, terms, metadata
        for quote_trace in quote_entries:
            data = quote_trace.get("payload", {}).get("data", {})
            assert "supplier_id" in data
            assert "price" in data
            assert "terms" in data
            assert "metadata" in data
            assert data["terms"] == "net_30"

        # Cleanup
        ray.kill(scout)
        ray.kill(evaluator)
        ray.kill(negotiator)
        ray.kill(kernel)

    finally:
        ray.shutdown()


@pytest.mark.asyncio
async def test_buyer_reads_deal_risk_score_writes_decision():
    """BuyerActor (singleton) reads DEAL/RISK/SCORE, writes DECISION channel."""
    import ray

    from mesh.actors import (
        BuyerActor,
        EvaluatorActor,
        NegotiatorActor,
        SafetyKernelActor,
        ScoutActor,
    )
    from mesh.blackboard import create_blackboard
    from mesh.channels import ChannelType
    from swarm.domain.governance import STANDARD_POLICY
    from swarm.memory import SupplierMemoryStore

    ray.init(ignore_reinit_error=True, num_cpus=4, include_dashboard=False)

    try:
        # Setup
        bb = await create_blackboard("test_bb4")
        kernel = SafetyKernelActor.options(name="safety_kernel4").remote()
        # ray.util.register_actor("safety_kernel4", kernel)

        # Write requirement first with correlation_id
        correlation_id = "test-buyer-correlation-123"
        await bb.write.remote(
            "kernel",
            ChannelType.REQUIREMENT,
            {
                "requirement": {
                    "constraints": {
                        "material": "steel",
                        "quantity": 1000,
                        "target_lead_time_days": 30,
                    }
                },
                "correlation_id": correlation_id,
            },
        )

        # Run full pipeline
        scout = ScoutActor.options(name="scout_for_buyer").remote("scout_for_buyer", bb, kernel)
        await scout.step.remote()

        memory = SupplierMemoryStore()
        evaluator = EvaluatorActor.options(name="eval_for_buyer").remote(
            "eval_for_buyer", bb, kernel, memory
        )
        await evaluator.step.remote()

        negotiator = NegotiatorActor.options(name="neg_for_buyer").remote(
            "neg_for_buyer", bb, kernel
        )
        await negotiator.step.remote()

        # Create and run buyer
        buyer = BuyerActor.options(name="buyer_0").remote("buyer_0", bb, kernel, STANDARD_POLICY)
        result = await buyer.step.remote()

        assert result["status"] == "success"

        # Verify DECISION channel has decision
        decisions = await bb.read.remote("kernel", ChannelType.DECISION, limit=5)
        assert len(decisions) > 0
        decision_entries = [d for d in decisions if d.get("payload", {}).get("type") == "decision"]
        assert len(decision_entries) >= 1

        decision_data = decision_entries[0].get("payload", {}).get("data", {})
        assert "selected_supplier" in decision_data
        assert "composite_score" in decision_data
        assert "ranked" in decision_data
        assert "method" in decision_data
        assert decision_data["method"] == "deterministic_mcda"

        # Selected supplier should be one of the 5
        selected = decision_data["selected_supplier"]
        assert selected is not None
        assert selected in [
            "MinerCorp_A",
            "DistribCorp_B",
            "RecycleCorp_C",
            "TraderCorp_D",
            "PremiumSteel_E",
        ]

        # Cleanup
        ray.kill(scout)
        ray.kill(evaluator)
        ray.kill(negotiator)
        ray.kill(buyer)
        ray.kill(kernel)

    finally:
        ray.shutdown()


@pytest.mark.asyncio
async def test_full_pipeline_via_cluster():
    """End-to-end test using ProcurementCluster."""
    import ray

    from mesh.cluster import ClusterContext, MeshConfig

    ray.init(ignore_reinit_error=True, num_cpus=4, include_dashboard=False)

    try:
        config = MeshConfig(
            n_scouts=2,
            n_evaluators=2,
            n_negotiators=2,
        )

        async with ClusterContext(config) as cluster:
            requirement = {
                "correlation_id": "test-correlation-123",
                "constraints": {
                    "material": "steel",
                    "quantity": 1000,
                    "target_lead_time_days": 30,
                    "budget": 500_000.0,
                },
            }

            result = await cluster.run_procurement(requirement)

            assert result["status"] == "completed"
            assert result["decision"] is not None
            decision_data = result["decision"].get("payload", {}).get("data", {})
            assert decision_data["selected_supplier"] is not None

    finally:
        ray.shutdown()


@pytest.mark.asyncio
async def test_acl_enforcement():
    """Test that ACLs are enforced on blackboard read/write."""
    import ray

    from mesh.actors import SafetyKernelActor
    from mesh.blackboard import create_blackboard
    from mesh.channels import ChannelType

    ray.init(ignore_reinit_error=True, num_cpus=2, include_dashboard=False)

    try:
        bb = await create_blackboard("test_bb_acl")
        kernel = SafetyKernelActor.options(name="safety_kernel_acl").remote()
        # The kernel actor is created but we don't need to look it up by name for this test
        # since we're testing the blackboard ACLs directly

        # Scout CAN write to DISCOVERY
        trace_id = await bb.write.remote("scout", ChannelType.DISCOVERY, {"test": "data"})
        assert trace_id is not None

        # Scout CANNOT write to SCORE (should raise PermissionError)
        with pytest.raises(PermissionError):
            await bb.write.remote("scout", ChannelType.SCORE, {"test": "data"})

        # Scout CANNOT read from SCORE (only evaluator, negotiator, buyer, kernel)
        with pytest.raises(PermissionError):
            await bb.read.remote("scout", ChannelType.SCORE)

        # Evaluator CAN read DISCOVERY
        discoveries = await bb.read.remote("evaluator", ChannelType.DISCOVERY)
        assert isinstance(discoveries, list)

        # Cleanup
        ray.kill(kernel)

    finally:
        ray.shutdown()


@pytest.mark.asyncio
async def test_kernel_validation_rejects_invalid_proposal():
    """Test that SafetyKernelActor rejects invalid proposals."""
    import ray

    from mesh.actors import NeuralProposal, SafetyKernelActor

    ray.init(ignore_reinit_error=True, num_cpus=2, include_dashboard=False)

    try:
        kernel = SafetyKernelActor.options(name="safety_kernel_val").remote()

        # Valid proposal
        valid = NeuralProposal(
            proposal_id="test1",
            archetype="negotiator",
            payload={
                "price": 100,
                "lead_time_days": 30,
                "payment_terms": "net_30",
                "material": "steel",
            },
            confidence=1.0,
        )
        verdict = await kernel.validate.remote(valid)
        assert verdict.approved is True

        # Invalid price (too high)
        invalid_price = NeuralProposal(
            proposal_id="test2",
            archetype="negotiator",
            payload={
                "price": 20_000_000,
                "lead_time_days": 30,
                "payment_terms": "net_30",
                "material": "steel",
            },
            confidence=1.0,
        )
        verdict = await kernel.validate.remote(invalid_price)
        assert verdict.approved is False
        assert "price_out_of_bounds" in verdict.reason

        # Invalid payment terms
        invalid_terms = NeuralProposal(
            proposal_id="test3",
            archetype="negotiator",
            payload={
                "price": 100,
                "lead_time_days": 30,
                "payment_terms": "invalid_terms",
                "material": "steel",
            },
            confidence=1.0,
        )
        verdict = await kernel.validate.remote(invalid_terms)
        assert verdict.approved is False
        assert "invalid_payment_terms" in verdict.reason

        # Invalid material
        invalid_material = NeuralProposal(
            proposal_id="test4",
            archetype="negotiator",
            payload={
                "price": 100,
                "lead_time_days": 30,
                "payment_terms": "net_30",
                "material": "unobtanium",
            },
            confidence=1.0,
        )
        verdict = await kernel.validate.remote(invalid_material)
        assert verdict.approved is False
        assert "invalid_material" in verdict.reason

        # Low confidence
        low_conf = NeuralProposal(
            proposal_id="test5",
            archetype="negotiator",
            payload={
                "price": 100,
                "lead_time_days": 30,
                "payment_terms": "net_30",
                "material": "steel",
            },
            confidence=0.3,
        )
        verdict = await kernel.validate.remote(low_conf)
        assert verdict.approved is False
        assert "confidence_below_threshold" in verdict.reason

        ray.kill(kernel)

    finally:
        ray.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
