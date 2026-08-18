"""ScoutActor — discovers suppliers and writes to DISCOVERY channel.

Wraps the deterministic SupplierDiscoveryAgent logic as a Ray actor.
"""

from __future__ import annotations

from typing import Any

import ray
import structlog

from core.market_simulator import MarketSimulator
from mesh.actors.base import MeshActor
from mesh.blackboard import DistributedBlackboard
from mesh.channels import ChannelType
from mesh.neuro import (
    LLMConfig,
    NeuroSymbolicBridge,
    OpenAICompatibleBackend,
    ScoutProposal,
)
from mesh.neuro.backend import StructuredBackend
from mesh.neuro.types import NeuralProposal, SymbolicVerdict

logger = structlog.get_logger(__name__)

SUPPLIER_TEMPLATES: list[dict[str, Any]] = [
    {
        "name": "MinerCorp_A",
        "base_mult": 0.35,
        "logistics": 50,
        "cap": 5000,
        "util": 0.3,
        "margin": 0.20,
        "rel": 0.85,
        "carbon": 1800.0,
    },
    {
        "name": "DistribCorp_B",
        "base_mult": 0.75,
        "logistics": 20,
        "cap": 10000,
        "util": 0.6,
        "margin": 0.12,
        "rel": 0.90,
        "carbon": 1200.0,
    },
    {
        "name": "RecycleCorp_C",
        "base_mult": 0.80,
        "logistics": 30,
        "cap": 3000,
        "util": 0.4,
        "margin": 0.15,
        "rel": 0.75,
        "carbon": 400.0,
    },
    {
        "name": "TraderCorp_D",
        "base_mult": 0.90,
        "logistics": 10,
        "cap": 8000,
        "util": 0.8,
        "margin": 0.08,
        "rel": 0.70,
        "carbon": 1500.0,
    },
    {
        "name": "PremiumSteel_E",
        "base_mult": 0.50,
        "logistics": 80,
        "cap": 2000,
        "util": 0.2,
        "margin": 0.30,
        "rel": 0.95,
        "carbon": 2000.0,
    },
]


@ray.remote(max_restarts=3, max_task_retries=3)
class ScoutActor(MeshActor):
    """Discovers candidate suppliers for a requirement.

    Reads from REQUIREMENT channel, builds supplier pool, writes to DISCOVERY.

    When a ``neuro_bridge`` (or ``llm_config``) is supplied the actor uses
    schema-constrained LLM generation validated by the SafetyKernelActor with an
    auto-correction retry loop.  If the LLM path is exhausted or unavailable it
    transparently falls back to the deterministic :meth:`_build_pool` so the
    procurement pipeline never blocks on a cognitive failure.
    """

    def __init__(
        self,
        actor_id: str,
        blackboard: ray.actor.ActorHandle,
        kernel: ray.actor.ActorHandle | None = None,
        neuro_bridge: NeuroSymbolicBridge | None = None,
        llm_config: LLMConfig | None = None,
        neuro_max_retries: int = 3,
    ) -> None:
        super().__init__(actor_id, "scout", blackboard, kernel)
        self._processed_requirements: set[str] = set()
        self._neuro_bridge = neuro_bridge or self._build_neuro_bridge(llm_config, neuro_max_retries)

    def _build_neuro_bridge(
        self,
        llm_config: LLMConfig | None,
        max_retries: int,
    ) -> NeuroSymbolicBridge | None:
        """Construct the neuro bridge from an LLM config, if provided.

        The kernel validator wraps the SafetyKernelActor so the bridge's retry
        loop and the base ``step()`` gate consult the same authority.
        """
        if llm_config is None:
            return None

        async def _kernel_validate(proposal: NeuralProposal) -> SymbolicVerdict:
            return await self.kernel.validate.remote(proposal)  # type: ignore[no-any-return]

        backend: StructuredBackend = OpenAICompatibleBackend(llm_config)
        return NeuroSymbolicBridge(
            backend=backend,
            validator=_kernel_validate,
            max_retries=max_retries,
            raise_on_exhaustion=False,
        )

    async def perceive(self, blackboard: DistributedBlackboard) -> dict[str, Any]:
        """Read REQUIREMENT channel for new requirements."""
        traces = await self.read_channel(ChannelType.REQUIREMENT, limit=10)
        return {"requirements": traces}

    async def reason(self, perception: dict[str, Any]) -> dict[str, Any]:
        """Process new requirements and build supplier pools."""
        requirements = perception.get("requirements", [])
        proposals = []

        for trace in requirements:
            payload = trace.get("payload", {})
            correlation_id = trace.get("correlation_id", "")
            trace_id = trace.get("id", "")

            if correlation_id in self._processed_requirements:
                continue

            # Extract requirement constraints
            requirement_data = payload.get("requirement", {})
            constraints = requirement_data.get("constraints", {})
            material = str(constraints.get("material") or "steel")
            quantity = int(constraints.get("quantity") or 1000)
            target_lead_time_days = int(constraints.get("target_lead_time_days") or 30)

            if material not in MarketSimulator.MATERIALS:
                material = "steel"

            # Build supplier pool (deterministic)
            market = MarketSimulator(seed=42)
            spot_price = market.get_current_state(material).spot_price

            # Neuro path: structured LLM generation with symbolic retry loop.
            pool_data = await self._neuro_pool(
                material, quantity, target_lead_time_days, spot_price, correlation_id
            )
            if pool_data is None:  # LLM path exhausted / unavailable -> deterministic
                pool_data = {
                    "material": material,
                    "quantity": quantity,
                    "target_lead_time_days": target_lead_time_days,
                    "spot_price": spot_price,
                    "suppliers": self._build_pool(material, spot_price),
                }

            suppliers = pool_data["suppliers"]

            proposals.append(
                {
                    "correlation_id": correlation_id,
                    "requirement_trace_id": trace_id,
                    "pool": pool_data,
                    "supplier_ids": [s["supplier_id"] for s in suppliers],
                    "confidence": 1.0,  # Deterministic
                }
            )
            self._processed_requirements.add(correlation_id)

        return {"proposals": proposals}

    async def act(self, blackboard: DistributedBlackboard, proposal: dict[str, Any]) -> None:
        """Write supplier pool to DISCOVERY channel."""
        for prop in proposal.get("proposals", []):
            correlation_id = prop["correlation_id"]
            pool = prop["pool"]
            supplier_ids = prop["supplier_ids"]

            # Write the supplier list artifact to DISCOVERY channel
            trace_id = await self.write_channel(
                ChannelType.DISCOVERY,
                {
                    "type": "supplier_list",
                    "data": pool,
                    "correlation_id": correlation_id,
                },
                parent_ids=[prop["requirement_trace_id"]],
            )

            # Write individual supplier discovery traces
            for supplier_id in supplier_ids:
                await self.write_channel(
                    ChannelType.DISCOVERY,
                    {
                        "type": "supplier_discovered",
                        "supplier_id": supplier_id,
                        "material": pool["material"],
                        "pool_trace_id": trace_id,
                        "correlation_id": correlation_id,
                    },
                    parent_ids=[trace_id],
                )

            logger.info(
                "scout_discovered",
                actor_id=self.actor_id,
                correlation_id=correlation_id,
                supplier_count=len(supplier_ids),
                trace_id=trace_id,
            )

    async def _neuro_pool(
        self,
        material: str,
        quantity: int,
        target_lead_time_days: int,
        spot_price: float,
        correlation_id: str,
    ) -> dict[str, Any] | None:
        """Build the supplier pool via structured LLM generation + kernel retry.

        Returns the kernel-approved pool dict, or ``None`` if the neuro path is
        unavailable or exhausted (the caller then uses the deterministic pool).
        """
        bridge = self._neuro_bridge
        if bridge is None:
            return None
        assert bridge is not None, "Neuro bridge should be available here"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a procurement supplier-discovery agent. Generate a "
                    "structured JSON list of candidate suppliers for the requested "
                    "material. Each supplier must have realistic, non-negative "
                    "economics and a reliability score in [0, 1]. Return ONLY valid "
                    "JSON conforming to the ScoutProposal schema."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Discover suppliers for: material={material}, quantity={quantity}, "
                    f"target_lead_time_days={target_lead_time_days}, spot_price={spot_price}. "
                    "Provide 5 suppliers with diverse cost, logistics, reliability and "
                    "carbon profiles."
                ),
            },
        ]

        try:
            result = await bridge.safe_propose(
                archetype="scout",
                response_model=ScoutProposal,
                messages=messages,
                payload_builder=lambda m: m.to_kernel_payload(),
                correlation_id=correlation_id,
                confidence=1.0,
            )
        except Exception as exc:
            logger.warning(
                "scout_neuro_generation_error",
                actor_id=self.actor_id,
                error=str(exc),
            )
            return None

        if not result.verdict.approved:
            logger.warning(
                "scout_neuro_exhausted",
                actor_id=self.actor_id,
                correlation_id=correlation_id,
                reason=result.verdict.reason,
                attempts=result.attempts,
            )
            return None

        model: ScoutProposal = result.model_instance  # type: ignore[assignment]
        return model.to_pool_dict()

    @staticmethod
    def _build_pool(material: str, spot_price: float) -> list[dict[str, Any]]:
        """Build deterministic supplier profiles from templates."""
        pool: list[dict[str, Any]] = []
        for template in SUPPLIER_TEMPLATES:
            pool.append(
                {
                    "supplier_id": template["name"],
                    "material": material,
                    "base_cost_per_unit": round(spot_price * template["base_mult"], 2),
                    "logistics_premium_per_unit": template["logistics"],
                    "capacity_units": template["cap"],
                    "current_utilization": template["util"],
                    "min_margin_pct": template["margin"],
                    "reliability_score": template["rel"],
                    "esg_carbon_per_unit": template["carbon"],
                }
            )
        return pool
