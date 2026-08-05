"""SupplierDiscoveryAgent — discovers candidate suppliers for a requirement.

Reuses the existing market model (``core.market_simulator.MarketSimulator``,
seeded for determinism) to derive a spot reference price, then builds a
heterogeneous supplier pool from the same CostModel templates the CNP auction
uses (``api/main._create_suppliers``). It keeps the pool in a single
:class:`SupplierListArtifact` and announces one ``SupplierDiscovered`` event
per supplier, so evaluation can start on the first discovery instead of waiting
for the whole pool. It also declares the per-request completion expectations
for evaluation and quote artifacts (both sized to the pool), which the
:class:`CompletionTracker` uses to fire the phase-gate events.

Phase 4: discovery triggers on ``StrategySelected`` (not ``RequirementCreated``)
so the pool and its completion expectations are only created after the
:class:`StrategyArtifact` exists — otherwise the concurrent event bus could let
evaluation run before the strategy that weights it. ``RequirementCreated`` is
kept as a fallback for direct unit-test drives, where no strategy flow runs.
"""

from typing import Any

import structlog

from core.market_simulator import MarketSimulator
from swarm.core.agent import BaseAgent
from swarm.core.capability import Capability
from swarm.core.event import Event
from swarm.core.state import SwarmState
from swarm.domain.artifacts import (
    REQUIREMENT_ARTIFACT_NAME,
    STRATEGY_ARTIFACT_NAME,
    SupplierListArtifact,
)
from swarm.domain.events import ProcurementEventType

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


class SupplierDiscoveryAgent(BaseAgent):
    """Discovers candidate suppliers for a published requirement."""

    name = "supplier_discovery_agent"
    description = "Discovers candidate suppliers for a published requirement"
    capabilities = [
        Capability(
            name="supplier.search",
            description="Searches the supplier pool for a requirement",
        )
    ]

    def __init__(self) -> None:
        super().__init__()
        self._correlation_id: str | None = None
        self._requirement_artifact: str = REQUIREMENT_ARTIFACT_NAME
        self._strategy_artifact: str | None = None
        self._pool: dict[str, Any] | None = None
        self._pending = False
        self._discovered_for: set[str] = set()

    async def perceive(self, event: Event) -> None:
        if event.replayed:
            return
        if event.type == ProcurementEventType.STRATEGY_SELECTED:
            self._pending = True
            self._correlation_id = event.correlation_id
            self._strategy_artifact = str(event.payload.get("artifact", STRATEGY_ARTIFACT_NAME))
        elif event.type == ProcurementEventType.REQUIREMENT_CREATED:
            if event.correlation_id in self._discovered_for:
                return
            self._pending = True
            self._correlation_id = event.correlation_id
            self._strategy_artifact = None
            self._requirement_artifact = str(
                event.payload.get("artifact", REQUIREMENT_ARTIFACT_NAME)
            )

    async def reason(self, state: SwarmState) -> None:
        if not self._pending:
            return
        if self._correlation_id is None or self._correlation_id in self._discovered_for:
            self._pending = False
            return
        requirement = state.get_artifact(self._requirement_artifact)
        if requirement is None and self._strategy_artifact is not None:
            strategy = state.get_artifact(self._strategy_artifact)
            if strategy is not None and strategy.parent_ids:
                self._requirement_artifact = strategy.parent_ids[0]
                requirement = state.get_artifact(self._requirement_artifact)
        if requirement is None:
            self._pending = False
            return
        constraints = requirement.data.get("constraints", {})
        material = str(constraints.get("material") or "steel")
        if material not in MarketSimulator.MATERIALS:
            material = "steel"
        quantity = int(constraints.get("quantity") or 1000)
        target_lead_time_days = int(constraints.get("target_lead_time_days") or 30)

        market = MarketSimulator(seed=42)
        spot_price = market.get_current_state(material).spot_price
        suppliers = self._build_pool(material, spot_price)

        self._pool = {
            "material": material,
            "quantity": quantity,
            "target_lead_time_days": target_lead_time_days,
            "spot_price": spot_price,
            "suppliers": suppliers,
        }
        logger.info(
            "agent_executing",
            agent=self.name,
            phase="suppliers_discovered",
            count=len(suppliers),
            correlation_id=self._correlation_id,
        )

    async def act(self, state: SwarmState) -> None:
        if not self._pending or self._pool is None:
            return
        artifact = SupplierListArtifact(
            data=self._pool,
            parent_ids=[REQUIREMENT_ARTIFACT_NAME],
            created_by=self.name,
            correlation_id=self._correlation_id,
        )
        state.put_artifact(artifact)
        suppliers = list(self._pool["suppliers"])
        logger.info(
            "artifact_created",
            agent=self.name,
            kind=artifact.kind,
            name=artifact.name,
            correlation_id=self._correlation_id,
        )
        if self._correlation_id is not None:
            state.expect_artifact(
                kind="evaluation", count=len(suppliers), correlation_id=self._correlation_id
            )
            state.expect_artifact(
                kind="quote", count=len(suppliers), correlation_id=self._correlation_id
            )
        for supplier in suppliers:
            await self.publish_event(
                Event(
                    type=ProcurementEventType.SUPPLIER_DISCOVERED,
                    source=self.name,
                    payload={
                        "supplier_id": str(supplier["supplier_id"]),
                        "material": self._pool["material"],
                        "artifact": artifact.name,
                    },
                    correlation_id=self._correlation_id,
                )
            )
        if self._correlation_id is not None:
            self._discovered_for.add(self._correlation_id)
        self._pending = False
        self._pool = None

    @staticmethod
    def _build_pool(material: str, spot_price: float) -> list[dict[str, Any]]:
        """Build deterministic supplier profiles from the CNP templates."""
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
