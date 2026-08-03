"""Deterministic pricing helpers for the Phase 2 procurement swarm.

Quotes and pre-quote evaluation prices are derived from each supplier's
CostModel-shaped profile using the same floor-price rule as
``core.agents.supplier.SupplierAgent._compute_floor_price`` (unit cost plus the
supplier's minimum margin). Using a fixed floor with no markup keeps the
adapter layer deterministic while staying consistent with the existing CNP
logic.
"""

from typing import Any

from swarm.domain.artifacts import SupplierListArtifact

DEFAULT_PAYMENT_TERMS = "net_30"
DEFAULT_BID_BOND_PCT = 0.05


def floor_price(supplier: dict[str, Any]) -> float:
    """Deterministic ask price for ``supplier``.

    Mirrors ``SupplierAgent._compute_floor_price``: unit cost (base plus
    logistics) inflated by the supplier's minimum margin.
    """
    unit_cost = float(supplier["base_cost_per_unit"]) + float(
        supplier["logistics_premium_per_unit"]
    )
    return round(unit_cost * (1 + float(supplier["min_margin_pct"])), 2)


def lead_time_days(supplier: dict[str, Any], target: int, index: int) -> int:
    """Deterministic lead time: a small stable offset around the target.

    The offset is derived from the supplier's position in the pool, so the same
    requirement always yields the same lead times without any randomness.
    """
    return max(1, target + ((index % 5) - 2))


def carbon_footprint(supplier: dict[str, Any], quantity: int) -> float:
    """Total order carbon (kg CO2e) for ``quantity`` units from ``supplier``."""
    return round(float(supplier["esg_carbon_per_unit"]) * quantity, 2)


def bid_bond_amount(
    unit_price: float, quantity: int, bond_pct: float = DEFAULT_BID_BOND_PCT
) -> float:
    """Bid bond for a quote, matching the core CNP convention."""
    return round(unit_price * quantity * bond_pct, 2)


def supplier_ids(pool: SupplierListArtifact) -> list[str]:
    """Extract supplier ids from a supplier list artifact."""
    return [str(supplier["supplier_id"]) for supplier in pool.data["suppliers"]]
