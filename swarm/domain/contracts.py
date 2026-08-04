"""Supplier contract model and validation (Phase 8 integration layer).

A :class:`Contract` is a deterministic, serializable agreement bound to one
supplier. The :class:`swarm.domain.agents.contract_validation_agent.
ContractValidationAgent` applies a contract (or a configurable no-contract policy)
to a decision and emits a :class:`~swarm.domain.artifacts.
ContractValidationArtifact` describing whether the decision is contract-valid.

Contracts are *data*; validation is pure so the same decision + contract always
yields the same outcome — keeping the execution trace replay-safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

COMPLIANCE_FLAG_ACTIVE = "active"
COMPLIANCE_FLAG_SUSPENDED = "suspended"


class ContractStatus(StrEnum):
    """Lifecycle state of a supplier contract."""

    ACTIVE = "active"
    EXPIRED = "expired"
    SUSPENDED = "suspended"


@dataclass(frozen=True)
class PricingRule:
    """A deterministic pricing constraint for a contract line.

    ``data`` contract::

        {
            "material": str,
            "max_unit_price": float | None,     # None → price un-capped
        }
    """

    material: str
    max_unit_price: float | None = None

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> PricingRule:
        return cls(
            material=str(record["material"]),
            max_unit_price=record.get("max_unit_price"),
        )

    def matches(self, material: str, unit_price: float | None) -> bool:
        if self.material != material:
            return False
        if self.max_unit_price is not None and unit_price is not None:
            return unit_price <= self.max_unit_price
        return True


@dataclass(frozen=True)
class Contract:
    """A deterministic supplier contract.

    ``data`` contract::

        {
            "contract_id": str,
            "supplier_id": str,
            "allowed_items": [str, ...],        # materials the contract covers
            "pricing_rules": [PricingRule, ...],
            "expiry_date": str | None,          # ISO-8601 date; None → no expiry
            "compliance_flags": {str, ...},     # e.g. {"active"}
            "status": str,                      # ContractStatus value
        }
    """

    contract_id: str
    supplier_id: str
    allowed_items: list[str] = field(default_factory=list)
    pricing_rules: list[PricingRule] = field(default_factory=list)
    expiry_date: str | None = None
    compliance_flags: set[str] = field(default_factory=set)
    status: ContractStatus = ContractStatus.ACTIVE

    def is_active(self) -> bool:
        """Whether the contract is currently usable for ordering."""
        if self.status != ContractStatus.ACTIVE:
            return False
        if self.expiry_date is not None and self._has_expired():
            return False
        return COMPLIANCE_FLAG_ACTIVE in self.compliance_flags

    def _has_expired(self) -> bool:
        if not self.expiry_date:
            return False
        try:
            expiry = datetime.fromisoformat(self.expiry_date)
        except ValueError:
            return False
        now = datetime.now(UTC)
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=UTC)
        return expiry < now

    def allows_item(self, material: str) -> bool:
        return material in self.allowed_items

    def pricing_ok(self, material: str, unit_price: float | None) -> bool:
        relevant = [r for r in self.pricing_rules if r.material == material]
        if not relevant:
            return True
        return any(r.matches(material, unit_price) for r in relevant)

    def validate(
        self,
        *,
        supplier_id: str,
        material: str,
        unit_price: float | None = None,
    ) -> tuple[bool, str | None]:
        """Validate a prospective order line against this contract.

        Returns ``(valid, reason)``. ``reason`` is ``None`` when valid.
        """
        if supplier_id != self.supplier_id:
            return False, f"Contract {self.contract_id} is not for supplier {supplier_id}"
        if not self.is_active():
            return False, f"Contract {self.contract_id} is not active"
        if not self.allows_item(material):
            return False, f"Material {material} not covered by contract {self.contract_id}"
        if not self.pricing_ok(material, unit_price):
            return (
                False,
                f"Unit price {unit_price} exceeds contract {self.contract_id} cap "
                f"for {material}",
            )
        return True, None
