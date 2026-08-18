"""Pure-Python symbolic validation (the "symbolic" half of the bridge).

The :class:`~mesh.actors.base.SafetyKernelActor` is a Ray actor, but the
validation logic it runs is plain, deterministic Python.  Extracting it here
lets the Neuro-Symbolic bridge and its retry loop be unit-tested against the
*real* policy rules without a Ray cluster.

These checks are the non-overridable "root of trust" — they mirror (and must
stay consistent with) ``SafetyKernelActor.validate``.
"""

from __future__ import annotations

from typing import Any

from mesh.neuro.types import NeuralProposal, SymbolicVerdict

#: Hard ceilings enforced on every neural proposal.
MAX_PRICE = 10_000_000.0
MIN_LEAD_TIME_DAYS = 1
MAX_LEAD_TIME_DAYS = 365
MIN_CONFIDENCE = 0.5
MAX_UNIT_PRICE = 1_000_000.0

VALID_PAYMENT_TERMS = frozenset({"net_30", "net_60", "cod", "letter_of_credit"})
VALID_MATERIALS = frozenset({"steel", "aluminum", "copper", "plastic", "lumber", "rubber"})


def _price_from(payload: dict[str, Any]) -> float | None:
    if "price" in payload:
        return float(payload["price"])
    if "total_price" in payload:
        return float(payload["total_price"])
    return None


def symbolic_validate(proposal: NeuralProposal) -> SymbolicVerdict:
    """Deterministic, side-effect-free validation of a neural proposal.

    A pure function: same input always yields the same verdict.  Used both by
    :class:`~mesh.actors.base.SafetyKernelActor` (cluster mode) and the test
    harness (offline mode).
    """
    payload = proposal.payload
    failures: list[str] = []

    # 1. Price bounds
    if "price" in payload:
        price = float(payload["price"])
        if price <= 0 or price > MAX_PRICE:
            failures.append(f"price_out_of_bounds: {price}")

    # 2. Lead time bounds
    if "lead_time_days" in payload:
        lt = int(payload["lead_time_days"])
        if lt < MIN_LEAD_TIME_DAYS or lt > MAX_LEAD_TIME_DAYS:
            failures.append(f"lead_time_out_of_bounds: {lt}")

    # 3. Payment terms whitelist
    if "payment_terms" in payload:
        terms = payload["payment_terms"]
        if terms not in VALID_PAYMENT_TERMS:
            failures.append(f"invalid_payment_terms: {terms}")

    # 4. Material whitelist
    if "material" in payload:
        material = payload["material"]
        if material not in VALID_MATERIALS:
            failures.append(f"invalid_material: {material}")

    # 5. Budget compliance (price * quantity <= budget, when both present)
    unit_price = _price_from(payload)
    if unit_price is not None and "budget" in payload:
        budget = float(payload["budget"])
        quantity = int(payload.get("quantity", 1))
        if unit_price * quantity > budget:
            failures.append(f"exceeds_budget: {unit_price * quantity} > {budget}")

    # 6. Confidence threshold
    if proposal.confidence < MIN_CONFIDENCE:
        failures.append(f"confidence_below_threshold: {proposal.confidence}")

    if failures:
        return SymbolicVerdict.rejected(
            reason=f"VALIDATION_FAILED: {'; '.join(failures)}",
            violations=failures,
        )

    # Clamp to safe ranges.
    clamped = dict(payload)
    if "price" in clamped:
        clamped["price"] = max(0.01, min(float(clamped["price"]), MAX_UNIT_PRICE))
    if "total_price" in clamped:
        clamped["total_price"] = max(0.01, min(float(clamped["total_price"]), MAX_UNIT_PRICE))
    if "lead_time_days" in clamped:
        clamped["lead_time_days"] = max(
            MIN_LEAD_TIME_DAYS,
            min(int(clamped["lead_time_days"]), MAX_LEAD_TIME_DAYS),
        )

    return SymbolicVerdict.approved_verdict(clamped_payload=clamped)
