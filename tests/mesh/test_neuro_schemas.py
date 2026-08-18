"""Unit tests for the neuro schema models (Pydantic constraints).

No Ray / LLM required.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mesh.neuro import (
    NegotiatorProposal,
    NegotiatorQuote,
    QuoteMetadata,
    ScoutProposal,
    SupplierDiscoveryItem,
)

# ─── ScoutProposal ────────────────────────────────────────────────────────────


def _valid_supplier(**overrides) -> SupplierDiscoveryItem:
    base = {
        "supplier_id": "MinerCorp_A",
        "material": "steel",
        "base_cost_per_unit": 100.0,
        "logistics_premium_per_unit": 5.0,
        "capacity_units": 5000,
        "current_utilization": 0.3,
        "min_margin_pct": 0.2,
        "reliability_score": 0.9,
        "esg_carbon_per_unit": 400.0,
    }
    base.update(overrides)
    return SupplierDiscoveryItem(**base)


def _valid_scout(**overrides) -> ScoutProposal:
    base = {
        "correlation_id": "REQ-001",
        "material": "steel",
        "quantity": 1000,
        "target_lead_time_days": 30,
        "spot_price": 450.0,
        "confidence": 1.0,
        "suppliers": [_valid_supplier().model_dump()],
    }
    base.update(overrides)
    return ScoutProposal(**base)


def test_scout_valid_construction():
    scout = _valid_scout()
    assert scout.material == "steel"
    assert len(scout.suppliers) == 1


def test_scout_rejects_invalid_material():
    with pytest.raises(ValidationError):
        _valid_scout(material="unobtanium")


def test_scout_rejects_empty_suppliers():
    with pytest.raises(ValidationError):
        _valid_scout(suppliers=[])


def test_scout_rejects_non_positive_quantity():
    with pytest.raises(ValidationError):
        _valid_scout(quantity=0)


def test_scout_to_kernel_payload_exposes_economic_bounds():
    payload = _valid_scout().to_kernel_payload()
    assert payload["material"] == "steel"
    assert payload["price"] == 450.0
    assert payload["lead_time_days"] == 30
    assert payload["payment_terms"] == "net_30"
    assert payload["quantity"] == 1000
    assert payload["confidence"] == 1.0


def test_scout_to_pool_dict_round_trip():
    supplier = _valid_supplier()
    scout = _valid_scout(suppliers=[supplier.model_dump()])
    pool = scout.to_pool_dict()
    assert pool["material"] == "steel"
    assert pool["suppliers"][0]["supplier_id"] == "MinerCorp_A"


def test_supplier_rejects_negative_reliability():
    with pytest.raises(ValidationError):
        _valid_supplier(reliability_score=1.5)


def test_supplier_rejects_negative_carbon():
    with pytest.raises(ValidationError):
        _valid_supplier(esg_carbon_per_unit=-1.0)


# ─── NegotiatorQuote / NegotiatorProposal ───────────────────────────────────


def _valid_quote(**overrides) -> NegotiatorQuote:
    base = {
        "supplier_id": "S1",
        "price": 135.0,
        "terms": "net_30",
        "metadata": {
            "quantity": 1000,
            "lead_time_days": 30,
            "carbon_footprint_kg": 4000.0,
            "reliability_score": 0.9,
        },
    }
    base.update(overrides)
    return NegotiatorQuote(**base)


def _valid_neural_quote(**overrides) -> NegotiatorProposal:
    base = {
        "correlation_id": "REQ-001",
        "supplier_id": "S1",
        "quote": _valid_quote(),
        "confidence": 1.0,
    }
    base.update(overrides)
    return NegotiatorProposal(**base)


def test_negotiator_valid_construction():
    q = _valid_neural_quote()
    assert q.quote.price == 135.0
    assert q.quote.metadata.quantity == 1000


def test_negotiator_rejects_invalid_terms():
    with pytest.raises(ValidationError):
        _valid_quote(terms="pay_later")


def test_negotiator_rejects_non_positive_price():
    with pytest.raises(ValidationError):
        _valid_quote(price=0.0)


def test_negotiator_rejects_out_of_range_lead_time():
    with pytest.raises(ValidationError):
        QuoteMetadata(
            quantity=1000, lead_time_days=0, carbon_footprint_kg=1.0, reliability_score=0.9
        )


def test_negotiator_to_kernel_payload_includes_budget():
    q = _valid_neural_quote()
    payload = q.to_kernel_payload(material="steel", quantity=1000, budget=500000.0)
    assert payload["supplier_id"] == "S1"
    assert payload["material"] == "steel"
    assert payload["price"] == 135.0
    assert payload["payment_terms"] == "net_30"
    assert payload["budget"] == 500000.0
    assert payload["total_price"] == 135000.0


def test_negotiator_to_kernel_payload_without_budget_still_valid():
    q = _valid_neural_quote()
    payload = q.to_kernel_payload(material="steel", quantity=1000, budget=None)
    assert payload["budget"] == 0.0
    assert "total_price" in payload
