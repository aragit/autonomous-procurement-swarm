"""Unit tests for bilateral FSM and bartering logic."""

import pytest

from core.agents.buyer import BuyerOrchestrator
from core.agents.supplier import CostModel, SupplierAgent
from core.llm_engine import LLMEngineFactory
from core.protocol.fsm import BilateralFSM
from core.protocol.schema import MessageType, RFQPayload


@pytest.fixture
def mock_llm():
    return LLMEngineFactory.create(use_mock=True)


@pytest.fixture
def policy_engine():
    from core.protocol.policy_engine import PolicyEngine

    return PolicyEngine()


def test_bilateral_fsm_init():
    fsm = BilateralFSM(max_turns=4)
    assert fsm.turn_count == 0
    assert not fsm.has_offer
    assert not fsm.is_terminal


def test_bilateral_fsm_offer_then_accept():
    fsm = BilateralFSM(max_turns=4)
    assert fsm.record_message(MessageType.OFFER, "buyer", {"unit_price": 100.0})
    assert fsm.has_offer
    assert fsm.record_message(MessageType.ACCEPT, "supplier", {"final_price": 100.0})
    assert fsm.is_terminal
    assert fsm.turn_count == 2


def test_bilateral_fsm_cannot_accept_without_offer():
    fsm = BilateralFSM(max_turns=4)
    # Try to accept with no prior offer
    assert not fsm.record_message(MessageType.ACCEPT, "supplier", {"final_price": 100.0})
    assert fsm.turn_count == 0


def test_bilateral_fsm_max_turns_timeout():
    fsm = BilateralFSM(max_turns=2)
    fsm.record_message(MessageType.OFFER, "buyer", {"unit_price": 100.0})
    fsm.record_message(MessageType.COUNTER, "supplier", {"counter_price": 90.0})
    # Third message should fail (max turns = 2)
    assert not fsm.record_message(MessageType.COUNTER, "buyer", {"counter_price": 95.0})


def test_bilateral_fsm_reject_ends_terminal():
    fsm = BilateralFSM(max_turns=4)
    fsm.record_message(MessageType.OFFER, "buyer", {"unit_price": 100.0})
    fsm.record_message(MessageType.REJECT, "supplier", {"reason": "too low"})
    assert fsm.is_terminal
    # No further messages allowed
    assert not fsm.record_message(MessageType.COUNTER, "buyer", {"counter_price": 110.0})


@pytest.mark.asyncio
async def test_supplier_responds_to_offer(mock_llm):
    supplier = SupplierAgent(
        "TestSupplier",
        mock_llm,
        CostModel(
            base_cost_per_unit=300.0,
            logistics_premium_per_unit=50.0,
            capacity_units=1000,
            current_utilization=0.0,
            min_margin_pct=0.10,
            reliability_score=0.8,
            esg_carbon_per_unit=100.0,
        ),
    )

    # Offer well above floor (floor = 350 * 1.1 = 385)
    response = await supplier.respond_to_offer(500.0, 100, "steel", turn=0, max_turns=4)
    assert response.type == MessageType.ACCEPT


@pytest.mark.asyncio
async def test_supplier_counters_low_offer(mock_llm):
    supplier = SupplierAgent(
        "TestSupplier",
        mock_llm,
        CostModel(
            base_cost_per_unit=300.0,
            logistics_premium_per_unit=50.0,
            capacity_units=1000,
            current_utilization=0.0,
            min_margin_pct=0.10,
            reliability_score=0.8,
            esg_carbon_per_unit=100.0,
        ),
    )

    # Offer below floor (floor = 385)
    response = await supplier.respond_to_offer(350.0, 100, "steel", turn=0, max_turns=4)
    assert response.type == MessageType.COUNTER
    assert response.payload["counter_price"] > 350.0  # Must improve for supplier


@pytest.mark.asyncio
async def test_full_bilateral_thread(mock_llm, policy_engine):
    buyer = BuyerOrchestrator("TestBuyer", mock_llm, policy_engine)
    supplier = SupplierAgent(
        "TestSupplier",
        mock_llm,
        CostModel(
            base_cost_per_unit=300.0,
            logistics_premium_per_unit=50.0,
            capacity_units=1000,
            current_utilization=0.0,
            min_margin_pct=0.10,
            reliability_score=0.8,
            esg_carbon_per_unit=100.0,
        ),
    )

    rfq = RFQPayload(
        session_id="bil-test",
        material="steel",
        quantity=100,
        max_unit_price=600.0,
        target_lead_time_days=30,
        delivery_window_start="2026-08-15",
        delivery_window_end="2026-09-15",
        payment_terms="net_30",
    )

    result = await buyer.barter_with_supplier(supplier, rfq, initial_bid_price=500.0, max_turns=4)

    assert "success" in result
    assert "turns" in result
    assert result["turns"] <= 4
    assert len(result["history"]) == result["turns"]

    # First message should be buyer's OFFER
    assert result["history"][0]["type"] == "offer"
    assert result["history"][0]["sender"] == "TestBuyer"

    # Last message should be terminal (accept, reject, or counter at max)
    last = result["history"][-1]
    assert last["type"] in {"accept", "reject", "counter"}

    if result["success"]:
        assert result["final_price"] is not None
        assert result["final_price"] > 0
