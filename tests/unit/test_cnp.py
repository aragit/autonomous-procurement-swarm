"""Unit tests for CNP protocol, FSM, and auction orchestration."""

import pytest

from core.agents.supplier import CostModel, SupplierAgent
from core.llm_engine import LLMEngineFactory
from core.protocol.auction_orchestrator import AuctionOrchestrator
from core.protocol.fsm import GlobalAuctionFSM, GlobalAuctionState
from core.protocol.policy_engine import PolicyContext, PolicyEngine
from core.protocol.schema import BidPayload, MessageType, RFQPayload


@pytest.fixture
def mock_llm():
    return LLMEngineFactory.create(use_mock=True)


@pytest.fixture
def policy_engine():
    return PolicyEngine()


def test_rfq_validation():
    rfq = RFQPayload(
        session_id="test-1",
        material="steel",
        quantity=1000,
        max_unit_price=500.0,
        target_lead_time_days=30,
        delivery_window_start="2026-08-15",
        delivery_window_end="2026-09-15",
        payment_terms="net_30",
    )
    assert rfq.quantity == 1000
    assert rfq.payment_terms == "net_30"


def test_rfq_invalid_date_order():
    with pytest.raises(ValueError):
        RFQPayload(
            session_id="test-1",
            material="steel",
            quantity=1000,
            max_unit_price=500.0,
            target_lead_time_days=30,
            delivery_window_start="2026-09-15",
            delivery_window_end="2026-08-15",  # Invalid: end before start
            payment_terms="net_30",
        )


def test_policy_blacklist():
    engine = PolicyEngine()
    ctx = PolicyContext(
        buyer_max_budget_total=1_000_000.0,
        blacklisted_vendors={"BadVendor"},
        max_unit_price=1000.0,
    )
    bid = {"supplier_id": "BadVendor", "unit_price": 100.0, "quantity": 10}
    passed, reason = engine.evaluate_bid(bid, ctx)
    assert not passed
    assert "BLACKLISTED" in reason


def test_policy_budget_cap():
    engine = PolicyEngine()
    ctx = PolicyContext(buyer_max_budget_total=1000.0, max_unit_price=500.0)
    bid = {
        "supplier_id": "GoodVendor",
        "unit_price": 200.0,
        "quantity": 10,
        "bid_bond_amount": 1000.0,
    }
    passed, reason = engine.evaluate_bid(bid, ctx)
    assert not passed
    assert "BUDGET" in reason


def test_fsm_transitions():
    fsm = GlobalAuctionFSM("test-session")
    assert fsm.state == GlobalAuctionState.INIT

    assert fsm.transition(GlobalAuctionState.RFQ_BROADCAST)
    assert fsm.transition(GlobalAuctionState.BID_COLLECTION)
    assert fsm.transition(GlobalAuctionState.EVALUATION)
    assert fsm.transition(GlobalAuctionState.AWARDED)

    # Cannot transition from terminal
    assert not fsm.transition(GlobalAuctionState.TERMINATED)


def test_fsm_invalid_transition():
    fsm = GlobalAuctionFSM("test")
    fsm.transition(GlobalAuctionState.RFQ_BROADCAST)
    # Cannot skip BID_COLLECTION to EVALUATION
    assert not fsm.can_transition(GlobalAuctionState.EVALUATION)


@pytest.mark.asyncio
async def test_supplier_responds_to_rfq(mock_llm):
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
        session_id="test",
        material="steel",
        quantity=100,
        max_unit_price=500.0,
        target_lead_time_days=30,
        delivery_window_start="2026-08-15",
        delivery_window_end="2026-09-15",
        payment_terms="net_30",
    )

    response = await supplier.respond_to_rfq(rfq)
    assert response.type == MessageType.BID
    bid = BidPayload(**response.payload)
    assert bid.supplier_id == "TestSupplier"
    assert bid.unit_price > 0
    assert bid.unit_price <= 500.0


@pytest.mark.asyncio
async def test_supplier_rejects_when_no_capacity(mock_llm):
    supplier = SupplierAgent(
        "FullSupplier",
        mock_llm,
        CostModel(
            base_cost_per_unit=300.0,
            logistics_premium_per_unit=50.0,
            capacity_units=100,
            current_utilization=1.0,  # 100% utilized!
            min_margin_pct=0.10,
            reliability_score=0.8,
            esg_carbon_per_unit=100.0,
        ),
    )

    rfq = RFQPayload(
        session_id="test",
        material="steel",
        quantity=500,  # Needs 500, has 0 available
        max_unit_price=500.0,
        target_lead_time_days=30,
        delivery_window_start="2026-08-15",
        delivery_window_end="2026-09-15",
        payment_terms="net_30",
    )

    response = await supplier.respond_to_rfq(rfq)
    assert response.type == MessageType.REJECT_BID


@pytest.mark.asyncio
async def test_full_auction_3_suppliers(mock_llm, policy_engine):
    suppliers = [
        SupplierAgent(
            f"Supplier_{i}",
            mock_llm,
            CostModel(
                base_cost_per_unit=300.0 + (i * 50),
                logistics_premium_per_unit=20.0,
                capacity_units=10000,
                current_utilization=0.2,
                min_margin_pct=0.10 + (i * 0.05),
                reliability_score=0.8,
                esg_carbon_per_unit=100.0,
            ),
        )
        for i in range(3)
    ]

    rfq = RFQPayload(
        session_id="auction-test",
        material="steel",
        quantity=100,
        max_unit_price=600.0,
        target_lead_time_days=30,
        delivery_window_start="2026-08-15",
        delivery_window_end="2026-09-15",
        payment_terms="net_30",
    )

    ctx = PolicyContext(
        buyer_max_budget_total=1_000_000.0,
        max_unit_price=600.0,
    )

    orchestrator = AuctionOrchestrator(policy_engine)
    result = await orchestrator.run_sealed_bid_auction(
        session_id="auction-test",
        rfq=rfq,
        suppliers=suppliers,
        policy_context=ctx,
        market_spot_price=450.0,
        timeout_sec=5.0,
    )

    assert result["success"] is True
    assert result["fsm_state"] == "AWARDED"
    assert result["winner"] is not None
    assert len(result["valid_bids"]) == 3
    assert len(result["rejections"]) == 2  # 2 losers get OUTBID rejections
    # Sprint 2: auction result includes multi-criteria scores
    assert "scored_bids" in result
    assert len(result["scored_bids"]) == 3
    assert "shortlist" in result
