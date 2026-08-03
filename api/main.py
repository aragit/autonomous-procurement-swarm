"""FastAPI control plane for autonomous procurement swarm."""

import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from configs.settings import settings
from core.agents.buyer import BuyerOrchestrator
from core.agents.supplier import CostModel, SupplierAgent
from core.evaluator.scoring import EvaluationWeights, MultiCriteriaEvaluator
from core.ledger.repository import PostgresLedgerRepository
from core.llm_engine import LLMEngineFactory
from core.logging_config import configure_logging, get_logger
from core.market_simulator import MarketSimulator
from core.memory.heuristics import HeuristicReservationEstimator
from core.memory.semantic import PgVectorMemoryStore
from core.protocol.auction_orchestrator import AuctionOrchestrator
from core.protocol.policy_engine import PolicyContext, PolicyEngine
from swarm import SwarmState
from swarm.domain.events import CREATE_REQUIREMENT_INTENT
from swarm.domain.wiring import build_procurement_swarm

# Configure logging at import time
configure_logging()
logger = get_logger("api")

# Database URL
DB_URL = os.getenv(
    "DATABASE_URL", "postgresql+asyncpg://procurement:procurement@localhost:5433/procurement"
)

# Global state (initialized in lifespan)
ledger: PostgresLedgerRepository | None = None
shared_memory: HeuristicReservationEstimator | None = None
shared_vector_store: PgVectorMemoryStore | None = None

# Read-only swarm trace store: most recent wired-swarm executions by request_id.
swarm_states: dict[str, SwarmState] = {}
MAX_SWARM_STATES = 50


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """Startup and shutdown events."""
    global ledger, shared_memory, shared_vector_store
    logger.info("Starting up procurement API", db_url=DB_URL)
    ledger = PostgresLedgerRepository(DB_URL)
    await ledger.init_schema()

    shared_vector_store = PgVectorMemoryStore(DB_URL)
    await shared_vector_store.init_schema()

    shared_memory = HeuristicReservationEstimator()

    logger.info("Database and vector store initialized")
    yield
    await ledger.close()
    logger.info("Shutting down procurement API")


app = FastAPI(
    title="Autonomous Procurement Swarm API",
    version="0.2.0",
    lifespan=lifespan,
)


class AuctionRequest(BaseModel):
    material: str = "steel"
    quantity: int = Field(default=1000, gt=0)
    max_unit_price: float | None = None
    target_lead_time_days: int = 30
    supplier_count: int = Field(default=5, gt=0)
    enable_bartering: bool = True

    @field_validator("material")
    @classmethod
    def validate_material(cls, v: str) -> str:
        valid = settings.negotiation.valid_materials
        if v not in valid:
            raise ValueError(f"material must be one of {valid} (got '{v}')")
        return v


class AuctionResponse(BaseModel):
    session_id: str
    status: str
    winner: dict[str, Any] | None = None
    final_price: float | None = None
    scored_bids: list[dict[str, Any]] = []
    shortlist: list[dict[str, Any]] = []
    bartering_result: dict[str, Any] | None = None


class SwarmRequirementRequest(BaseModel):
    """Payload for dispatching a requirement into the deterministic swarm."""

    material: str = "aluminum"
    quantity: int = Field(default=1000, gt=0)
    budget: float = Field(default=2_000_000.0, gt=0)
    target_lead_time_days: int = 30
    max_carbon_per_unit: float | None = Field(default=None, gt=0)
    goal: str | None = None

    @field_validator("material")
    @classmethod
    def validate_material(cls, v: str) -> str:
        valid = settings.negotiation.valid_materials
        if v not in valid:
            raise ValueError(f"material must be one of {valid} (got '{v}')")
        return v


class SwarmDispatchResponse(BaseModel):
    request_id: str
    correlation_id: str
    decision: dict[str, Any] | None = None
    completions: dict[str, list[str]]
    event_count: int
    artifact_count: int


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "database": "connected" if ledger else "disconnected",
        "version": "0.2.0",
    }


@app.post("/auctions", response_model=AuctionResponse)
async def create_auction(
    request: AuctionRequest, background_tasks: BackgroundTasks
) -> AuctionResponse:
    """Start a new sealed-bid procurement auction."""
    session_id = str(uuid.uuid4())[:8]
    logger.info("Auction requested", session_id=session_id, material=request.material)

    market = MarketSimulator(seed=42)
    market_state = market.get_current_state(request.material)

    max_price = request.max_unit_price or market_state.spot_price * 1.2

    # Create heterogeneous suppliers
    suppliers = _create_suppliers(request.material, market_state.spot_price, request.supplier_count)

    buyer = BuyerOrchestrator("API_Buyer", LLMEngineFactory.create(use_mock=True), PolicyEngine())
    rfq = buyer.create_rfq(
        session_id=session_id,
        material=request.material,
        quantity=request.quantity,
        max_unit_price=max_price,
        target_lead_time_days=request.target_lead_time_days,
        budget=500_000.0,
    )

    policy_ctx = PolicyContext(
        buyer_max_budget_total=500_000.0,
        max_unit_price=max_price,
        max_carbon_limit_kg=2_000_000.0,
    )

    evaluator = MultiCriteriaEvaluator(
        weights=EvaluationWeights(),
        esg_baselines=settings.evaluation.esg_baselines,
    )

    orchestrator = AuctionOrchestrator(
        policy_engine=PolicyEngine(),
        evaluator=evaluator,
        ledger=ledger,
        memory=shared_memory,
        vector_store=shared_vector_store,
    )

    # Run sealed bid auction
    result = await orchestrator.run_sealed_bid_auction(
        session_id=session_id,
        rfq=rfq,
        suppliers=suppliers,
        policy_context=policy_ctx,
        market_spot_price=market_state.spot_price,
    )

    response = AuctionResponse(
        session_id=session_id,
        status=result["fsm_state"],
        winner=result.get("winner"),
        final_price=result.get("winner", {}).get("unit_price") if result.get("winner") else None,
        scored_bids=result.get("scored_bids", []),
        shortlist=result.get("shortlist", []),
    )

    # Optional bilateral bartering
    if request.enable_bartering and len(result.get("shortlist", [])) > 1:
        barter_result = await orchestrator.run_bilateral_bartering(
            session_id=session_id,
            buyer=buyer,
            shortlist=result["shortlist"],
            rfq=rfq,
            market_spot_price=market_state.spot_price,
        )
        response.bartering_result = barter_result
        if barter_result.get("success"):
            response.final_price = barter_result["best_deal"]["final_price"]
            response.winner = barter_result["best_deal"]

    logger.info("Auction completed", session_id=session_id, status=response.status)
    return response


@app.get("/auctions/{session_id}")
async def get_auction(session_id: str) -> dict[str, Any]:
    """Retrieve auction events from ledger."""
    if not ledger:
        raise HTTPException(status_code=503, detail="Ledger not available")

    events = await ledger.get_events(session_id)
    if not events:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session_id,
        "events": events,
        "chain_valid": await ledger.verify_chain(session_id),
    }


@app.get("/auctions/{session_id}/stats")
async def get_auction_stats(session_id: str) -> dict[str, Any]:
    """Get statistics for a specific auction session."""
    if not ledger:
        raise HTTPException(status_code=503, detail="Ledger not available")

    events = await ledger.get_events(session_id)
    bids = [e for e in events if e["message_type"] == "bid"]
    awards = [e for e in events if e["message_type"] == "award"]

    return {
        "session_id": session_id,
        "total_events": len(events),
        "bids_received": len(bids),
        "awards_made": len(awards),
        "chain_valid": await ledger.verify_chain(session_id),
    }


@app.get("/ledger/stats")
async def get_global_stats() -> dict[str, Any]:
    """Global ledger statistics."""
    if not ledger:
        raise HTTPException(status_code=503, detail="Ledger not available")
    return await ledger.get_stats()


@app.get("/suppliers/{supplier_id}/profile")
async def get_supplier_profile(supplier_id: str) -> dict[str, Any]:
    """Retrieve a supplier's behavioral profile learned from memory."""
    if not shared_memory:
        raise HTTPException(status_code=503, detail="Memory not available")
    profile = shared_memory.get_profile(supplier_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="No profile for supplier")
    return profile.to_dict()


@app.get("/suppliers")
async def list_suppliers() -> dict[str, Any]:
    """List all suppliers with profiles in the memory store."""
    if not shared_memory:
        raise HTTPException(status_code=503, detail="Memory not available")
    return {"suppliers": [profile.to_dict() for profile in shared_memory.all_profiles()]}


@app.get("/suppliers/similar")
async def find_similar_suppliers(supplier_id: str | None = None, n: int = 3) -> dict[str, Any]:
    """Find suppliers similar to a given supplier (by id or profile)."""
    if not shared_memory or not shared_vector_store:
        raise HTTPException(status_code=503, detail="Memory not available")

    if supplier_id:
        profile = shared_memory.get_profile(supplier_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="No profile for supplier")
        query = profile.to_dict()
    else:
        query = {"material": "steel", "unit_price": 100.0, "esg_carbon_per_unit": 1200.0}

    similar = await shared_vector_store.query_similar_suppliers(query, n_results=n)
    return {"query_supplier_id": supplier_id, "similar": similar}


# ─── Deterministic swarm trace endpoints (read-only) ────────────────────────


def _remember(state: SwarmState) -> None:
    """Keep the most recent swarm states for read-only trace lookups."""
    swarm_states[state.request_id] = state
    while len(swarm_states) > MAX_SWARM_STATES:
        swarm_states.pop(next(iter(swarm_states)))


@app.post("/swarm/requirements", response_model=SwarmDispatchResponse)
async def dispatch_swarm_requirement(
    request: SwarmRequirementRequest,
) -> SwarmDispatchResponse:
    """Dispatch a requirement into the deterministic Phase 4 swarm.

    Runs the full parallel flow (requirement → strategy → per-supplier
    discovery → evaluation → quoting → completion-tracked decision) without any
    LLM, then stores the resulting state so the trace endpoints below can serve
    it. An optional ``max_carbon_per_unit`` constraint selects the low-carbon
    strategy; a tight budget selects the cost-optimized strategy; otherwise the
    balanced strategy is used.
    """
    request_id = f"REQ-{uuid.uuid4().hex[:8].upper()}"
    swarm = build_procurement_swarm(
        request_id=request_id,
        goal=request.goal or f"Source {request.quantity} units of {request.material}",
    )
    await swarm.start()
    correlation_id = f"{request_id}-CONV"
    await swarm.send_message(
        CREATE_REQUIREMENT_INTENT,
        {
            "text": f"Source {request.quantity} units of {request.material}",
            "material": request.material,
            "quantity": request.quantity,
            "budget": request.budget,
            "target_lead_time_days": request.target_lead_time_days,
            "max_carbon_per_unit": request.max_carbon_per_unit,
        },
        sender="user",
        correlation_id=correlation_id,
    )
    await swarm.shutdown()
    _remember(swarm.state)

    decision = swarm.state.get_artifact("decision")
    return SwarmDispatchResponse(
        request_id=request_id,
        correlation_id=correlation_id,
        decision=decision.data if decision else None,
        completions=swarm.state.completions,
        event_count=len(swarm.state.events),
        artifact_count=len(swarm.state.artifacts),
    )


def _swarm_state(request_id: str) -> SwarmState:
    state = swarm_states.get(request_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"No swarm run for request_id {request_id}")
    return state


@app.get("/swarm/trace/{request_id}")
async def get_swarm_execution_trace(request_id: str) -> dict[str, Any]:
    """Read-only execution trace for one swarm conversation."""
    state = _swarm_state(request_id)
    correlation_ids = sorted(
        {event.correlation_id for event in state.events if event.correlation_id}
    )
    cid = correlation_ids[0] if correlation_ids else request_id
    return state.get_execution_trace(cid)


@app.get("/swarm/trace/{request_id}/completions")
async def get_swarm_completions(request_id: str) -> dict[str, Any]:
    """Completion groups closed per correlation id for a swarm run."""
    state = _swarm_state(request_id)
    return {"request_id": request_id, "completions": state.completions}


@app.get("/swarm/state/{request_id}")
async def get_swarm_state(request_id: str) -> dict[str, Any]:
    """Full serialized read-only snapshot of a swarm run's state."""
    state = _swarm_state(request_id)
    return state.to_dict()


@app.get("/swarm/explanation/{request_id}")
async def get_swarm_explanation(request_id: str) -> dict[str, Any]:
    """Human-readable explanation of the swarm's supplier selection.

    Serves the ``DecisionExplanationArtifact`` (deterministic, no LLM): the
    selected supplier, the strategy used, the deciding factors, and why every
    other supplier was rejected.
    """
    state = _swarm_state(request_id)
    explanation = state.get_artifact("decision_explanation")
    if explanation is None:
        raise HTTPException(
            status_code=404,
            detail=f"No decision explanation for request_id {request_id}",
        )
    return {"request_id": request_id, "explanation": explanation.data}


def _create_suppliers(material: str, spot_price: float, count: int = 5) -> list[SupplierAgent]:
    """Create a pool of heterogeneous suppliers."""
    configs: list[dict[str, Any]] = [
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

    from core.llm_engine import LLMEngineFactory

    llm = LLMEngineFactory.create(use_mock=True)

    suppliers = []
    for cfg in configs[:count]:
        suppliers.append(
            SupplierAgent(
                cfg["name"],
                llm,
                CostModel(
                    base_cost_per_unit=spot_price * cfg["base_mult"],
                    logistics_premium_per_unit=cfg["logistics"],
                    capacity_units=cfg["cap"],
                    current_utilization=cfg["util"],
                    min_margin_pct=cfg["margin"],
                    reliability_score=cfg["rel"],
                    esg_carbon_per_unit=cfg["carbon"],
                ),
            )
        )
    return suppliers
