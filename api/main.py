"""FastAPI control plane for autonomous procurement swarm.

.. deprecated::
    Use the V2 mesh API (``api.v2:app``) for new development. The V1 endpoints
    at ``/swarm/*`` and ``/auctions/*`` remain available but are no longer the
    recommended integration path. The V2 runtime provides the same capabilities
    through a distributed Ray mesh with typed blackboard channels.
"""

import warnings

warnings.warn(
    "api.main (V1 control plane) is deprecated. Use api.v2:app (V2 mesh runtime) "
    "for new development. The V1 endpoints at /swarm/* and /auctions/* remain "
    "available but may be removed in a future major release.",
    DeprecationWarning,
    stacklevel=2,
)

import os  # noqa: E402
import uuid  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from typing import Any  # noqa: E402

from fastapi import BackgroundTasks, FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field, field_validator  # noqa: E402

from api.v2 import router as _v2_mesh_router  # noqa: E402
from configs.settings import settings  # noqa: E402
from core.agents.buyer import BuyerOrchestrator  # noqa: E402
from core.agents.supplier import CostModel, SupplierAgent  # noqa: E402
from core.evaluator.scoring import EvaluationWeights, MultiCriteriaEvaluator  # noqa: E402
from core.ledger.repository import PostgresLedgerRepository  # noqa: E402
from core.llm_engine import LLMEngineFactory  # noqa: E402
from core.logging_config import configure_logging, get_logger  # noqa: E402
from core.market_simulator import MarketSimulator  # noqa: E402
from core.memory.heuristics import HeuristicReservationEstimator  # noqa: E402
from core.memory.semantic import PgVectorMemoryStore  # noqa: E402
from core.protocol.auction_orchestrator import AuctionOrchestrator  # noqa: E402
from core.protocol.policy_engine import PolicyContext, PolicyEngine  # noqa: E402
from swarm import SwarmState  # noqa: E402
from swarm.api.policies import router as _policies_router  # noqa: E402
from swarm.api.procurement import router as _procurement_router  # noqa: E402
from swarm.api.simulation import router as _simulation_router  # noqa: E402
from swarm.api.strategy import app as llm_observability_app  # noqa: E402
from swarm.api.strategy import set_state_provider  # noqa: E402
from swarm.core.timeline import build_timeline  # noqa: E402
from swarm.domain.agents import (  # noqa: E402
    ApprovalAgent,
    ExecutionTrackingAgent,
    PurchaseOrderAgent,
)
from swarm.domain.artifacts import (  # noqa: E402
    EXECUTION_AUTHORIZATION_ARTIFACT_NAME,
    EXECUTION_STATUS_ARTIFACT_NAME,
    EXTERNAL_CALL_ARTIFACT_NAME,
    GOVERNANCE_DECISION_ARTIFACT_NAME,
    PURCHASE_ORDER_ARTIFACT_NAME,
    RISK_ASSESSMENT_ARTIFACT_NAME,
)
from swarm.domain.events import (  # noqa: E402
    CREATE_REQUIREMENT_INTENT,
    RECORD_OUTCOME_INTENT,
)
from swarm.domain.wiring import build_procurement_swarm  # noqa: E402
from swarm.integrations import (  # noqa: E402
    PROVIDERS,
    BaseConnector,
    ConnectorConfig,
    build_connector,
    build_connector_from_env,
)
from swarm.memory import default_store  # noqa: E402

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

# Default base connector used by the execution endpoints. Resolved from the
# runtime environment (PROCUREMENT_CONNECTOR_PROVIDER /
# PROCUREMENT_CONNECTOR_MODE) so the same swarm targets a different external
# system per environment without code changes — DEV defaults to the in-memory
# MockConnector, STAGING to SupplierAPIConnector, PROD to CoupaConnector. When no
# credentials are configured the adapter simulates deterministically, so every
# external interaction stays audited via ExternalCallArtifact and replay-safe.
def _default_base_connector() -> BaseConnector:
    return build_connector_from_env()


default_base_connector: BaseConnector = _default_base_connector()

# Deterministic, in-memory supplier performance shared across requests (Phase 5).
# No external database: history is append-only and reproducible from the
# recorded outcome sequence.
supplier_memory = default_store


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

# Mount v0.9 LLM observability sub-app at /llm-obs and wire its state
# provider to look up remembered swarm states by correlation ID.
app.mount("/llm-obs", llm_observability_app)

def _lookup_swarm_state_by_correlation_id(cid: str) -> SwarmState | None:
    """Resolve a remembered swarm state by correlation ID prefix."""
    for state in swarm_states.values():
        if state.request_id in cid or cid.startswith(state.request_id):
            return state
    return None

set_state_provider(_lookup_swarm_state_by_correlation_id)

# Mount the single-entry procurement endpoint
app.include_router(_procurement_router)

# Mount the deterministic replay / simulation endpoints
app.include_router(_simulation_router)

# Mount the closed-loop policy learning + promotion endpoints (v1.1 Step 22)
app.include_router(_policies_router)

# Mount the v2 (Mesh Runtime) API — Ray-backed procurement, separate namespace.
app.include_router(_v2_mesh_router)


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
        supplier_memory=supplier_memory,
        base_connector=default_base_connector,
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


@app.get("/swarm/timeline/{request_id}")
async def get_swarm_timeline(request_id: str) -> dict[str, Any]:
    """Causally ordered, read-only timeline for one swarm run.

    Merges that run's events and artifacts into a single stream sorted by
    timestamp (with a stable, replay-deterministic tie-break), normalized into a
    flat shape with phase markers and sensitive payload fields masked. Pure
    projection — no agent logic is re-run and no external system is consulted.
    """
    state = _swarm_state(request_id)
    return build_timeline(state).model_dump()


class OutcomeRecordRequest(BaseModel):
    """A post-decision procurement outcome fed back into supplier memory."""

    supplier_id: str = Field(..., min_length=1)
    delivered_on_time: bool = True
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
    actual_price: float = Field(..., gt=0)
    carbon_score: float = Field(default=0.0, ge=0.0)


@app.post("/swarm/{request_id}/outcome")
async def record_swarm_outcome(request_id: str, request: OutcomeRecordRequest) -> dict[str, Any]:
    """Feed a post-decision outcome back into the deterministic supplier memory.

    Replays the original decision (looked up by request id) as the
    ``decision_id`` lineage anchor, runs the outcome through an ephemeral swarm
    that shares the in-memory supplier store, and returns the updated supplier
    performance. No external database is written; memory is the module-level
    :data:`swarm.memory.default_store`.
    """
    state = _swarm_state(request_id)
    decision = state.get_artifact("decision")
    if decision is None:
        raise HTTPException(
            status_code=404,
            detail=f"No decision recorded for request_id {request_id}",
        )
    decision_id = str(decision.id)

    correlation_id = f"{request_id}-OUTCOME"
    swarm = build_procurement_swarm(
        request_id=request_id, goal=f"Record outcome for {request.supplier_id}",
        supplier_memory=supplier_memory,
    )
    await swarm.start()
    await swarm.send_message(
        RECORD_OUTCOME_INTENT,
        {
            "decision_id": decision_id,
            "supplier_id": request.supplier_id,
            "delivered_on_time": request.delivered_on_time,
            "quality_score": request.quality_score,
            "actual_price": request.actual_price,
            "carbon_score": request.carbon_score,
        },
        sender="user",
        correlation_id=correlation_id,
    )
    await swarm.shutdown()
    _remember(swarm.state)

    outcomes = swarm.state.find_artifacts(kind="procurement_outcome", correlation_id=correlation_id)
    performance = supplier_memory.get_supplier_performance(request.supplier_id)
    return {
        "request_id": request_id,
        "correlation_id": correlation_id,
        "decision_id": decision_id,
        "outcome": outcomes[0].data if outcomes else None,
        "supplier_performance": performance.to_summary() if performance else None,
    }


@app.get("/swarm/supplier/{supplier_id}/performance")
async def get_supplier_performance(supplier_id: str) -> dict[str, Any]:
    """Deterministic supplier performance summary from in-memory supplier memory."""
    performance = supplier_memory.get_supplier_performance(supplier_id)
    if performance is None:
        raise HTTPException(
            status_code=404,
            detail=f"No performance record for supplier {supplier_id}",
        )
    return {"supplier_id": supplier_id, "performance": performance.to_summary()}


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


@app.get("/swarm/risk/{request_id}")
async def get_swarm_risk(request_id: str) -> dict[str, Any]:
    """Deterministic risk assessment for the selected decision of a swarm run."""
    state = _swarm_state(request_id)
    risk = state.get_artifact(RISK_ASSESSMENT_ARTIFACT_NAME)
    if risk is None:
        raise HTTPException(
            status_code=404,
            detail=f"No risk assessment for request_id {request_id}",
        )
    return {"request_id": request_id, "risk": risk.data}


@app.get("/swarm/governance/{request_id}")
async def get_swarm_governance(request_id: str) -> dict[str, Any]:
    """Governance decision (approve / approval-required / reject) for a swarm run."""
    state = _swarm_state(request_id)
    governance = state.get_artifact(GOVERNANCE_DECISION_ARTIFACT_NAME)
    if governance is None:
        raise HTTPException(
            status_code=404,
            detail=f"No governance decision for request_id {request_id}",
        )
    return {"request_id": request_id, "governance": governance.data}


class ApproveRequest(BaseModel):
    """A simulated human approval resolution for a pending authorization."""

    approver: str = "governance_sim"


@app.post("/swarm/{request_id}/approve")
async def approve_swarm_decision(
    request_id: str, request: ApproveRequest
) -> dict[str, Any]:
    """Resolve a pending authorization (deterministic simulated human approval).

    Looks up the remembered swarm run, applies the :class:`ApprovalAgent` simulated
    approval to the pending ``ExecutionAuthorizationArtifact``, and returns the
    resulting authorization. Governance has already excluded rejected decisions,
    so a pending authorization is granted deterministically.
    """
    state = _swarm_state(request_id)
    agent = ApprovalAgent()
    authorization = agent.approve(state, approver=request.approver)
    if authorization is None:
        raise HTTPException(
            status_code=409,
            detail=f"No pending authorization to approve for request_id {request_id}",
        )
    return {
        "request_id": request_id,
        "authorization": authorization.data,
    }


@app.get("/swarm/authorization/{request_id}")
async def get_swarm_authorization(request_id: str) -> dict[str, Any]:
    """Execution authorization status for a swarm run (after governance+approval)."""
    state = _swarm_state(request_id)
    authorization = state.get_artifact(EXECUTION_AUTHORIZATION_ARTIFACT_NAME)
    if authorization is None:
        raise HTTPException(
            status_code=404,
            detail=f"No execution authorization for request_id {request_id}",
        )
    return {"request_id": request_id, "authorization": authorization.data}


class ExecuteRequest(BaseModel):
    """Optional overrides for resolving a pending authorization into an order."""

    approver: str = "governance_sim"


@app.post("/swarm/{request_id}/execute")
async def execute_swarm_decision(
    request_id: str, request: ExecuteRequest
) -> dict[str, Any]:
    """Create the purchase order (and track it) for an authorized swarm run.

    Resolves the remembered swarm run: if the authorization is still pending it
    is first granted deterministically (simulated approval), then a
    :class:`PurchaseOrderAgent` creates the order and an
    :class:`ExecutionTrackingAgent` records the realized execution status. The
    step is idempotent — a present order status is returned as-is. A rejected or
    absent authorization yields ``409 Conflict``.
    """
    state = _swarm_state(request_id)
    authorization = state.get_artifact(EXECUTION_AUTHORIZATION_ARTIFACT_NAME)
    if authorization is None:
        raise HTTPException(
            status_code=404,
            detail=f"No execution authorization for request_id {request_id}",
        )
    status = authorization.data.get("authorization_status")
    if status == "pending":
        ApprovalAgent().approve(state, approver=request.approver)
        authorization = state.get_artifact(EXECUTION_AUTHORIZATION_ARTIFACT_NAME)
        status = authorization.data.get("authorization_status")  # type: ignore[union-attr]
    if status != "authorized":
        raise HTTPException(
            status_code=409,
            detail=f"Execution blocked: authorization is {status!r}",
        )

    order_agent = PurchaseOrderAgent(base_connector=default_base_connector)
    order = order_agent.create_order(state)
    if order is None:
        raise HTTPException(
            status_code=409,
            detail=f"No purchase order created for request_id {request_id}",
        )
    exec_agent = ExecutionTrackingAgent(base_connector=default_base_connector)
    execution = exec_agent.track(state)
    return {
        "request_id": request_id,
        "purchase_order": order.data,
        "execution_status": execution.data if execution else None,
        "external_calls": [
            a.data
            for a in state.find_artifacts(kind=EXTERNAL_CALL_ARTIFACT_NAME)
        ],
    }


@app.get("/swarm/order/{request_id}")
async def get_swarm_order(request_id: str) -> dict[str, Any]:
    """Read-only purchase order for a swarm run (after execution)."""
    state = _swarm_state(request_id)
    order = state.get_artifact(PURCHASE_ORDER_ARTIFACT_NAME)
    if order is None:
        raise HTTPException(
            status_code=404,
            detail=f"No purchase order for request_id {request_id}",
        )
    return {"request_id": request_id, "purchase_order": order.data}


@app.get("/swarm/execution/{request_id}")
async def get_swarm_execution(request_id: str) -> dict[str, Any]:
    """Read-only execution status (purchase order lifecycle) for a swarm run."""
    state = _swarm_state(request_id)
    status = state.get_artifact(EXECUTION_STATUS_ARTIFACT_NAME)
    if status is None:
        raise HTTPException(
            status_code=404,
            detail=f"No execution status for request_id {request_id}",
        )
    return {"request_id": request_id, "execution_status": status.data}


@app.get("/swarm/external/{request_id}")
async def get_swarm_external_calls(request_id: str) -> dict[str, Any]:
    """Audit trail of outbound external-system calls for a swarm run.

    Each call is a recorded :class:`ExternalCallArtifact` ({system, action,
    request_payload, response_payload, status, idempotency_key, timestamp}).
    Returns ``404`` when no external interactions have been recorded.
    """
    state = _swarm_state(request_id)
    calls = state.find_artifacts(kind=EXTERNAL_CALL_ARTIFACT_NAME)
    if not calls:
        raise HTTPException(
            status_code=404,
            detail=f"No external calls recorded for request_id {request_id}",
        )
    return {
        "request_id": request_id,
        "external_calls": [artifact.data for artifact in calls],
    }


class SyncRequest(BaseModel):
    """Optional overrides for forcing an external reconciliation pass."""

    connector: str | None = None
    force: bool = False


@app.post("/swarm/{request_id}/sync")
async def sync_swarm_external(
    request_id: str, request: SyncRequest
) -> dict[str, Any]:
    """Trigger an external-system reconciliation for a swarm run.

    Re-runs the :class:`ExecutionTrackingAgent` against the remembered
    :class:`PurchaseOrderArtifact` using the configured base connector so the
    :class:`ExecutionStatusArtifact` reflects the latest external reality. The
    step is idempotent: the idempotency guard ensures the external ``status``
    call is not duplicated, and a present execution status is returned as-is
    when no re-fetch is requested.
    """
    state = _swarm_state(request_id)
    order = state.get_artifact(PURCHASE_ORDER_ARTIFACT_NAME)
    if order is None:
        raise HTTPException(
            status_code=404,
            detail=f"No purchase order for request_id {request_id}",
        )
    if request.connector:
        try:
            connector = build_connector(
                ConnectorConfig(provider=request.connector, mode="sandbox")
            )
        except ValueError as err:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported connector override {request.connector!r}; "
                    f"expected one of {list(PROVIDERS)}"
                ),
            ) from err
    else:
        connector = default_base_connector
    exec_agent = ExecutionTrackingAgent(base_connector=connector)
    execution = exec_agent.track(state)
    calls = state.find_artifacts(kind=EXTERNAL_CALL_ARTIFACT_NAME)
    return {
        "request_id": request_id,
        "purchase_order": order.data,
        "execution_status": execution.data if execution else None,
        "external_calls": [artifact.data for artifact in calls],
    }


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
