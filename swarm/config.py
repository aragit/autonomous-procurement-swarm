"""Production configuration for the procurement swarm (v0.9 Step 14).

Centralizes timeouts, idempotency keys, and LLM thresholds so they can be
overridden from environment variables without code changes.
"""

from __future__ import annotations

import os


def _get_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _get_float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


def _get_bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).lower() in ("1", "true", "yes")


# LLM thresholds
TRUST_THRESHOLD: float = _get_float("SWARM_TRUST_THRESHOLD", 0.7)
STABILITY_THRESHOLD: float = _get_float("SWARM_STABILITY_THRESHOLD", 0.5)
CONFIDENCE_THRESHOLD: float = _get_float("SWARM_CONFIDENCE_THRESHOLD", 0.6)

# Temporal stability tolerance band
STABILITY_TOLERANCE: float = _get_float("SWARM_STABILITY_TOLERANCE", 0.02)

# History limits
MAX_HISTORY_LENGTH: int = _get_int("SWARM_MAX_HISTORY_LENGTH", 5)

# Drift detection sensitivity
CONFIDENCE_DROP_THRESHOLD: float = _get_float("SWARM_CONFIDENCE_DROP_THRESHOLD", 0.15)
MIN_COMPLETIONS: int = _get_int("SWARM_MIN_COMPLETIONS", 2)

# Timeouts (seconds)
LLM_TIMEOUT_SECONDS: int = _get_int("SWARM_LLM_TIMEOUT_SECONDS", 30)
AGENT_TIMEOUT_SECONDS: int = _get_int("SWARM_AGENT_TIMEOUT_SECONDS", 60)
API_TIMEOUT_SECONDS: int = _get_int("SWARM_API_TIMEOUT_SECONDS", 10)

# Idempotency
IDEMPOTENCY_KEY_TTL_SECONDS: int = _get_int("SWARM_IDEMPOTENCY_KEY_TTL_SECONDS", 3600)

# FastAPI
API_HOST: str = os.getenv("SWARM_API_HOST", "0.0.0.0")
API_PORT: int = _get_int("SWARM_API_PORT", 8000)
API_RELOAD: bool = _get_bool("SWARM_API_RELOAD", False)

# Policy constraints (hard bounds on strategy weights)
POLICY_PRICE_MAX: float = _get_float("SWARM_POLICY_PRICE_MAX", 0.7)
POLICY_DELIVERY_MIN: float = _get_float("SWARM_POLICY_DELIVERY_MIN", 0.3)

# Persistent event store
DB_PATH: str = os.getenv("SWARM_DB_PATH", "swarm.db")

# Learning / feedback ingestion
ENABLE_LEARNING: bool = _get_bool("SWARM_ENABLE_LEARNING", True)

# Adaptive policy tuning
ADAPTIVE_K: float = _get_float("ADAPTIVE_K", 0.2)
MIN_FEEDBACK_SAMPLES: int = _get_int("MIN_FEEDBACK_SAMPLES", 5)

# Simulation / replay
SIMULATION_LIMIT: int = _get_int("SIMULATION_LIMIT", 100)

# Closed-loop policy learning (v1.1 Step 22)
# Minimum number of traces-with-feedback required before a candidate policy can
# be learned. Falls back to "insufficient data" when unmet.
MIN_TRACES_FOR_LEARNING: int = _get_int("MIN_TRACES_FOR_LEARNING", 25)

# Objective function: metric = SUCCESS_WEIGHT * success_rate + SCORE_WEIGHT * avg_score
OBJECTIVE_SUCCESS_WEIGHT: float = _get_float("OBJECTIVE_SUCCESS_WEIGHT", 0.7)
OBJECTIVE_SCORE_WEIGHT: float = _get_float("OBJECTIVE_SCORE_WEIGHT", 0.3)

# Promotion rule: a candidate must beat the active metric by at least this margin
# AND have a success_rate no worse than the active policy.
IMPROVEMENT_MARGIN: float = _get_float("IMPROVEMENT_MARGIN", 0.0)

# Hybrid objective weights (v1.1 Step 22): ground-truth feedback is the primary
# signal; decision stability regularizes to avoid chaotic policies.
#   metric = FEEDBACK_SUCCESS_WEIGHT * feedback_success_rate
#          + FEEDBACK_SCORE_WEIGHT   * feedback_outcome_score
#          + STABILITY_WEIGHT       * decision_stability
FEEDBACK_SUCCESS_WEIGHT: float = _get_float("FEEDBACK_SUCCESS_WEIGHT", 0.5)
FEEDBACK_SCORE_WEIGHT: float = _get_float("FEEDBACK_SCORE_WEIGHT", 0.3)
STABILITY_WEIGHT: float = _get_float("STABILITY_WEIGHT", 0.2)

# Max allowed parameter deviation from the base (active) policy during candidate
# generation. Prevents unbounded drift across learning cycles.
POLICY_MAX_PARAM_DELTA: float = _get_float("POLICY_MAX_PARAM_DELTA", 0.20)

# Grid search granularity (v1.1 Step 22)
GRID_STEP: float = _get_float("GRID_STEP", 0.05)
GRID_DELTA_MAX: float = _get_float("GRID_DELTA_MAX", 0.10)

#: Per-parameter perturbation deltas: {-0.10, -0.05, 0, +0.05, +0.10}.
GRID_DELTAS: list[float] = [-GRID_DELTA_MAX, -GRID_STEP, 0.0, GRID_STEP, GRID_DELTA_MAX]

# Policy safety clamping bounds
THRESHOLD_CLAMP_MIN: float = _get_float("THRESHOLD_CLAMP_MIN", 0.6)
THRESHOLD_CLAMP_MAX: float = _get_float("THRESHOLD_CLAMP_MAX", 0.9)
WEIGHT_CLAMP_MIN: float = _get_float("WEIGHT_CLAMP_MIN", 0.2)
WEIGHT_CLAMP_MAX: float = _get_float("WEIGHT_CLAMP_MAX", 0.8)

# Strategy space (v1.1 Step 23): finite, predefined set of decision strategies.
# Each strategy defines a deterministic weight transformation applied during
# evaluation. No dynamic strategy generation is allowed.
STRATEGY_TYPES: list[str] = ["balanced", "cost_optimized", "quality_first", "trust_weighted"]
DEFAULT_STRATEGY_TYPE: str = "balanced"

# Contextual routing (v1.1 Step 24): predefined condition space for routing rules.
# Each condition maps a context key + value to a strategy type.
# Rules are evaluated in order (first match wins), falling back to default.
ROUTING_CONDITION_KEYS: list[str] = ["budget_level", "urgency", "supplier_count"]
BUDGET_LEVELS: list[str] = ["low", "medium", "high"]
URGENCY_LEVELS: list[str] = ["low", "high"]
MAX_ROUTING_RULES: int = 3

# Contextual parameter adaptation (v1.1 Step 25): param overrides in routing rules.
# Each rule can specify up to MAX_PARAMS_PER_RULE parameter deltas.
MAX_PARAMS_PER_RULE: int = 2
#: Maximum absolute delta per parameter override (must be within POLICY_MAX_PARAM_DELTA).
PARAM_OVERRIDE_DELTA: float = 0.10
#: Maximum total absolute delta across all param overrides in a single rule.
PARAM_OVERRIDE_TOTAL_DELTA_CAP: float = 0.15
#: Hard cap on total candidates generated by generate_candidates.
MAX_CANDIDATES: int = 128
#: Deterministic seed for candidate sampling when cap is exceeded.
CANDIDATE_SAMPLING_SEED: int = 42
