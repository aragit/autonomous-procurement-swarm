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
