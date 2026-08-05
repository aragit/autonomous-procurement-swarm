"""Replay and simulation engine for safe, deterministic intelligence evolution.

(v1.0 Step 21 — Replay + Simulation Engine)

Exposes :func:`replay_trace`, :func:`run_procurement`, :func:`compare_results`
and :func:`simulate_all_traces` to re-execute past procurement traces against
the *current* adaptive policy and compare the outcomes, without ever mutating
the production database.
"""

from swarm.simulation.replay_engine import (
    compare_results,
    extract_input,
    replay_trace,
    run_procurement,
    simulate_all_traces,
)

__all__ = [
    "compare_results",
    "extract_input",
    "replay_trace",
    "run_procurement",
    "simulate_all_traces",
]
