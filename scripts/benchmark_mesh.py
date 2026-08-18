#!/usr/bin/env python3
"""End-to-end benchmark for the V2 distributed neuro-symbolic procurement mesh.

This script is designed to run against a live Docker Compose mesh deployment:

    docker compose -f docker-compose.mesh.yml up --build -d
    python scripts/benchmark_mesh.py

It submits a requirement to ``POST /v2/procurement/run``, polls
``GET /v2/procurement/{trace_id}/status`` until the run completes, and
verifies the full signal path through the typed blackboard:

    REQUIREMENT → DISCOVERY → SCORE / RISK → DEAL → DECISION

Exit codes
----------
0  — all phases verified
1  — signal path mismatch or timeout
2  — unreachable API or malformed response
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Any

import httpx

API_BASE = os.environ.get("MESH_API_BASE_URL", "http://localhost:8000")
POLL_INTERVAL = float(os.environ.get("BENCHMARK_POLL_INTERVAL", "1.0"))
MAX_POLL_ATTEMPTS = int(os.environ.get("BENCHMARK_MAX_ATTEMPTS", "120"))

# Expected signal path through the typed blackboard.
EXPECTED_PATH = [
    ("requirement", "REQUIREMENT"),
    ("discovery", "DISCOVERY"),
    ("score", "SCORE"),
    ("risk", "RISK"),
    ("deal", "DEAL"),
    ("decision", "DECISION"),
]


def banner(title: str) -> None:
    print(f"\n{'━' * 60}")
    print(f"  {title}")
    print(f"{'━' * 60}")


async def check_health(client: httpx.AsyncClient) -> bool:
    """Probe the mesh health endpoint."""
    resp = await client.get(f"{API_BASE}/v2/procurement/health", timeout=10.0)
    if resp.status_code != 200:
        return False
    return resp.json().get("status") == "ready"


async def submit_requirement(
    client: httpx.AsyncClient,
    material: str,
    quantity: int,
    budget: float,
    lead_time: int,
    max_carbon: float | None = None,
) -> str:
    """Submit a requirement and return the trace_id."""
    payload: dict[str, Any] = {
        "material": material,
        "quantity": quantity,
        "budget": budget,
        "target_lead_time_days": lead_time,
    }
    if max_carbon is not None:
        payload["max_carbon_per_unit"] = max_carbon

    resp = await client.post(
        f"{API_BASE}/v2/procurement/run",
        json=payload,
        timeout=120.0,
    )
    if resp.status_code != 200:
        print(f"  ERROR: POST /v2/procurement/run → {resp.status_code}: {resp.text}")
        sys.exit(2)

    data = resp.json()
    trace_id = data["trace_id"]
    print(f"  Submitted requirement: trace_id={trace_id}")
    print(f"  Status: {data.get('status')}")
    if data.get("decision"):
        decision = data["decision"]
        print(
            f"  Decision: {decision.get('selected_supplier', 'N/A')} "
            f"(score={decision.get('composite_score', 'N/A')})"
        )
    return trace_id


async def poll_status(client: httpx.AsyncClient, trace_id: str) -> dict[str, Any]:
    """Poll the status endpoint until the run completes or times out."""
    url = f"{API_BASE}/v2/procurement/{trace_id}/status"
    for attempt in range(1, MAX_POLL_ATTEMPTS + 1):
        resp = await client.get(url, timeout=10.0)
        if resp.status_code == 404:
            print(f"  [{attempt}] 404 — run not found yet, retrying...")
            await asyncio.sleep(POLL_INTERVAL)
            continue
        if resp.status_code != 200:
            print(f"  ERROR: GET {url} → {resp.status_code}: {resp.text}")
            sys.exit(2)

        data = resp.json()
        status = data.get("status")
        channels = data.get("channels", {})

        print(
            f"  [{attempt}] status={status} "
            f"channels={{ {', '.join(f'{k}={v}' for k, v in channels.items())} }}"
        )

        if status == "completed":
            return data
        await asyncio.sleep(POLL_INTERVAL)

    print(f"  TIMEOUT: run did not complete after {MAX_POLL_ATTEMPTS} attempts")
    sys.exit(1)


def verify_signal_path(channels: dict[str, int]) -> bool:
    """Verify that every expected channel has at least one entry."""
    print("\n  Verifying signal path:")
    all_ok = True
    for chan_key, chan_name in EXPECTED_PATH:
        count = channels.get(chan_key, 0)
        if count > 0:
            print(f"    [PASS] {chan_name:<12} {count} entries")
        else:
            print(f"    [FAIL] {chan_name:<12} 0 entries — signal missing!")
            all_ok = False
    return all_ok


async def main() -> None:
    banner("V2 Mesh Benchmark — Neuro-Symbolic Procurement Signal Path")
    print(f"  API base:    {API_BASE}")
    print(f"  Poll interval: {POLL_INTERVAL}s")
    print(f"  Max attempts: {MAX_POLL_ATTEMPTS}")

    async with httpx.AsyncClient(timeout=120.0) as client:
        banner("Phase 0 — Health Check")
        if not await check_health(client):
            print("  FAIL: mesh health check did not return ready")
            sys.exit(2)
        print("  PASS: mesh is ready")

        banner("Phase 1 — Submit Requirement")
        trace_id = await submit_requirement(
            client,
            material="aluminum",
            quantity=1000,
            budget=500_000.0,
            lead_time=30,
            max_carbon=800.0,
        )

        banner("Phase 2 — Poll for Completion")
        result = await poll_status(client, trace_id)

        banner("Phase 3 — Signal Path Verification")
        channels = result.get("channels", {})
        if verify_signal_path(channels):
            banner("RESULT: PASS — Full signal path verified")
            sys.exit(0)
        else:
            banner("RESULT: FAIL — Signal path incomplete")
            sys.exit(1)


if __name__ == "__main__":
    start = time.monotonic()
    asyncio.run(main())
    elapsed = time.monotonic() - start
    print(f"\n  Total elapsed: {elapsed:.1f}s")
