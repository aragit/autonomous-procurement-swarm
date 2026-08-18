"""Container entrypoint for the Autonomous Procurement Swarm mesh.

A single container image supports three roles selected via ``MESH_ROLE``
environment variable or the ``--role`` CLI flag:

``head``
    Start (or join) a Ray cluster as the head node and then idle so the
    container stays alive for workers to connect.

``worker``
    Connect to an existing Ray cluster (``RAY_ADDRESS=auto``) and idle.
    The Ray autoscaler schedules tasks here.

``api``
    Run the v2 FastAPI server (``api.v2:app``) with uvicorn.

``legacy``
    Run the v1 FastAPI server (``api.main:app``) with uvicorn.

Usage::

    python -m mesh.entrypoint --role api
    python -m mesh.entrypoint --role head
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, NoReturn


def _role_from_env() -> str:
    return os.environ.get("MESH_ROLE", "api").lower()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mesh.entrypoint",
        description="Container entrypoint dispatcher for the procurement swarm.",
    )
    parser.add_argument(
        "--role",
        choices=["auto", "head", "worker", "api", "legacy"],
        default="auto",
        help="'auto' reads MESH_ROLE (default: api).",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("HOST", "0.0.0.0"),
        help="Bind host for the API role.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8000")),
        help="Bind port for the API role.",
    )
    return parser.parse_args(argv)


def _run_head(host: str, port: int) -> NoReturn:
    """Start the Ray head node and block forever."""
    import ray

    ray.init(
        address="auto" if host else "auto",
        dashboard_host=host or "0.0.0.0",
        dashboard_port=8265,
        include_dashboard=True,
    )

    import time

    _print_ready("Ray head node is ready", extra={"host": host, "dashboard": 8265})
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        ray.shutdown()
        sys.exit(0)


def _run_worker() -> NoReturn:
    """Join an existing Ray cluster as a worker and block forever."""
    import ray

    ray.init(address="auto", ignore_reinit_error=True, include_dashboard=False)
    _print_ready("Ray worker joined cluster")

    import time

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        ray.shutdown()
        sys.exit(0)


def _run_api(app_module: str, host: str, port: int) -> None:
    """Run a uvicorn server for the given app module."""
    import uvicorn

    _print_ready(f"Starting FastAPI ({app_module})", extra={"host": host, "port": port})
    uvicorn.run(
        app_module,
        host=host,
        port=port,
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )


def _print_ready(msg: str, *, extra: dict[str, Any] | None = None) -> None:
    """Print a readiness banner (consumed by Docker / orchestrators)."""
    import json

    payload = {"status": "ready", "message": msg, **(extra or {})}
    print(json.dumps(payload), flush=True)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    role = _role_from_env() if args.role == "auto" else args.role

    if role == "head":
        _run_head(args.host, args.port)
    elif role == "worker":
        _run_worker()
    elif role == "api":
        _run_api("api.v2:app", args.host, args.port)
    elif role == "legacy":
        _run_api("api.main:app", args.host, args.port)
    else:
        print(f"Unknown role: {role}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
