"""v2 (Mesh Runtime) API package.

Exposes a standalone FastAPI ``app`` (runnable on its own port) and a ``router``
that the legacy control plane can mount under ``/v2``.

Usage::

    from api.v2 import app, router, set_runtime
    # inject a fake runtime in tests
    set_runtime(my_fake_runtime)
"""

from fastapi import FastAPI

from api.v2.router import router as router
from api.v2.runtime import (
    MeshRuntime,
    RayMeshRuntime,
    get_runtime,
    make_trace_id,
    set_runtime,
)

__all__ = [
    "router",
    "MeshRuntime",
    "RayMeshRuntime",
    "get_runtime",
    "set_runtime",
    "make_trace_id",
    "create_app",
    "app",
]


def create_app() -> FastAPI:
    """Build a standalone FastAPI app for the v2 mesh API."""
    app = FastAPI(
        title="Autonomous Procurement Swarm - Mesh Runtime API v2",
        version="2.0.0",
    )
    app.include_router(router)
    return app


app = create_app()
