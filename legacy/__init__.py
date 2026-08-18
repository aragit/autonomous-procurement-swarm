"""Legacy V1 code preserved for backward compatibility.

This package contains the deprecated V1 asyncio swarm runtime:
- :mod:`legacy.event` — original EventBus (replaced by mesh.channels)
- :mod:`legacy.coordinator` — original SwarmCoordinator (replaced by mesh.cluster)
- :mod:`legacy.api` — original V1 FastAPI server (replaced by api.v2)

All modules emit :class:`DeprecationWarning` on import. New code should use
the V2 mesh runtime in the ``mesh`` package and ``api.v2`` instead.
"""

from __future__ import annotations

import warnings

__all__ = ["event", "coordinator", "api"]

warnings.warn(
    "The 'legacy' package contains deprecated V1 code. "
    "Use the V2 mesh runtime (mesh package, api.v2) for new development.",
    DeprecationWarning,
    stacklevel=2,
)
