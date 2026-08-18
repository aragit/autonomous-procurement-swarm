"""Legacy V1 FastAPI control plane — re-exported from the deprecated :mod:`api.main`.

.. deprecated::
    Use ``api.v2:app`` (V2 mesh runtime) for new development. This module
    provides backward-compatible imports for the V1 API surface (the ``app``
    FastAPI instance, route functions, and shared mutable state like
    ``swarm_states``).

The implementation lives in :mod:`api.main` (preserved for module-level state
compatibility with tests and downstream consumers). New code should use the V2
mesh runtime instead.
"""

import warnings

warnings.warn(
    "legacy.api (V1 control plane) is deprecated. Use api.v2:app (V2 mesh runtime) "
    "for new development.",
    DeprecationWarning,
    stacklevel=2,
)

from api.main import *  # noqa: E402, F401, F403

# Re-export private names used by tests and downstream code.
from api.main import (  # noqa: E402, F401
    _create_suppliers,
    _default_base_connector,
    _lookup_swarm_state_by_correlation_id,
    _remember,
    _swarm_state,
    app,
    default_base_connector,
    ledger,
    lifespan,
    shared_memory,
    shared_vector_store,
    supplier_memory,
    swarm_states,
)
