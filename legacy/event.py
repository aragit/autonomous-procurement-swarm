"""Legacy V1 EventBus — re-exported from :mod:`swarm.core.event`.

.. deprecated::
    Use :mod:`mesh.channels` and :mod:`mesh.blackboard` (V2 mesh runtime)
    instead. The EventBus and Event model have been superseded by the
    typed blackboard with capability-scoped ACLs.

This module re-exports the V1 implementation for backward compatibility.
New code should use the V2 mesh runtime.
"""

import warnings

warnings.warn(
    "legacy.event is deprecated. Use mesh.channels and mesh.blackboard "
    "(V2 mesh runtime) instead.",
    DeprecationWarning,
    stacklevel=2,
)

from swarm.core.event import (  # noqa: E402, F401
    ANY_EVENT,
    Event,
    EventBus,
    EventHandler,
    SwarmEventType,
)
