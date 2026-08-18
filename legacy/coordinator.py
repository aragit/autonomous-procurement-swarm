"""Legacy V1 SwarmCoordinator — re-exported from :mod:`swarm.orchestration.coordinator`.

.. deprecated::
    Use :class:`mesh.cluster.ProcurementCluster` (V2 mesh runtime) instead.
    The SwarmCoordinator is preserved here for backward compatibility with
    existing tests and downstream consumers.
"""

import warnings

warnings.warn(
    "legacy.coordinator is deprecated. Use mesh.cluster.ProcurementCluster "
    "(V2 mesh runtime) instead.",
    DeprecationWarning,
    stacklevel=2,
)

from swarm.orchestration.coordinator import SwarmCoordinator  # noqa: E402, F401
