"""End-to-end demo of the swarm runtime.

Flow: user request -> Swarm (public interface) -> requirement agent ->
RequirementCreated event -> supplier discovery agent -> suppliers recorded in
shared state.

The demo drives the swarm through the public ``Swarm`` facade, the way Phase 2
applications will. The internal ``SwarmCoordinator`` engine stays behind the
scenes.

Run from the repository root with:

    python -m examples.run_swarm_demo
"""

import asyncio

from examples.requirement_agent import SimpleRequirementAgent
from examples.supplier_discovery_agent import SimpleSupplierDiscoveryAgent
from swarm import Swarm, SwarmState
from swarm.core.logging import configure_logging


async def run() -> SwarmState:
    """Execute the demo flow and return the resulting shared state."""
    swarm = Swarm(
        request_id="REQ-001",
        goal="Source laptops for the engineering team",
    )
    requirement_agent = SimpleRequirementAgent(bus=swarm.bus)
    discovery_agent = SimpleSupplierDiscoveryAgent()

    swarm.register(requirement_agent)
    swarm.register(discovery_agent)
    await swarm.start()

    await swarm.send_message(
        "requirement.requested",
        {"text": "Find laptops for engineering team"},
        sender="user",
        correlation_id="REQ-001-CONV",
    )

    await swarm.shutdown()
    return swarm.state


def main() -> None:
    """Configure logging and run the demo."""
    configure_logging()
    state = asyncio.run(run())
    requirement = state.get_artifact("requirement")
    shortlist = state.get_artifact("suppliers_found")
    print(f"request_id: {state.request_id}")
    print(f"goal: {state.goal}")
    print(f"requirement artifact: {requirement.data if requirement else None}")
    print(f"supplier shortlist: {shortlist.data['candidates'] if shortlist else None}")
    print(f"event flow: {[event.type for event in state.events]}")
    print(f"correlation ids: {[event.correlation_id for event in state.events]}")


if __name__ == "__main__":
    main()
