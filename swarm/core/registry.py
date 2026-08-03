"""Agent registry for the swarm runtime."""

import structlog

from swarm.core.agent import BaseAgent

logger = structlog.get_logger(__name__)


class AgentRegistry:
    """Name-indexed registry of agents with capability lookup.

    Registration is explicit: registering a duplicate name raises
    :class:`ValueError` so naming mistakes surface immediately rather than
    silently shadowing an existing agent.
    """

    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        """Add ``agent`` to the registry under its unique name."""
        if agent.name in self._agents:
            raise ValueError(f"Agent '{agent.name}' is already registered")
        self._agents[agent.name] = agent
        logger.info("agent_registered", agent=agent.name)

    def unregister(self, name: str) -> BaseAgent | None:
        """Remove the agent with ``name`` and return it, or None if absent."""
        agent = self._agents.pop(name, None)
        if agent is not None:
            logger.info("agent_unregistered", agent=name)
        return agent

    def get(self, name: str) -> BaseAgent | None:
        return self._agents.get(name)

    def require(self, name: str) -> BaseAgent:
        """Like :meth:`get` but raises :class:`KeyError` when missing."""
        agent = self._agents.get(name)
        if agent is None:
            raise KeyError(f"Agent '{name}' is not registered")
        return agent

    def by_capability(self, capability: str) -> list[BaseAgent]:
        """Return agents advertising ``capability``, best match first.

        Matches are ranked by the capability's ``priority`` (higher first);
        ties keep registration order, so the result is deterministic.
        """
        matches = [agent for agent in self._agents.values() if agent.has_capability(capability)]
        return sorted(
            matches,
            key=lambda agent: max(
                (cap.priority for cap in agent.capabilities if cap.name == capability),
                default=0,
            ),
            reverse=True,
        )

    def best_for_capability(self, capability: str, **tags: str) -> BaseAgent | None:
        """Return the highest-priority agent for ``capability``, or None.

        ``tags`` are specialization filters, e.g. ``region="EU"``. An agent is
        eligible when it declares every required key and its value equals the
        requirement or is prefixed by it (so ``region="EU"`` matches an agent
        tagged ``region="EU-west"``). Generalist agents (no declared tags) are
        only eligible when no requirements are given, so specialization never
        wins by accident. Eligibility keeps ``by_capability``'s priority
        ordering, so among eligible agents the higher priority still wins.
        """
        matches = self.by_capability(capability)
        if not tags:
            return matches[0] if matches else None
        eligible = [
            agent
            for agent in matches
            if agent.tags
            and all(
                agent.tags.get(key) is not None
                and (agent.tags[key] == value or agent.tags[key].startswith(value))
                for key, value in tags.items()
            )
        ]
        return eligible[0] if eligible else None

    def list_agents(self) -> dict[str, BaseAgent]:
        """Snapshot of every registered agent, mapped by name.

        Enables dynamic discovery: callers can enumerate who is currently
        participating in the swarm without coupling to a concrete list.
        """
        return dict(self._agents)

    def all(self) -> list[BaseAgent]:
        return list(self._agents.values())

    def names(self) -> list[str]:
        return list(self._agents)

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, name: str) -> bool:
        return name in self._agents
