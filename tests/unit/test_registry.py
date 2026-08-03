"""Unit tests for the agent registry (Phase 1)."""

import pytest

from swarm import AgentRegistry, BaseAgent, Capability, Event, SwarmState


class StubAgent(BaseAgent):
    """Minimal agent used to exercise the registry."""

    def __init__(
        self,
        name: str,
        *,
        capabilities: list[str | Capability] | None = None,
    ) -> None:
        super().__init__(name=name, capabilities=capabilities)

    async def perceive(self, event: Event) -> None:
        pass

    async def reason(self, state: SwarmState) -> None:
        pass

    async def act(self, state: SwarmState) -> None:
        pass


def test_register_and_retrieve():
    registry = AgentRegistry()
    agent = StubAgent("supplier_agent")
    registry.register(agent)
    assert registry.get("supplier_agent") is agent
    assert registry.require("supplier_agent") is agent
    assert "supplier_agent" in registry
    assert len(registry) == 1
    assert registry.names() == ["supplier_agent"]
    assert registry.all() == [agent]


def test_list_agents_maps_name_to_instance():
    registry = AgentRegistry()
    a = StubAgent("requirement_agent")
    b = StubAgent("supplier_discovery")
    registry.register(a)
    registry.register(b)
    assert registry.list_agents() == {"requirement_agent": a, "supplier_discovery": b}
    registry.unregister("supplier_discovery")
    assert registry.list_agents() == {"requirement_agent": a}


def test_duplicate_registration_raises_value_error():
    registry = AgentRegistry()
    registry.register(StubAgent("dup"))
    with pytest.raises(ValueError):
        registry.register(StubAgent("dup"))


def test_require_missing_raises_key_error():
    registry = AgentRegistry()
    with pytest.raises(KeyError):
        registry.require("ghost")


def test_unregister_removes_agent():
    registry = AgentRegistry()
    agent = StubAgent("a")
    registry.register(agent)
    assert registry.unregister("a") is agent
    assert registry.get("a") is None
    assert registry.unregister("a") is None


def test_by_capability_filters_agents():
    registry = AgentRegistry()
    registry.register(StubAgent("a", capabilities=["pricing"]))
    registry.register(StubAgent("b", capabilities=["pricing", "logistics"]))
    registry.register(StubAgent("c", capabilities=["scoring"]))
    assert {agent.name for agent in registry.by_capability("pricing")} == {"a", "b"}
    assert {agent.name for agent in registry.by_capability("logistics")} == {"b"}
    assert registry.by_capability("nope") == []


def test_registry_ranks_capabilities_by_priority():
    registry = AgentRegistry()
    registry.register(
        StubAgent(
            "global",
            capabilities=[Capability(name="supplier.search", description="", priority=1)],
        )
    )
    registry.register(
        StubAgent(
            "eu",
            capabilities=[Capability(name="supplier.search", description="", priority=5)],
        )
    )
    registry.register(
        StubAgent("negotiator", capabilities=[Capability(name="negotiation", priority=9)])
    )

    ranked = registry.by_capability("supplier.search")
    assert [agent.name for agent in ranked] == ["eu", "global"]
    assert registry.best_for_capability("supplier.search").name == "eu"
    assert registry.best_for_capability("missing") is None


def test_registry_by_capability_keeps_registration_order_for_ties():
    registry = AgentRegistry()
    registry.register(
        StubAgent("first", capabilities=[Capability(name="quote", description="", priority=3)])
    )
    registry.register(
        StubAgent("second", capabilities=[Capability(name="quote", description="", priority=3)])
    )
    assert [agent.name for agent in registry.by_capability("quote")] == ["first", "second"]


def test_best_for_capability_prefers_tagged_specialist_when_tags_required():
    registry = AgentRegistry()
    generalist = StubAgent("generalist", capabilities=["supplier.evaluate"])
    generalist.tags = {}
    specialist = StubAgent("eu_specialist", capabilities=["supplier.evaluate"])
    specialist.tags = {"region": "EU"}
    registry.register(generalist)
    registry.register(specialist)

    assert registry.best_for_capability("supplier.evaluate").name == "generalist"
    assert registry.best_for_capability("supplier.evaluate", region="EU").name == "eu_specialist"


def test_best_for_capability_matches_prefix_of_agent_tag():
    registry = AgentRegistry()
    agent = StubAgent("eu_west", capabilities=["supplier.evaluate"])
    agent.tags = {"region": "EU-west-1"}
    registry.register(agent)

    assert registry.best_for_capability("supplier.evaluate", region="EU").name == "eu_west"


def test_best_for_capability_requires_all_tags():
    registry = AgentRegistry()
    eu_agent = StubAgent("eu", capabilities=["supplier.evaluate"])
    eu_agent.tags = {"region": "EU"}
    global_agent = StubAgent("global", capabilities=["supplier.evaluate"])
    global_agent.tags = {"region": "EU", "material": "aluminum"}
    registry.register(eu_agent)
    registry.register(global_agent)

    assert registry.best_for_capability("supplier.evaluate", region="EU").name == "eu"
    assert (
        registry.best_for_capability(
            "supplier.evaluate", region="EU", material="aluminum"
        ).name
        == "global"
    )


def test_best_for_capability_returns_none_when_no_tagged_match():
    registry = AgentRegistry()
    generalist = StubAgent("generalist", capabilities=["supplier.evaluate"])
    generalist.tags = {}
    registry.register(generalist)

    assert registry.best_for_capability("supplier.evaluate", region="APAC") is None
