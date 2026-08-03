"""Unit tests for the Artifact model and artifact-backed state (Phase 1.5)."""

from swarm import Artifact, SwarmState


def test_artifact_creation_defaults():
    artifact = Artifact(kind="requirement", name="requirement", data={"item": "laptops"})
    assert artifact.kind == "requirement"
    assert artifact.name == "requirement"
    assert artifact.data == {"item": "laptops"}
    assert artifact.tags == {}
    assert artifact.version == 1
    assert artifact.id
    assert artifact.created_at
    assert artifact.created_by == ""
    assert artifact.updated_by is None
    assert artifact.updated_at is None
    assert artifact.correlation_id is None


def test_artifact_update_creates_new_version():
    artifact = Artifact(kind="requirement", name="requirement", data={"qty": 1}, created_by="a")
    updated = artifact.update({"qty": 2}, by="b")

    assert updated.version == 2
    assert updated.data == {"qty": 2}
    assert updated.created_by == "a"
    assert updated.updated_by == "b"
    assert updated.updated_at is not None
    assert updated.name == "requirement"
    assert artifact.version == 1


def test_artifact_serialization_roundtrip():
    artifact = Artifact(
        kind="requirement",
        name="requirement",
        data={"item": "laptops"},
        created_by="requirement_agent",
        correlation_id="CONV-1",
    )
    restored = Artifact.model_validate(artifact.model_dump())
    assert restored == artifact
    assert restored.correlation_id == "CONV-1"


def test_state_put_and_get_latest_artifact():
    state = SwarmState()
    first = state.put_artifact(Artifact(kind="requirement", name="requirement", data={"qty": 1}))
    second = state.put_artifact(first.update({"qty": 2}))

    assert state.get_artifact("requirement") is second
    assert state.get_artifact("requirement").data == {"qty": 2}
    assert state.get_artifact("missing") is None


def test_state_artifacts_by_kind():
    state = SwarmState()
    state.put_artifact(Artifact(kind="requirement", name="requirement", data={}))
    state.put_artifact(Artifact(kind="bid", name="bid_a", data={}))
    state.put_artifact(Artifact(kind="bid", name="bid_b", data={}))

    bids = state.artifacts_by_kind("bid")
    assert [bid.name for bid in bids] == ["bid_a", "bid_b"]
    assert [a.kind for a in state.artifacts_by_kind("requirement")] == ["requirement"]


def test_state_artifact_serialization_roundtrip():
    state = SwarmState(request_id="REQ-1")
    state.put_artifact(
        Artifact(kind="requirement", name="requirement", data={"item": "laptops"})
    )
    restored = SwarmState.from_dict(state.to_dict())
    assert restored == state
    assert restored.get_artifact("requirement").data == {"item": "laptops"}


def test_find_artifacts_by_tags():
    state = SwarmState()
    state.put_artifact(
        Artifact(kind="quote", name="abc_eu", data={}, tags={"supplier": "abc", "region": "EU"})
    )
    state.put_artifact(
        Artifact(kind="quote", name="xyz_eu", data={}, tags={"supplier": "xyz", "region": "EU"})
    )
    state.put_artifact(
        Artifact(kind="quote", name="abc_us", data={}, tags={"supplier": "abc", "region": "US"})
    )

    abc = state.find_artifacts(tags={"supplier": "abc"})
    assert [artifact.name for artifact in abc] == ["abc_eu", "abc_us"]

    eu_quotes = state.find_artifacts(kind="quote", tags={"region": "EU"})
    assert {artifact.name for artifact in eu_quotes} == {"abc_eu", "xyz_eu"}

    assert state.find_artifacts(kind="quote", tags={"region": "ASIA"}) == []
    assert state.find_artifacts(name="abc_eu", tags={"region": "EU"})[0].name == "abc_eu"


def test_find_artifacts_without_filters_returns_all():
    state = SwarmState()
    state.put_artifact(Artifact(kind="quote", name="q1", data={}))
    state.put_artifact(Artifact(kind="bid", name="b1", data={}))
    assert {artifact.name for artifact in state.find_artifacts()} == {"q1", "b1"}
