from typing import Callable
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from exo.master import api as api_module
from exo.master.api import API
from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelId, ModelMetadata
from exo.shared.types.state import State
from exo.shared.types.topology import NodeInfo
from exo.shared.types.worker.instances import InstanceId, InstanceMeta, MlxRingInstance
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata, Sharding


class DummyAPIState:
    """Minimal stand-in for API instance used in placement tests.

    We only need the `state` attribute with `topology` and `instances` fields.
    """

    def __init__(self) -> None:
        self.state = State()


@pytest.mark.asyncio
async def test_get_placement_retries_with_allow_low_memory_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_placement should retry with allow_low_memory=True when the first attempt fails.

    The first call to get_instance_placements raises ValueError when allow_low_memory=False,
    and the second call with allow_low_memory=True succeeds.
    """

    # Arrange
    calls: list[bool] = []
    placements_result = {"instance-1": object()}

    async def fake_resolve_model_meta(model_id: str) -> ModelMetadata:  # type: ignore[override]
        return ModelMetadata(
            model_id=ModelId(model_id),
            storage_size=Memory.from_kb(1000),
            pretty_name="Test Model",
            n_layers=1,
        )

    def fake_get_instance_placements(command, *, topology, current_instances):  # type: ignore[override]
        # Record whether this call used allow_low_memory
        calls.append(command.allow_low_memory)
        if not command.allow_low_memory:
            raise ValueError("strict placement failed")
        return placements_result

    monkeypatch.setattr(api_module, "resolve_model_meta", fake_resolve_model_meta)
    monkeypatch.setattr(
        api_module, "get_instance_placements", fake_get_instance_placements
    )

    dummy_api = DummyAPIState()

    # Act
    result = await API.get_placement(
        dummy_api,  # type: ignore[arg-type]
        model_id="test-model",
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=1,
        allow_low_memory=False,
    )

    # Assert - first call with False, then retried with True
    assert calls == [False, True]
    assert result is placements_result["instance-1"]


@pytest.mark.asyncio
async def test_get_placement_raises_http_exception_when_both_strict_and_low_memory_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When both strict and low-memory placements fail, get_placement should raise HTTPException(400)."""

    async def fake_resolve_model_meta(model_id: str) -> ModelMetadata:  # type: ignore[override]
        return ModelMetadata(
            model_id=ModelId(model_id),
            storage_size=Memory.from_kb(1000),
            pretty_name="Test Model",
            n_layers=1,
        )

    def fake_get_instance_placements(command, *, topology, current_instances):  # type: ignore[override]
        raise ValueError("placement failed")

    monkeypatch.setattr(api_module, "resolve_model_meta", fake_resolve_model_meta)
    monkeypatch.setattr(
        api_module, "get_instance_placements", fake_get_instance_placements
    )

    dummy_api = DummyAPIState()

    # Act / Assert
    with pytest.raises(HTTPException) as exc_info:
        await API.get_placement(  # type: ignore[arg-type]
            dummy_api,
            model_id="test-model",
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=1,
            allow_low_memory=False,
        )

    assert exc_info.value.status_code == 400
    assert "placement failed" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_placement_previews_sets_is_low_memory_false_when_strict_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    create_node: Callable[[int, NodeId | None], NodeInfo],
) -> None:
    """If strict placement succeeds, is_low_memory should be False in the preview."""

    model_id = ModelId("test-model")
    metadata = ModelMetadata(
        model_id=model_id,
        storage_size=Memory.from_kb(1000),
        pretty_name="Test Model",
        n_layers=1,
    )

    card = SimpleNamespace(
        short_id=model_id,
        model_id=model_id,
        name="Test Model",
        description="",
        tags=[],
        metadata=metadata,
    )

    # Only one card in the MODEL_CARDS mapping for this test
    monkeypatch.setattr(api_module, "MODEL_CARDS", {model_id: card}, raising=False)

    # Create a topology with a single node so previews are attempted
    topo = Topology()
    node_id = NodeId()
    topo.add_node(create_node(2000 * 1024, node_id))
    state = State()
    state.topology = topo
    state.instances = {}

    dummy_api = SimpleNamespace(state=state)

    def fake_get_instance_placements(command, *, topology, current_instances):  # type: ignore[override]
        # Always succeed on the first (strict) attempt
        runner_id = RunnerId()
        instance = MlxRingInstance(
            instance_id=InstanceId(),
            shard_assignments=ShardAssignments(
                model_id=model_id,
                runner_to_shard={
                    runner_id: PipelineShardMetadata(
                        model_meta=metadata,
                        start_layer=0,
                        end_layer=1,
                        n_layers=1,
                        device_rank=0,
                        world_size=1,
                    )
                },
                node_to_runner={node_id: runner_id},
            ),
            hosts=[],
        )
        return {InstanceId(): instance}

    monkeypatch.setattr(
        api_module, "get_instance_placements", fake_get_instance_placements
    )

    # Act
    response = await API.get_placement_previews(  # type: ignore[arg-type]
        dummy_api,
        model_id=model_id,
    )

    # Assert - at least one successful preview, and all successful ones are not low-memory
    assert response.previews
    successful = [p for p in response.previews if p.error is None]
    assert successful
    assert all(not p.is_low_memory for p in successful)


@pytest.mark.asyncio
async def test_get_placement_previews_sets_is_low_memory_true_when_only_low_memory_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    create_node: Callable[[int, NodeId | None], NodeInfo],
) -> None:
    """If strict placement fails but low-memory placement succeeds, is_low_memory should be True."""

    model_id = ModelId("test-model-lowmem")
    metadata = ModelMetadata(
        model_id=model_id,
        storage_size=Memory.from_kb(1000),
        pretty_name="LowMem Model",
        n_layers=1,
    )

    card = SimpleNamespace(
        short_id=model_id,
        model_id=model_id,
        name="LowMem Model",
        description="",
        tags=[],
        metadata=metadata,
    )

    monkeypatch.setattr(api_module, "MODEL_CARDS", {model_id: card}, raising=False)

    topo = Topology()
    node_id = NodeId()
    topo.add_node(create_node(2000 * 1024, node_id))
    state = State()
    state.topology = topo
    state.instances = {}

    dummy_api = SimpleNamespace(state=state)

    def fake_get_instance_placements(command, *, topology, current_instances):  # type: ignore[override]
        # Fail when allow_low_memory is False, succeed when True
        if not command.allow_low_memory:
            raise ValueError("strict placement failed")
        runner_id = RunnerId()
        instance = MlxRingInstance(
            instance_id=InstanceId(),
            shard_assignments=ShardAssignments(
                model_id=model_id,
                runner_to_shard={
                    runner_id: PipelineShardMetadata(
                        model_meta=metadata,
                        start_layer=0,
                        end_layer=1,
                        n_layers=1,
                        device_rank=0,
                        world_size=1,
                    )
                },
                node_to_runner={node_id: runner_id},
            ),
            hosts=[],
        )
        return {InstanceId(): instance}

    monkeypatch.setattr(
        api_module, "get_instance_placements", fake_get_instance_placements
    )

    # Act
    response = await API.get_placement_previews(  # type: ignore[arg-type]
        dummy_api,
        model_id=model_id,
    )

    # Assert - at least one successful preview, and at least one is marked low-memory
    assert response.previews
    successful = [p for p in response.previews if p.error is None]
    assert successful
    assert any(p.is_low_memory for p in successful)
