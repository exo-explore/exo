"""A tensor-parallel model on a ring keeps one core busy while it generates, and only then."""

import threading

import pytest

from exo.shared.models.model_cards import ModelId
from exo.shared.types.common import NodeId
from exo.shared.types.worker.instances import InstanceId, MlxJacclInstance
from exo.shared.types.worker.runners import RunnerId
from exo.shared.types.worker.shards import ShardMetadata, TensorShardMetadata
from exo.worker.runner.busy_core import busy_core_while_generating, wants_a_busy_core
from exo.worker.tests.unittests.conftest import (
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)

MODEL = ModelId("test-model")


def _tensor_shard(world_size: int) -> TensorShardMetadata:
    pipeline = get_pipeline_shard_metadata(MODEL, device_rank=0)
    return TensorShardMetadata(
        model_card=pipeline.model_card,
        device_rank=0,
        world_size=world_size,
        start_layer=pipeline.start_layer,
        end_layer=pipeline.end_layer,
        n_layers=pipeline.n_layers,
    )


def _ring(shard: ShardMetadata):
    node, runner = NodeId("node"), RunnerId()
    return get_mlx_ring_instance(
        instance_id=InstanceId(),
        model_id=MODEL,
        node_to_runner={node: runner},
        runner_to_shard={runner: shard},
    )


def _busy_threads() -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name == "busy-core"]


def test_only_a_tensor_parallel_model_across_devices_on_a_ring_wants_one() -> None:
    tensor = _tensor_shard(world_size=2)
    assert wants_a_busy_core(_ring(tensor), tensor)
    alone = _tensor_shard(world_size=1)
    assert not wants_a_busy_core(_ring(alone), alone)
    pipeline = get_pipeline_shard_metadata(MODEL, device_rank=0)
    assert not wants_a_busy_core(_ring(pipeline), pipeline)
    ring = _ring(tensor)
    rdma = MlxJacclInstance(
        instance_id=ring.instance_id,
        shard_assignments=ring.shard_assignments,
        jaccl_devices=[],
        jaccl_coordinators={},
    )
    assert not wants_a_busy_core(rdma, tensor)


def test_the_core_is_busy_only_while_generating() -> None:
    shard = _tensor_shard(world_size=2)
    instance = _ring(shard)
    assert not _busy_threads()
    with busy_core_while_generating(instance, shard):
        assert len(_busy_threads()) == 1
    assert not _busy_threads()


def test_a_failure_while_generating_stops_it_too() -> None:
    shard = _tensor_shard(world_size=2)
    with (
        pytest.raises(ValueError, match="failed"),
        busy_core_while_generating(_ring(shard), shard),
    ):
        raise ValueError("failed")
    assert not _busy_threads()


def test_nothing_runs_for_a_model_that_doesnt_want_one() -> None:
    shard = get_pipeline_shard_metadata(MODEL, device_rank=0)
    with busy_core_while_generating(_ring(shard), shard):
        assert not _busy_threads()
