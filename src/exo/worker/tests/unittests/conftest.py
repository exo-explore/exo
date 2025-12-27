from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from anyio import ClosedResourceError, WouldBlock

from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelId, ModelMetadata
from exo.shared.types.tasks import BaseTask
from exo.shared.types.worker.instances import (
    BoundInstance,
    Instance,
    InstanceId,
    MlxRingInstance,
)
from exo.shared.types.worker.runners import RunnerId, RunnerStatus, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata


# Synchronous trivial sender and receiver.
@dataclass
class _State[T]:
    buffer: list[T]
    closed: bool = False


class MockSender[T]:
    def __init__(self, _state: _State[T] | None = None):
        self._state = _state or _State(buffer=[])
        self._closed = False

    def send(self, item: T):
        if self._closed:
            raise ClosedResourceError
        self._state.buffer.append(item)

    def close(self):
        self._closed = True
        self._state.closed = True

    def join(self):
        pass

    def clone(self) -> MockSender[T]:
        if self._closed:
            raise ClosedResourceError
        return MockSender(_state=self._state)

    def clone_receiver(self) -> MockReceiver[T]:
        if self._closed:
            raise ClosedResourceError
        return MockReceiver(_state=self._state)


class MockReceiver[T]:
    def __init__(self, _state: _State[T] | None = None):
        self._state = _state or _State(buffer=[])
        self._closed = False

    def close(self):
        self._closed = True
        self._state.closed = True

    def join(self):
        pass

    def clone(self) -> MockReceiver[T]:
        if self._closed:
            raise ClosedResourceError
        return MockReceiver(_state=self._state)

    def clone_sender(self) -> MockSender[T]:
        if self._closed:
            raise ClosedResourceError
        return MockSender(_state=self._state)

    def receive_nowait(self) -> T:
        if self._state.buffer:
            return self._state.buffer.pop(0)
        raise WouldBlock

    def collect(self) -> list[T]:
        out: list[T] = []
        while True:
            try:
                out.append(self.receive_nowait())
            except WouldBlock:
                break
        return out

    async def receive_at_least(self, n: int) -> list[T]:
        raise NotImplementedError

    def __enter__(self):
        return self

    def __iter__(self) -> Iterator[T]:
        while True:
            try:
                yield self.receive_nowait()
            except WouldBlock:
                break


# Runner supervisor without multiprocessing logic.
@dataclass(frozen=True)
class FakeRunnerSupervisor:
    bound_instance: BoundInstance
    status: RunnerStatus


class OtherTask(BaseTask):
    pass


# TODO: Is this actually better than using Mock/Fake dataclasses?
#  e.g. commit d01cd292344df15759070966826a6c027945792b
def get_pipeline_shard_metadata(
    model_id: ModelId, device_rank: int, world_size: int = 1
) -> ShardMetadata:
    return PipelineShardMetadata(
        model_meta=ModelMetadata(
            model_id=model_id,
            pretty_name=str(model_id),
            storage_size=Memory.from_mb(100000),
            n_layers=32,
            hidden_size=2048,
            supports_tensor=False,
        ),
        device_rank=device_rank,
        world_size=world_size,
        start_layer=0,
        end_layer=32,
        n_layers=32,
    )


def get_shard_assignments(
    model_id: ModelId,
    node_to_runner: dict[NodeId, RunnerId],
    runner_to_shard: dict[RunnerId, ShardMetadata],
) -> ShardAssignments:
    return ShardAssignments(
        model_id=model_id,
        node_to_runner=node_to_runner,
        runner_to_shard=runner_to_shard,
    )


def get_mlx_ring_instance(
    instance_id: InstanceId,
    model_id: ModelId,
    node_to_runner: dict[NodeId, RunnerId],
    runner_to_shard: dict[RunnerId, ShardMetadata],
) -> Instance:
    return MlxRingInstance(
        instance_id=instance_id,
        shard_assignments=get_shard_assignments(
            model_id, node_to_runner, runner_to_shard
        ),
        hosts=[],
    )


def get_bound_mlx_ring_instance(
    instance_id: InstanceId, model_id: ModelId, runner_id: RunnerId, node_id: NodeId
) -> BoundInstance:
    shard = get_pipeline_shard_metadata(model_id=model_id, device_rank=0, world_size=1)
    instance = get_mlx_ring_instance(
        instance_id=instance_id,
        model_id=model_id,
        node_to_runner={node_id: runner_id},
        runner_to_shard={runner_id: shard},
    )
    return BoundInstance(
        instance=instance, bound_runner_id=runner_id, bound_node_id=node_id
    )
