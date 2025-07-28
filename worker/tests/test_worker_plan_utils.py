from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, List, Optional

from shared.models.model_cards import MODEL_CARDS, ModelCard
from shared.types.common import CommandId, NodeId
from shared.types.models import ModelId, ModelMetadata
from shared.types.state import State
from shared.types.tasks import TaskId
from shared.types.worker.common import InstanceId, NodeStatus, RunnerId
from shared.types.worker.downloads import DownloadOngoing, DownloadProgressData
from shared.types.worker.instances import Instance, InstanceStatus
from shared.types.worker.ops import RunnerOp
from shared.types.worker.runners import (
    AssignedRunnerStatus,
    DownloadingRunnerStatus,
    RunnerStatus,
    ShardAssignments,
)
from shared.types.worker.shards import PipelineShardMetadata

NODE_A: Final[NodeId] = NodeId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
NODE_B: Final[NodeId] = NodeId("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")

# Define constant IDs for deterministic test cases
RUNNER_1_ID: Final[RunnerId] = RunnerId("cccccccc-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
INSTANCE_1_ID: Final[InstanceId] = InstanceId()
RUNNER_2_ID: Final[RunnerId] = RunnerId("dddddddd-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
INSTANCE_2_ID: Final[InstanceId] = InstanceId()
MODEL_A_ID: Final[ModelId] = 'mlx-community/Llama-3.2-1B-Instruct-4bit'
MODEL_B_ID: Final[ModelId] = 'mlx-community/Llama-3.2-1B-Instruct-4bit'
TASK_1_ID: Final[TaskId] = TaskId()
COMMAND_1_ID: Final[CommandId] = CommandId()

@dataclass(slots=True, frozen=True)
class InProcessRunner:
    """Minimal description of a runner's in-process state."""

    runner_id: RunnerId
    instance_id: InstanceId
    model_id: ModelId
    status: RunnerStatus
    downloaded: bool
    device_rank: int = 0


@dataclass(slots=True, frozen=True)
class PlanTestCase:
    """Table-driven description of an entire planning scenario."""

    description: str
    state: State
    in_process_runners: List[InProcessRunner]
    expected_op: Optional[RunnerOp]

    def id(self) -> str:  # noqa: D401
        return self.description.replace(" ", "_")


def make_shard_metadata(device_rank: int, world_size: int, model_id: ModelId = MODEL_A_ID) -> PipelineShardMetadata:
    """Create PipelineShardMetadata with proper layer assignments based on device_rank and world_size."""
    total_layers = world_size  # For simplicity in tests, total_layers = world_size
    
    if world_size == 1:
        start_layer = 0
        end_layer = 1
        n_layers = 1
    else:
        # For multi-device setup, each device gets one layer
        start_layer = device_rank
        end_layer = device_rank + 1
        n_layers = total_layers
    
    return PipelineShardMetadata(
        device_rank=device_rank,
        world_size=world_size,
        model_meta=make_model_meta(model_id),
        start_layer=start_layer,
        end_layer=end_layer,
        n_layers=n_layers,
    )


def make_downloading_status(node_id: NodeId) -> DownloadingRunnerStatus:
    """Factory for a *Downloading* status with placeholder progress."""
    return DownloadingRunnerStatus(
        download_progress=DownloadOngoing(
            node_id=node_id,
            download_progress=DownloadProgressData(total_bytes=1, downloaded_bytes=0),
        )
    )

def make_model_meta(
    model_id: str
) -> ModelMetadata:
    model_card: ModelCard
    for card in MODEL_CARDS.values():
        if card.model_id == model_id:
            model_card = card

            return ModelMetadata(
                model_id=model_id,
                pretty_name=model_card.model_id,
                storage_size_kilobytes=10**6,
                n_layers=16,
            )
    
    raise Exception(f'Unknown model_id passed: {model_id}')

    ## Alternatively, if we are ok for this method to be async:
    # await _get_model_meta(model_id)


def create_worker_state(
    *,
    node_id: NodeId,
    runner_configs: list[tuple[RunnerId, InstanceId, ModelId]],
    tmp_path: Path,
) -> State:
    """Create a test `State` based on a list of runner configurations."""
    instances: dict[InstanceId, Instance] = {}
    for runner_id, instance_id, model_id in runner_configs:
        model_path = tmp_path / f"model_for_runner_{runner_id}"
        model_path.mkdir(exist_ok=True, parents=True)

        shard_metadata = PipelineShardMetadata(
            device_rank=0,
            world_size=1,
            model_meta=make_model_meta(model_id),
            start_layer=0,
            end_layer=1,
            n_layers=1,
        )
        shard_assignments = ShardAssignments(
            model_id=model_id,
            runner_to_shard={runner_id: shard_metadata},
            node_to_runner={node_id: runner_id},
        )
        instance = Instance(
            instance_id=instance_id,
            instance_type=InstanceStatus.ACTIVE,
            shard_assignments=shard_assignments,
            hosts=[],
        )
        instances[instance_id] = instance

    return State(
        node_status={node_id: NodeStatus.Idle},
        instances=instances,
        runners={runner_id: AssignedRunnerStatus() for runner_id, _, _ in runner_configs},
        tasks={},
    )


def make_instance(
    instance_id: InstanceId,
    model_id: ModelId,
    tmp_path: Path,
    runner_specs: list[tuple[RunnerId, NodeId, int]],
) -> Instance:
    """Creates an instance with one or more runners."""
    runner_to_shard: dict[RunnerId, PipelineShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}
    world_size = len(runner_specs)

    for runner_id, node_id, device_rank in runner_specs:
        model_path = tmp_path / f"model_for_runner_{runner_id}"
        model_path.mkdir(exist_ok=True, parents=True)

        shard_metadata = PipelineShardMetadata(
            device_rank=device_rank,
            world_size=world_size,
            model_meta=make_model_meta(model_id),
            start_layer=0,
            end_layer=1,
            n_layers=1,
        )
        runner_to_shard[runner_id] = shard_metadata
        node_to_runner[node_id] = runner_id

    shard_assignments = ShardAssignments(
        model_id=model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )
    return Instance(
        instance_id=instance_id,
        instance_type=InstanceStatus.ACTIVE,
        shard_assignments=shard_assignments,
        hosts=[],
    )

### For worker plan tests