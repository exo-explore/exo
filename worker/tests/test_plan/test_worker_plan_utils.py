from __future__ import annotations

from dataclasses import dataclass
from typing import List, NotRequired, Optional, TypedDict

from typing_extensions import Literal

from shared.models.model_cards import MODEL_CARDS, ModelCard
from shared.types.api import ChatCompletionMessage, ChatCompletionTaskParams
from shared.types.common import CommandId, NodeId
from shared.types.models import ModelId, ModelMetadata
from shared.types.state import State
from shared.types.tasks import ChatCompletionTask, TaskId, TaskStatus, TaskType
from shared.types.worker.common import InstanceId, NodeStatus, RunnerId
from shared.types.worker.downloads import DownloadOngoing, DownloadProgressData
from shared.types.worker.instances import Instance, InstanceStatus
from shared.types.worker.ops import RunnerOp
from shared.types.worker.runners import (
    DownloadingRunnerStatus,
    RunnerStatus,
    RunningRunnerStatus,
    ShardAssignments,
)
from shared.types.worker.shards import PipelineShardMetadata
from worker.tests.constants import COMMAND_1_ID, INSTANCE_1_ID, MODEL_A_ID


class RunnerSpecDict(TypedDict):
    """Type definition for runner specification dictionaries."""
    runner_id: RunnerId
    node_id: NodeId
    device_rank: int
    status: RunnerStatus
    downloaded: NotRequired[bool]  # defaults to True if not provided


class MessageDict(TypedDict):
    """Type definition for message dictionaries."""
    role: Literal["system", "user", "assistant", "developer", "tool", "function"]
    content: NotRequired[str | None]
    name: NotRequired[str | None]
    tool_calls: NotRequired[list[dict[str, str]] | None]
    tool_call_id: NotRequired[str | None]
    function_call: NotRequired[dict[str, str] | None]


class TaskSpecDict(TypedDict):
    """Type definition for task specification dictionaries."""
    task_id: TaskId
    instance_id: NotRequired[InstanceId]  # defaults to function parameter if not provided
    command_id: NotRequired[CommandId]  # defaults to COMMAND_1_ID if not provided  
    status: NotRequired[TaskStatus]  # defaults to TaskStatus.PENDING if not provided
    model: NotRequired[str]  # defaults to model_id if not provided
    messages: NotRequired[list[MessageDict]]  # defaults to [{'role': 'user', 'content': 'Hello, world!'}] if not provided


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



def make_instance(
    instance_id: InstanceId,
    runner_specs: list[tuple[RunnerId, NodeId, int, RunnerStatus]],
    model_id: ModelId = MODEL_A_ID,
    instance_status: InstanceStatus = InstanceStatus.ACTIVE,
) -> tuple[Instance, dict[RunnerId, RunnerStatus], dict[NodeId, NodeStatus]]:
    """Creates an instance with one or more runners."""
    runner_to_shard: dict[RunnerId, PipelineShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}
    world_size = len(runner_specs)

    for runner_id, node_id, device_rank, _ in runner_specs:
        shard_metadata = make_shard_metadata(
            device_rank,
            world_size,
            model_id
        )
        runner_to_shard[runner_id] = shard_metadata
        node_to_runner[node_id] = runner_id

    shard_assignments = ShardAssignments(
        model_id=model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )
    instance = Instance(
        instance_id=instance_id,
        instance_type=instance_status,
        shard_assignments=shard_assignments,
        hosts=[],
    )

    # Currently nodes are only ever idle - as if they were running we would be blocking - so we wouldn't be running plan()
    # node_statuses = {node_id: NodeStatus.Idle for _, node_id, _, _ in runner_specs}  
    node_statuses: dict[NodeId, NodeStatus] = {}
    for _runner_id, node_id, _, status in runner_specs:
        if isinstance(status, RunningRunnerStatus):
            node_statuses[node_id] = NodeStatus.Running
        else:
            node_statuses[node_id] = NodeStatus.Idle
    runner_statuses = {runner_id: status for runner_id, _, _, status in runner_specs}

    return instance, runner_statuses, node_statuses

def make_state(
    runner_specs_per_instance: dict[InstanceId, list[tuple[RunnerId, NodeId, int, RunnerStatus]]],
    tasks: dict[TaskId, ChatCompletionTask] | None = None,
    model_id: ModelId = MODEL_A_ID,
    instance_status: InstanceStatus = InstanceStatus.ACTIVE,
) -> State:
    """Builds a full State from runner specs per instance, tasks, and defaults."""
    if tasks is None:
        tasks = {}
    instances: dict[InstanceId, Instance] = {}
    all_runner_statuses: dict[RunnerId, RunnerStatus] = {}
    all_node_statuses: dict[NodeId, NodeStatus] = {}

    for inst_id, specs in runner_specs_per_instance.items():
        # Build per-instance data using make_instance
        instance, runner_statuses, node_statuses = make_instance(
            instance_id=inst_id,
            runner_specs=specs,
            model_id=model_id,
            instance_status=instance_status,
        )
        instances[inst_id] = instance
        all_runner_statuses.update(runner_statuses)
        all_node_statuses.update(node_statuses)

    return State(
        node_status=all_node_statuses,
        instances=instances,
        runners=all_runner_statuses,
        tasks=tasks,
    )

def make_test_case(
    description: str,
    runner_specs: list[RunnerSpecDict],
    tasks: list[TaskSpecDict] | None = None,
    expected_op: Optional[RunnerOp] = None,
    instance_id: InstanceId = INSTANCE_1_ID,
    instance_status: InstanceStatus = InstanceStatus.ACTIVE,
    model_id: ModelId = MODEL_A_ID,
    command_id: CommandId = COMMAND_1_ID,  # Default for tasks
) -> PlanTestCase:
    """Builds a PlanTestCase from high-level specs."""
    if tasks is None:
        tasks = []
    # Convert runner_specs to tuple format for make_instance
    specs_tuple = [
        (r['runner_id'], r['node_id'], r['device_rank'], r['status'])
        for r in runner_specs
    ]

    # Build state using make_state (wrap single instance)
    state_tasks: dict[TaskId, ChatCompletionTask] = {}
    for t in tasks:
        task = ChatCompletionTask(
            instance_id=instance_id,
            task_id=t['task_id'],
            command_id=t.get('command_id', command_id),
            task_type=TaskType.CHAT_COMPLETION,
            task_status=t.get('status', TaskStatus.PENDING),
            task_params=ChatCompletionTaskParams(
                model=t.get('model', str(model_id)),
                messages=[ChatCompletionMessage(**m) for m in t.get('messages', [{'role': 'user', 'content': 'Hello, world!'}])],
            ),
        )
        state_tasks[t['task_id']] = task

    state = make_state(
        runner_specs_per_instance={instance_id: specs_tuple},
        tasks=state_tasks,
        model_id=model_id,
        instance_status=instance_status,
    )

    # Build in_process_runners with downloaded (default True if missing)
    in_process_runners = [
        InProcessRunner(
            runner_id=r['runner_id'],
            instance_id=instance_id,
            model_id=model_id,
            status=r['status'],
            downloaded=r.get('downloaded', True),
            device_rank=r['device_rank'],
        ) for r in runner_specs
    ]

    return PlanTestCase(
        description=description,
        state=state,
        in_process_runners=in_process_runners,
        expected_op=expected_op,
    )