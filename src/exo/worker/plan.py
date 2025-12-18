# pyright: reportUnusedImport = false

from collections.abc import Mapping, Sequence

from exo.shared.types.common import NodeId
from exo.shared.types.models import ModelId
from exo.shared.types.tasks import (
    ChatCompletion,
    CreateRunner,
    DownloadModel,
    LoadModel,
    ConnectToGroup,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
)
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadOngoing,
    DownloadProgress,
)
from exo.shared.types.worker.instances import BoundInstance, Instance, InstanceId
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerIdle,
    RunnerConnecting,
    RunnerConnected,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerStatus,
    RunnerWarmingUp,
    ShardAssignments,
)
from exo.shared.types.worker.shards import ShardMetadata
from exo.worker.runner.runner_supervisor import RunnerSupervisor


def plan(
    node_id: NodeId,
    # Runners is expected to be FRESH and so should not come from state
    runners: Mapping[RunnerId, RunnerSupervisor],
    # DL_status is expected to be FRESH and so should not come from state
    download_status: Mapping[ModelId, DownloadProgress],
    # gdls is not expected to be fresh
    global_download_status: Mapping[NodeId, Sequence[DownloadProgress]],
    instances: Mapping[InstanceId, Instance],
    all_runners: Mapping[RunnerId, RunnerStatus],  # all global
    tasks: Mapping[TaskId, Task],
) -> Task | None:
    # Python short circuiting OR logic should evaluate these sequentially.
    return (
        _kill_runner(runners, all_runners, instances)
        or _create_runner(node_id, runners, instances)
        or _model_needs_download(runners, download_status)
        or _init_distributed_backend(runners, all_runners, global_download_status)
        or _load_model(runners, all_runners)
        or _ready_to_warmup(runners, all_runners)
        or _pending_tasks(runners, tasks, all_runners)
    )


def _kill_runner(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
    instances: Mapping[InstanceId, Instance],
) -> Shutdown | None:
    for runner in runners.values():
        runner_id = runner.bound_instance.bound_runner_id
        if (instance_id := runner.bound_instance.instance.instance_id) not in instances:
            return Shutdown(instance_id=instance_id, runner_id=runner_id)

        for (
            global_runner_id
        ) in runner.bound_instance.instance.shard_assignments.node_to_runner.values():
            if runner_id == global_runner_id:
                continue

            if isinstance(all_runners.get(global_runner_id, None), RunnerFailed):
                return Shutdown(
                    instance_id=instance_id,
                    runner_id=runner_id,
                )


def _create_runner(
    node_id: NodeId,
    runners: Mapping[RunnerId, RunnerSupervisor],
    instances: Mapping[InstanceId, Instance],
) -> CreateRunner | None:
    for instance in instances.values():
        runner_id = instance.shard_assignments.node_to_runner.get(node_id, None)
        if runner_id is None:
            continue

        if runner_id in runners:
            continue

        shard = instance.shard(runner_id)
        assert shard is not None

        return CreateRunner(
            instance_id=instance.instance_id,
            bound_instance=BoundInstance(
                instance=instance, bound_runner_id=runner_id, bound_node_id=node_id
            ),
        )


def _model_needs_download(
    runners: Mapping[RunnerId, RunnerSupervisor],
    download_status: Mapping[ModelId, DownloadProgress],
) -> DownloadModel | None:
    for runner in runners.values():
        if isinstance(runner.status, RunnerIdle) and (
            runner.bound_instance.bound_shard.model_meta.model_id not in download_status
            or not isinstance(
                download_status[runner.bound_instance.bound_shard.model_meta.model_id],
                (DownloadOngoing, DownloadCompleted),
            )
        ):
            # We don't invalidate download_status randomly in case a file gets deleted on disk
            return DownloadModel(
                instance_id=runner.bound_instance.instance.instance_id,
                shard_metadata=runner.bound_instance.bound_shard,
            )


def _init_distributed_backend(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
    global_download_status: Mapping[NodeId, Sequence[DownloadProgress]],
):
    for runner in runners.values():
        instance = runner.bound_instance.instance
        shard_assignments = instance.shard_assignments

        is_single_node_instance = len(shard_assignments.runner_to_shard) == 1
        if is_single_node_instance:
            continue

        all_local_downloads_complete = all(
            nid in global_download_status
            and any(
                isinstance(dp, DownloadCompleted)
                and dp.shard_metadata.model_meta.model_id == shard_assignments.model_id
                for dp in global_download_status[nid]
            )
            for nid in shard_assignments.node_to_runner.keys()
        )

        runner_is_idle = isinstance(runner.status, RunnerIdle)
        all_runners_connecting = all(
            isinstance(
                all_runners.get(global_runner_id),
                (RunnerConnecting, RunnerIdle),
            )
            for global_runner_id in shard_assignments.runner_to_shard
        )

        if not (
            all_local_downloads_complete and runner_is_idle and all_runners_connecting
        ):
            continue

        runner_id = runner.bound_instance.bound_runner_id

        shard = runner.bound_instance.bound_shard
        device_rank = shard.device_rank
        world_size = shard.world_size

        assert device_rank < world_size
        assert device_rank >= 0

        accepting_ranks = device_rank < world_size - 1

        # Rank = n-1
        connecting_rank_ready = device_rank == world_size - 1 and all(
            isinstance(all_runners.get(global_runner_id, None), RunnerConnecting)
            for global_runner_id in shard_assignments.runner_to_shard
            if global_runner_id != runner_id
        )

        if not (accepting_ranks or connecting_rank_ready):
            continue

        return ConnectToGroup(instance_id=instance.instance_id)

    return None


def _load_model(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
) -> LoadModel | None:
    for runner in runners.values():
        instance = runner.bound_instance.instance
        shard_assignments = instance.shard_assignments

        is_runner_waiting = isinstance(runner.status, RunnerConnected)

        all_ready_for_model = all(
            isinstance(
                all_runners.get(global_runner_id, None),
                (RunnerConnected, RunnerLoading, RunnerLoaded),
            )
            for global_runner_id in shard_assignments.runner_to_shard
        )

        if is_runner_waiting and all_ready_for_model:
            return LoadModel(instance_id=instance.instance_id)

    return None


def _ready_to_warmup(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
) -> StartWarmup | None:
    for runner in runners.values():
        instance = runner.bound_instance.instance
        shard_assignments = instance.shard_assignments
        shard = runner.bound_instance.bound_shard
        device_rank = shard.device_rank
        runner_id = runner.bound_instance.bound_runner_id
        world_size = shard.world_size

        is_runner_loaded = isinstance(runner.status, RunnerLoaded)

        assert device_rank < world_size
        assert device_rank >= 0

        # Rank != 0
        accepting_ranks_ready = device_rank > 0 and all(
            isinstance(
                all_runners.get(global_runner_id, None),
                (RunnerLoaded, RunnerWarmingUp),
            )
            for global_runner_id in shard_assignments.runner_to_shard
        )

        # Rank = 0
        connecting_rank_ready = device_rank == 0 and all(
            isinstance(all_runners.get(global_runner_id, None), RunnerWarmingUp)
            for global_runner_id in shard_assignments.runner_to_shard
            if global_runner_id != runner_id
        )

        if is_runner_loaded and (accepting_ranks_ready or connecting_rank_ready):
            return StartWarmup(instance_id=instance.instance_id)

    return None


def _pending_tasks(
    runners: Mapping[RunnerId, RunnerSupervisor],
    tasks: Mapping[TaskId, Task],
    all_runners: Mapping[RunnerId, RunnerStatus],
) -> Task | None:
    for task in tasks.values():
        # for now, just forward chat completions
        if not isinstance(task, ChatCompletion):
            continue
        if task.task_status not in (TaskStatus.Pending, TaskStatus.Running):
            continue

        for runner in runners.values():
            if task.instance_id != runner.bound_instance.instance.instance_id:
                continue

            if isinstance(runner.status, RunnerReady) and all(
                isinstance(all_runners[global_runner_id], (RunnerReady, RunnerRunning))
                for global_runner_id in runner.bound_instance.instance.shard_assignments.runner_to_shard
            ):
                return task
