# pyright: reportUnusedImport = false

from collections.abc import Mapping, Sequence

from exo.shared.types.common import NodeId
from exo.shared.types.tasks import (
    ChatCompletion,
    CreateRunner,
    DownloadModel,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
)
from exo.shared.types.worker.downloads import DownloadCompleted, DownloadProgress
from exo.shared.types.worker.instances import BoundInstance, Instance, InstanceId
from exo.shared.types.worker.runners import (
    RunnerId,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerStatus,
    RunnerWaitingForModel,
    RunnerWarmingUp,
)
from exo.shared.types.worker.shards import ShardMetadata
from exo.worker.runner.runner_supervisor import RunnerSupervisor


def plan(
    node_id: NodeId,
    # Runners is expected to be FRESH and so should not come from state
    runners: Mapping[RunnerId, RunnerSupervisor],
    # DL_status is expected to be FRESH and so should not come from state
    download_status: Mapping[ShardMetadata, DownloadProgress],
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
        or _load_model(runners, all_runners, global_download_status)
        or _ready_to_warmup(runners, all_runners)
        or _pending_tasks(runners, tasks, all_runners)
    )


def _kill_runner(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
    instances: Mapping[InstanceId, Instance],
) -> Shutdown | None:
    for runner in runners.values():
        if (instance_id := runner.bound_instance.instance.instance_id) not in instances:
            return Shutdown(
                instance_id=instance_id, runner_id=runner.bound_instance.bound_runner_id
            )

        """ --- Potential code to kill a runner if any runners in its instance have failed ---
        global_runners_in_instance = runner.bound_instance.instance.shard_assignments.node_to_runner.values()
        if any(isinstance(all_runners[runner_id], RunnerFailed) for runner_id in global_runners_in_instance if runner_id != runner.bound_instance.bound_runner_id):
            Shutdown(instance_id=runner.bound_instance.instance.instance_id, runner_id=runner.bound_instance.bound_runner_id)
        """


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
            bound_instance=BoundInstance(instance=instance, bound_runner_id=runner_id),
        )


def _model_needs_download(
    runners: Mapping[RunnerId, RunnerSupervisor],
    download_status: Mapping[ShardMetadata, DownloadProgress],
) -> DownloadModel | None:
    for runner in runners.values():
        if (
            isinstance(runner.status, RunnerWaitingForModel)
            and runner.bound_instance.bound_shard() not in download_status
        ):
            # We don't invalidate download_status randomly in case a file gets deleted on disk
            return DownloadModel(
                instance_id=runner.bound_instance.instance.instance_id,
                shard_metadata=runner.bound_instance.bound_shard(),
            )


""" --- TODO!
def _init_backend(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
) -> LoadModel | None:
    for runner in runner.values()
    pass
"""


def _load_model(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
    global_download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> LoadModel | None:
    for runner in runners.values():
        if (
            all(
                isinstance(dp, DownloadCompleted)
                if dp.shard_metadata
                == runner.bound_instance.instance.shard_assignments.runner_to_shard[rid]
                else True
                for nid, rid in runner.bound_instance.instance.shard_assignments.node_to_runner.items()
                for dp in global_download_status[nid]
            )
            and isinstance(runner.status, RunnerWaitingForModel)
            and all(
                isinstance(
                    all_runners.get(global_runner_id, None),
                    (RunnerWaitingForModel, RunnerLoading, RunnerLoaded),
                )
                for global_runner_id in runner.bound_instance.instance.shard_assignments.runner_to_shard
            )
        ):
            return LoadModel(instance_id=runner.bound_instance.instance.instance_id)


def _ready_to_warmup(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
) -> StartWarmup | None:
    for runner in runners.values():
        if isinstance(runner.status, RunnerLoaded) and (
            (
                all(
                    isinstance(
                        all_runners.get(global_runner_id, None),
                        (RunnerLoaded, RunnerWarmingUp),
                    )
                    for global_runner_id in runner.bound_instance.instance.shard_assignments.runner_to_shard
                )
                and runner.bound_instance.bound_shard().device_rank != 0
            )
            or (
                all(
                    isinstance(
                        all_runners.get(global_runner_id, None), (RunnerWarmingUp)
                    )
                    for global_runner_id in runner.bound_instance.instance.shard_assignments.runner_to_shard
                    if global_runner_id != runner.bound_instance.bound_runner_id
                )
                and runner.bound_instance.bound_shard().device_rank == 0
            )
        ):
            return StartWarmup(instance_id=runner.bound_instance.instance.instance_id)


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
