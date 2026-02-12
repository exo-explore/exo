# pyright: reportUnusedImport = false

from collections.abc import Mapping, Sequence

from exo.shared.models.model_cards import ModelId
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.tasks import (
    CancelTask,
    ConnectToGroup,
    CreateRunner,
    DownloadModel,
    ImageEdits,
    ImageGeneration,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
    TransferModelToDisk,
)
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadProgress,
)
from exo.shared.types.worker.instances import BoundInstance, Instance, InstanceId
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
    RunnerFailed,
    RunnerId,
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
    RunnerWarmingUp,
    ShardAssignments,
)
from exo.worker.runner.runner_supervisor import RunnerSupervisor


def plan(
    node_id: NodeId,
    # Runners is expected to be FRESH and so should not come from state
    runners: Mapping[RunnerId, RunnerSupervisor],
    global_download_status: Mapping[NodeId, Sequence[DownloadProgress]],
    instances: Mapping[InstanceId, Instance],
    all_runners: Mapping[RunnerId, RunnerStatus],  # all global
    tasks: Mapping[TaskId, Task],
    input_chunk_buffer: Mapping[CommandId, dict[int, str]] | None = None,
    input_chunk_counts: Mapping[CommandId, int] | None = None,
) -> Task | None:
    # Python short circuiting OR logic should evaluate these sequentially.
    return (
        _cancel_tasks(runners, tasks)
        or _kill_runner(runners, all_runners, instances)
        or _create_runner(node_id, runners, instances, all_runners)
        or _model_needs_download(node_id, runners, global_download_status)
        or _init_distributed_backend(runners, all_runners)
        or _transfer_model_to_disk(runners, all_runners)
        or _load_model(runners, all_runners, global_download_status)
        or _ready_to_warmup(runners, all_runners)
        or _pending_tasks(runners, tasks, all_runners, input_chunk_buffer or {})
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

        # Master removed our runner from state (retry signal) and process is dead
        if runner_id not in all_runners and isinstance(
            runner.status, (RunnerFailed, RunnerShutdown)
        ):
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
    all_runners: Mapping[RunnerId, RunnerStatus],
) -> CreateRunner | None:
    for instance in instances.values():
        runner_id = instance.shard_assignments.node_to_runner.get(node_id, None)
        if runner_id is None:
            continue

        if runner_id in runners:
            continue

        # Don't create while any peer runner is in a terminal state — wait for
        # the master to emit InstanceRetrying which removes them from state.
        has_terminal_peer = any(
            isinstance(all_runners.get(peer_rid), (RunnerFailed, RunnerShutdown))
            for peer_rid in instance.shard_assignments.node_to_runner.values()
            if peer_rid != runner_id
        )
        if has_terminal_peer:
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
    node_id: NodeId,
    runners: Mapping[RunnerId, RunnerSupervisor],
    global_download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> DownloadModel | None:
    local_downloads = global_download_status.get(node_id, [])
    download_status = {
        dp.shard_metadata.model_card.model_id: dp for dp in local_downloads
    }

    for runner in runners.values():
        # Transfer-only instances don't need downloads
        if runner.bound_instance.instance.shard_assignments.transfer_only:
            continue

        model_id = runner.bound_instance.bound_shard.model_card.model_id
        if isinstance(runner.status, RunnerIdle) and (
            model_id not in download_status
            or not isinstance(
                download_status[model_id],
                (DownloadOngoing, DownloadCompleted, DownloadFailed),
            )
        ):
            # For multi-node instances, skip download if a peer already has the model.
            # The model will be transferred via MLX distributed during LoadModel.
            instance = runner.bound_instance.instance
            is_multi_node = len(instance.shard_assignments.node_to_runner) > 1
            if is_multi_node and _any_peer_has_model(
                node_id, model_id, instance, global_download_status
            ):
                continue

            # We don't invalidate download_status randomly in case a file gets deleted on disk
            return DownloadModel(
                instance_id=runner.bound_instance.instance.instance_id,
                shard_metadata=runner.bound_instance.bound_shard,
            )


def _init_distributed_backend(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
):
    for runner in runners.values():
        instance = runner.bound_instance.instance
        shard_assignments = instance.shard_assignments

        is_single_node_instance = len(shard_assignments.runner_to_shard) == 1
        if is_single_node_instance:
            continue

        runner_is_idle = isinstance(runner.status, RunnerIdle)
        all_runners_connecting = all(
            isinstance(
                all_runners.get(global_runner_id),
                (RunnerConnecting, RunnerIdle),
            )
            for global_runner_id in shard_assignments.runner_to_shard
        )

        if not (runner_is_idle and all_runners_connecting):
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


def _transfer_model_to_disk(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
) -> TransferModelToDisk | None:
    """For transfer-only instances: after all ranks are connected, emit TransferModelToDisk."""
    for runner in runners.values():
        instance = runner.bound_instance.instance
        shard_assignments = instance.shard_assignments

        if not shard_assignments.transfer_only:
            continue

        is_runner_connected = isinstance(runner.status, RunnerConnected)
        all_connected_or_further = all(
            isinstance(
                all_runners.get(global_runner_id, None),
                (RunnerConnected, RunnerLoading, RunnerShuttingDown, RunnerShutdown),
            )
            for global_runner_id in shard_assignments.runner_to_shard
        )

        if is_runner_connected and all_connected_or_further:
            return TransferModelToDisk(
                instance_id=instance.instance_id,
                shard_metadata=runner.bound_instance.bound_shard,
            )

    return None


def _load_model(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
    global_download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> LoadModel | None:
    for runner in runners.values():
        instance = runner.bound_instance.instance
        shard_assignments = instance.shard_assignments

        # Transfer-only instances don't load models for inference
        if shard_assignments.transfer_only:
            continue

        is_single_node_instance = len(shard_assignments.runner_to_shard) == 1

        if is_single_node_instance:
            # Single-node: require local download complete
            if not _all_downloads_complete(shard_assignments, global_download_status):
                continue
            if isinstance(runner.status, RunnerIdle):
                return LoadModel(instance_id=instance.instance_id)
        else:
            # Multi-node: require at least one node to have the model downloaded.
            # Nodes without the model will receive it via MLX distributed transfer
            # during model loading.
            if not _any_download_complete(shard_assignments, global_download_status):
                continue

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


def _any_peer_has_model(
    node_id: NodeId,
    model_id: ModelId,
    instance: Instance,
    global_download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> bool:
    """Check if any other node in the instance already has the model downloaded."""
    for peer_nid in instance.shard_assignments.node_to_runner:
        if peer_nid == node_id:
            continue
        for dp in global_download_status.get(peer_nid, []):
            if (
                isinstance(dp, DownloadCompleted)
                and dp.shard_metadata.model_card.model_id == model_id
            ):
                return True
    return False


def _all_downloads_complete(
    shard_assignments: ShardAssignments,
    global_download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> bool:
    """Check if ALL nodes in the instance have completed downloading the model."""
    return all(
        nid in global_download_status
        and any(
            isinstance(dp, DownloadCompleted)
            and dp.shard_metadata.model_card.model_id == shard_assignments.model_id
            for dp in global_download_status[nid]
        )
        for nid in shard_assignments.node_to_runner
    )


def _any_download_complete(
    shard_assignments: ShardAssignments,
    global_download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> bool:
    """Check if at least one node in the instance has completed downloading the model."""
    return any(
        nid in global_download_status
        and any(
            isinstance(dp, DownloadCompleted)
            and dp.shard_metadata.model_card.model_id == shard_assignments.model_id
            for dp in global_download_status[nid]
        )
        for nid in shard_assignments.node_to_runner
    )


def _ready_to_warmup(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
) -> StartWarmup | None:
    for runner in runners.values():
        instance = runner.bound_instance.instance
        shard_assignments = instance.shard_assignments

        # Transfer-only instances don't go through warmup
        if shard_assignments.transfer_only:
            continue

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
    input_chunk_buffer: Mapping[CommandId, dict[int, str]],
) -> Task | None:
    for task in tasks.values():
        # for now, just forward chat completions
        # TODO(ciaran): do this better!
        if not isinstance(task, (TextGeneration, ImageGeneration, ImageEdits)):
            continue
        if task.task_status not in (TaskStatus.Pending, TaskStatus.Running):
            continue

        # For ImageEdits tasks, verify all input chunks have been received
        if isinstance(task, ImageEdits) and task.task_params.total_input_chunks > 0:
            cmd_id = task.command_id
            expected = task.task_params.total_input_chunks
            received = len(input_chunk_buffer.get(cmd_id, {}))
            if received < expected:
                continue  # Wait for all chunks to arrive

        for runner in runners.values():
            if task.instance_id != runner.bound_instance.instance.instance_id:
                continue

            # the task status _should_ be set to completed by the LAST runner
            # it is currently set by the first
            # this is definitely a hack
            if task.task_id in runner.completed:
                continue

            if isinstance(runner.status, RunnerReady) and all(
                isinstance(all_runners[global_runner_id], (RunnerReady, RunnerRunning))
                for global_runner_id in runner.bound_instance.instance.shard_assignments.runner_to_shard
            ):
                return task


def _cancel_tasks(
    runners: Mapping[RunnerId, RunnerSupervisor],
    tasks: Mapping[TaskId, Task],
) -> CancelTask | None:
    """Find a cancelled task that hasn't been sent to the runner yet."""
    for task in tasks.values():
        if task.task_status != TaskStatus.Cancelled:
            continue
        for runner_id, runner in runners.items():
            if task.instance_id != runner.bound_instance.instance.instance_id:
                continue
            if task.task_id in runner.cancelled:
                continue
            return CancelTask(
                instance_id=task.instance_id,
                cancelled_task_id=task.task_id,
                runner_id=runner_id,
            )
