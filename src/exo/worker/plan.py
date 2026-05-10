# pyright: reportUnusedImport = false

from collections.abc import Mapping, Sequence

from exo.shared.types.chunks import InputImageChunk
from exo.shared.types.common import CommandId, ModelId, NodeId
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
)
from exo.shared.types.text_generation import Base64Image, Base64ImageHash
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
    RunnerStatus,
    RunnerWarmingUp,
)
from exo.utils.keyed_backoff import KeyedBackoff
from exo.worker.runner.supervisor import RunnerSupervisor


def plan(
    node_id: NodeId,
    # Runners is expected to be FRESH and so should not come from state
    runners: Mapping[RunnerId, RunnerSupervisor],
    global_download_status: Mapping[NodeId, Sequence[DownloadProgress]],
    instances: Mapping[InstanceId, Instance],
    all_runners: Mapping[RunnerId, RunnerStatus],  # all global
    tasks: Mapping[TaskId, Task],
    input_chunk_buffer: Mapping[CommandId, Mapping[int, InputImageChunk]],
    image_cache: Mapping[Base64ImageHash, Base64Image],
    instance_backoff: KeyedBackoff[InstanceId],
    download_backoff: KeyedBackoff[ModelId],
) -> Task | None:
    # Python short circuiting OR logic should evaluate these sequentially.
    return (
        _cancel_tasks(runners, tasks)
        or _kill_runner(runners, all_runners, instances)
        or _create_runner(node_id, runners, all_runners, instances, instance_backoff)
        or _model_needs_download(
            node_id, runners, global_download_status, download_backoff
        )
        or _init_distributed_backend(runners, all_runners)
        or _load_model(runners, all_runners, global_download_status)
        or _ready_to_warmup(runners, all_runners)
        or _pending_tasks(runners, tasks, all_runners, input_chunk_buffer, image_cache)
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
        if isinstance(runner.status, RunnerFailed):
            return Shutdown(
                instance_id=runner.bound_instance.instance.instance_id,
                runner_id=runner_id,
            )

        # Restart-cascade rule: only fires when our local rank is
        # ``RunnerRunning`` (mid-task), which guarantees we previously
        # cleared the bootstrap collective with every peer rank in lock-
        # step (warmup-complete on all ranks is a precondition for
        # ``RunnerRunning`` -- see ``handle_generation_tasks``). If a
        # peer is now ``RunnerIdle``, that is a backward jump only
        # reachable by a process restart; the transient ``RunnerFailed``
        # was gossiped too briefly for the rule above to fire (the
        # supervisor respawned the runner immediately and the new
        # process emitted ``RunnerIdle`` right away). Without this rule
        # the bootstrap predicate (``all_runners_connecting`` in
        # ``_init_distributed_backend``) never fires and the respawned
        # peer is stuck in ``RunnerIdle`` forever -- the failure mode
        # observed in the K=8 sweep regression at 14:35:05.
        #
        # We restrict the trigger to ``RunnerRunning`` (not
        # ``RunnerLoaded``/``RunnerReady``) because during initial
        # bootstrap a peer can legitimately sit at ``RunnerIdle`` while
        # we have completed our own loading -- ``LoadModel`` happens
        # per-rank without a collective barrier (see ``runner.py``
        # case ``LoadModel``), so warmup-gate predicates need to keep
        # waiting rather than tearing the cluster down.
        instance = runner.bound_instance.instance
        # Use ``all_runner_ids`` (target + drafter) so the staleness
        # predicate fires for asymmetric placements where the drafter
        # is the only peer (single-target + drafter on a different
        # node).
        is_multi_rank_instance = len(instance.all_runner_ids) > 1
        local_is_running = isinstance(runner.status, RunnerRunning)

        for global_runner_id in instance.all_runner_ids:
            if runner_id == global_runner_id:
                continue

            peer_status = all_runners.get(global_runner_id, None)
            if isinstance(peer_status, RunnerFailed):
                return Shutdown(
                    instance_id=instance_id,
                    runner_id=runner_id,
                )

            if (
                is_multi_rank_instance
                and local_is_running
                and isinstance(peer_status, RunnerIdle)
            ):
                return Shutdown(
                    instance_id=instance_id,
                    runner_id=runner_id,
                )


def _create_runner(
    node_id: NodeId,
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
    instances: Mapping[InstanceId, Instance],
    instance_backoff: KeyedBackoff[InstanceId],
) -> CreateRunner | None:
    for instance in instances.values():
        # ``all_node_to_runner`` includes the asymmetric drafter rank
        # when ``instance.drafter_placement`` is set, so the drafter
        # node spawns its drafter runner the same way target nodes
        # spawn target runners.
        per_node_runners = instance.all_node_to_runner
        runner_id = per_node_runners.get(node_id, None)
        if runner_id is None:
            continue

        if runner_id in runners:
            continue

        # don't create runners if any other nodes have runners that have failed - wait for them to fix themselves first.
        instance_has_failed_runner = any(
            isinstance(all_runners.get(remote_runner_id), RunnerFailed)
            for remote_runner_id in per_node_runners.values()
            if remote_runner_id != runner_id
        )
        we_have_failed_before = isinstance(all_runners.get(runner_id), RunnerFailed)
        if instance_has_failed_runner and not we_have_failed_before:
            continue

        if not instance_backoff.should_proceed(instance.instance_id):
            continue

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
    download_backoff: KeyedBackoff[ModelId],
) -> DownloadModel | None:
    local_downloads = global_download_status.get(node_id, [])
    download_status = {
        dp.shard_metadata.model_card.model_id: dp for dp in local_downloads
    }

    for runner in runners.values():
        # The drafter rank loads its model from disk; placement assumes
        # the operator has pre-downloaded the drafter weights on the
        # eligible node. Auto-download for drafter ranks is a TODO --
        # for now, the drafter runner fails loudly at load time if the
        # weights are missing and the user fixes the cluster.
        if runner.bound_instance.is_drafter_rank:
            continue

        model_id = runner.bound_instance.bound_shard.model_card.model_id
        if (
            isinstance(runner.status, RunnerIdle)
            and (
                model_id not in download_status
                or not isinstance(
                    download_status[model_id],
                    (DownloadOngoing, DownloadCompleted, DownloadFailed),
                )
            )
            and download_backoff.should_proceed(model_id)
        ):
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
        runner_id = runner.bound_instance.bound_runner_id
        bound_instance = runner.bound_instance

        runner_is_idle = isinstance(runner.status, RunnerIdle)
        if not runner_is_idle:
            continue

        # Asymmetric drafter rank: dial-only, no ``mx.distributed`` init.
        # Dispatch the ConnectToGroup task as soon as the drafter is
        # idle. ``dial_target`` retries with backoff so an early dial
        # before target rank 0 binds is recoverable. Decoupling the
        # drafter from the target's collective barrier is what lets a
        # multi-target asymmetric instance work without
        # ``Group.split``.
        if bound_instance.is_drafter_rank:
            return ConnectToGroup(instance_id=instance.instance_id)

        # Single-target symmetric: no mx.distributed group at all.
        # Single-target asymmetric *with* a drafter still needs the
        # target rank to enter ``ConnectToGroup`` so it can bind the
        # drafter listener. Differentiate via the placement.
        is_single_rank_target = instance.parent_group_size == 1
        if is_single_rank_target and instance.drafter_placement is None:
            continue

        # Target-only barrier: drafter ranks are dispatched in the
        # branch above and are NOT members of any ``mx.distributed``
        # group under the v3+ wire. Iterate ``shard_assignments`` so
        # we get the target ranks alone.
        target_runner_ids = list(instance.shard_assignments.runner_to_shard.keys())
        all_target_connecting = all(
            isinstance(
                all_runners.get(target_runner_id),
                (RunnerConnecting, RunnerIdle),
            )
            for target_runner_id in target_runner_ids
        )

        if not all_target_connecting:
            continue

        if is_single_rank_target:
            # Single target rank in asymmetric placement: it still has
            # to enter ConnectToGroup to bind the drafter listener and
            # accept the dial. No mx.distributed barrier to honour.
            return ConnectToGroup(instance_id=instance.instance_id)

        # Multi-target ranks: keep the original ordering -- earlier
        # ranks dispatch immediately, the last target rank dispatches
        # once every other target rank is already RunnerConnecting (or
        # later).
        parent_size = instance.parent_group_size  # target ranks only
        parent_rank = bound_instance.parent_rank
        assert parent_rank < parent_size
        assert parent_rank >= 0

        accepting_ranks = parent_rank < parent_size - 1

        connecting_rank_ready = parent_rank == parent_size - 1 and all(
            isinstance(all_runners.get(target_runner_id, None), RunnerConnecting)
            for target_runner_id in target_runner_ids
            if target_runner_id != runner_id
        )

        if not (accepting_ranks or connecting_rank_ready):
            continue

        return ConnectToGroup(instance_id=instance.instance_id)

    return None


def _load_model(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
    global_download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> LoadModel | None:
    for runner in runners.values():
        instance = runner.bound_instance.instance
        shard_assignments = instance.shard_assignments

        # Target shards must all be downloaded before any rank loads;
        # the drafter's pre-downloaded weights are the operator's
        # responsibility (see _model_needs_download), so we don't gate
        # on its DownloadCompleted entry here.
        all_local_downloads_complete = all(
            nid in global_download_status
            and any(
                isinstance(dp, DownloadCompleted)
                and dp.shard_metadata.model_card.model_id == shard_assignments.model_id
                for dp in global_download_status[nid]
            )
            for nid in shard_assignments.node_to_runner
        )
        if not all_local_downloads_complete:
            continue

        # Single-target SYMMETRIC instance: no mx.distributed group and
        # no drafter wire, so the runner can skip the ConnectToGroup
        # collective and go straight to LoadModel. Single-target
        # ASYMMETRIC (drafter on a different node) still has to enter
        # ConnectToGroup so target rank 0 can bind the drafter socket
        # listener; it falls through to the barrier check below.
        is_single_rank_target = instance.parent_group_size == 1
        is_symmetric_placement = instance.drafter_placement is None
        if (
            is_single_rank_target
            and is_symmetric_placement
            and isinstance(runner.status, RunnerIdle)
        ):
            return LoadModel(instance_id=instance.instance_id)

        is_runner_waiting = isinstance(runner.status, RunnerConnected)

        all_ready_for_model = all(
            isinstance(
                all_runners.get(global_runner_id, None),
                (RunnerConnected, RunnerLoading, RunnerLoaded),
            )
            for global_runner_id in instance.all_runner_ids
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
        runner_id = runner.bound_instance.bound_runner_id
        bound_instance = runner.bound_instance

        is_runner_loaded = isinstance(runner.status, RunnerLoaded)
        if not is_runner_loaded:
            continue

        # ``RunnerWarmingUp`` is the canonical "ready to run warmup" state
        # for an accepting rank, but a peer that has already advanced past
        # warmup (``RunnerReady``/``RunnerRunning``) is *strictly past*
        # the barrier we care about. Asymmetric drafter rank warmup is
        # near-instant (one forward pass) so it can race past
        # ``RunnerWarmingUp`` before the connecting rank's plan loop
        # observes it; without including the post-warmup states the
        # connecting rank stalls in ``RunnerLoaded`` forever.
        post_loaded_states = (
            RunnerWarmingUp,
            RunnerReady,
            RunnerRunning,
        )

        # Drafter rank: warmup is independent (one drafter forward) so
        # dispatch as soon as the drafter is RunnerLoaded.
        if bound_instance.is_drafter_rank:
            return StartWarmup(instance_id=instance.instance_id)

        # Target ranks: keep the rank-0-connector ordering across
        # target-only ranks. The drafter rank is excluded from this
        # barrier (its own warmup is independent).
        parent_rank = bound_instance.parent_rank
        parent_size = instance.parent_group_size  # target ranks only

        assert parent_rank < parent_size
        assert parent_rank >= 0

        target_runner_ids = list(instance.shard_assignments.runner_to_shard.keys())

        accepting_ranks_ready = parent_rank > 0 and all(
            isinstance(
                all_runners.get(target_runner_id, None),
                (RunnerLoaded, *post_loaded_states),
            )
            for target_runner_id in target_runner_ids
        )

        connecting_rank_ready = parent_rank == 0 and all(
            isinstance(all_runners.get(target_runner_id, None), post_loaded_states)
            for target_runner_id in target_runner_ids
            if target_runner_id != runner_id
        )

        if accepting_ranks_ready or connecting_rank_ready:
            return StartWarmup(instance_id=instance.instance_id)

    return None


def _pending_tasks(
    runners: Mapping[RunnerId, RunnerSupervisor],
    tasks: Mapping[TaskId, Task],
    all_runners: Mapping[RunnerId, RunnerStatus],
    input_chunk_buffer: Mapping[CommandId, Mapping[int, InputImageChunk]],
    image_cache: Mapping[Base64ImageHash, Base64Image],
) -> Task | None:
    for task in tasks.values():
        # for now, just forward chat completions
        # TODO(ciaran): do this better!
        if not isinstance(task, (TextGeneration, ImageGeneration, ImageEdits)):
            continue
        if task.task_status not in (TaskStatus.Pending, TaskStatus.Running):
            continue

        if isinstance(task, ImageEdits) and task.task_params.total_input_chunks > 0:
            received = len(input_chunk_buffer.get(task.command_id, {}))
            if received < task.task_params.total_input_chunks:
                continue  # Wait for all chunks to arrive

        if (
            isinstance(task, TextGeneration)
            and task.task_params.image_hashes
            and not all(
                h in image_cache for h in task.task_params.image_hashes.values()
            )
        ):
            continue  # Wait for all images to be assembled into the cache

        for runner in runners.values():
            if task.instance_id != runner.bound_instance.instance.instance_id:
                continue

            # the task status _should_ be set to completed by the LAST runner
            # it is currently set by the first
            # this is definitely a hack
            if task.task_id in runner.completed or task.task_id in runner.in_progress:
                continue

            if isinstance(runner.status, (RunnerReady, RunnerRunning)) and all(
                isinstance(all_runners[global_runner_id], (RunnerReady, RunnerRunning))
                for global_runner_id in runner.bound_instance.instance.all_runner_ids
            ):
                return task


def _cancel_tasks(
    runners: Mapping[RunnerId, RunnerSupervisor],
    tasks: Mapping[TaskId, Task],
) -> Task | None:
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
