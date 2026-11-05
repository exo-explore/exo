# pyright: reportUnusedImport = false

from collections.abc import Mapping, Sequence, Set

from exo.shared.types.common import ModelId, NodeId
from exo.shared.types.tasks import (
    ConnectToGroup,
    CreateRunner,
    DownloadModel,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
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
    RunnerShuttingDown,
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
    instance_backoff: KeyedBackoff[InstanceId],
    download_backoff: KeyedBackoff[ModelId],
    failed_download_backoff: KeyedBackoff[InstanceId] | None = None,
    # This node's runners whose process has exited, but whose last report (that they are
    # going) has yet to come back in `all_runners`
    exits_in_flight: Set[RunnerId] = frozenset(),
) -> Task | None:
    # A runner that is going (stopped, failed or exited) is left alone until its process has
    # exited and the worker forgets it; only then can it be replaced.
    live = {
        runner_id: runner
        for runner_id, runner in runners.items()
        if not runner.stopping
    }
    # A runner that has yet to pick up the last task sent to it still has the status from
    # before that task: starting the next phase from it would send the task again. Such a
    # runner sits out those steps until it does (or until it's found stuck and stopped), and
    # the other runners on this node carry on.
    settled = {
        runner_id: runner
        for runner_id, runner in live.items()
        if not runner.awaiting_acknowledgement
    }
    # Python short circuiting OR logic should evaluate these sequentially. Requests aren't
    # planned here: the API nodes hand them to the runners directly (exo.worker.serving).
    return (
        _kill_runner(live, all_runners, instances)
        or _create_runner(
            node_id, runners, all_runners, instances, instance_backoff, exits_in_flight
        )
        or _model_needs_download(
            node_id,
            live,
            global_download_status,
            download_backoff,
            failed_download_backoff or KeyedBackoff(),
        )
        or _init_distributed_backend(settled, all_runners)
        or _load_model(settled, all_runners, global_download_status)
        or _ready_to_warmup(settled, all_runners)
    )


# A runner that has joined, or is joining, its instance's other runners: it is tied to the
# processes it connects with, and it is no use once any of them is gone
_JOINED = (
    RunnerConnecting,
    RunnerConnected,
    RunnerLoading,
    RunnerLoaded,
    RunnerWarmingUp,
    RunnerReady,
    RunnerRunning,
)
# A runner on its way out: it isn't coming back, and is replaced by a new one
_GOING = (RunnerFailed, RunnerShuttingDown)


def joined_to(runner: RunnerSupervisor, peer: RunnerId) -> bool:
    """Whether `runner` has joined the other runner `peer` of its instance, so that it must go
    when `peer` goes."""
    return (
        not runner.stopping
        and isinstance(runner.status, _JOINED)
        and peer != runner.bound_instance.bound_runner_id
        and peer in runner.bound_instance.instance.shard_assignments.runner_to_shard
    )


def _kill_runner(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
    instances: Mapping[InstanceId, Instance],
) -> Shutdown | None:
    """Stop a runner whose instance is gone, or that has joined runners of its instance
    of which one is going, or was seen going (`peer_went`). A runner that failed stops
    itself. An idle runner is tied to no other: it waits for the others' replacements."""
    for runner in runners.values():
        runner_id = runner.bound_instance.bound_runner_id
        instance_id = runner.bound_instance.instance.instance_id
        if instance_id not in instances:
            return Shutdown(instance_id=instance_id, runner_id=runner_id)
        if runner.peer_went or any(
            joined_to(runner, peer) and isinstance(all_runners.get(peer), _GOING)
            for peer in runner.bound_instance.instance.shard_assignments.node_to_runner.values()
        ):
            return Shutdown(instance_id=instance_id, runner_id=runner_id)


def _create_runner(
    node_id: NodeId,
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
    instances: Mapping[InstanceId, Instance],
    instance_backoff: KeyedBackoff[InstanceId],
    exits_in_flight: Set[RunnerId],
) -> CreateRunner | None:
    for instance in instances.values():
        runner_id = instance.shard_assignments.node_to_runner.get(node_id, None)
        if runner_id is None:
            continue

        if runner_id in runners:
            continue

        # Until this node's last report on the runner it replaces is back in its copy of the
        # state, that copy can still hold what the old process last said (Idle, Connecting):
        # this plan, and the other nodes', would take it for the new runner's.
        if runner_id in exits_in_flight:
            continue

        # A runner of the instance that has joined others is tied to this runner's last
        # process, which is gone: wait until it is going too, so that the new runner can
        # never pair with it.
        if any(
            isinstance(all_runners.get(peer), _JOINED)
            for peer in instance.shard_assignments.node_to_runner.values()
            if peer != runner_id
        ):
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
    failed_download_backoff: KeyedBackoff[InstanceId],
) -> DownloadModel | None:
    """Ask for the model of each local runner that hasn't loaded it, until it is
    downloaded.

    That includes runners already connecting to the rest of their instance (a
    multi-node instance connects before its downloads finish). A download that
    isn't running (never started, stopped, or failed) is asked for again; a failed
    one with a slower backoff, kept per instance, so a new instance asks at once.
    Models no instance on this node needs are never asked for.
    """
    local_downloads = global_download_status.get(node_id, [])
    download_status = {
        dp.shard_metadata.model_card.model_id: dp for dp in local_downloads
    }

    for runner in runners.values():
        if not isinstance(
            runner.status, (RunnerIdle, RunnerConnecting, RunnerConnected)
        ):
            continue  # loading or past it: the model is on disk
        model_id = runner.bound_instance.bound_shard.model_card.model_id
        instance_id = runner.bound_instance.instance.instance_id
        status = download_status.get(model_id)
        if isinstance(status, (DownloadOngoing, DownloadCompleted)):
            continue
        if (
            failed_download_backoff.should_proceed(instance_id)
            if isinstance(status, DownloadFailed)
            else download_backoff.should_proceed(model_id)
        ):
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


def _load_model(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
    global_download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> LoadModel | None:
    for runner in runners.values():
        instance = runner.bound_instance.instance
        shard_assignments = instance.shard_assignments

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

        is_single_node_instance = len(instance.shard_assignments.runner_to_shard) == 1
        if is_single_node_instance and isinstance(runner.status, RunnerIdle):
            return LoadModel(instance_id=instance.instance_id)

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
