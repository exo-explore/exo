import exo.worker.plan as plan_mod
from exo.shared.types.common import ModelId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import LoadModel
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
    DownloadProgress,
    DownloadProgressData,
)
from exo.shared.types.worker.instances import BoundInstance, Instance, InstanceId
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerIdle,
)
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.keyed_backoff import KeyedBackoff
from exo.worker.tests.constants import (
    INSTANCE_1_ID,
    MODEL_A_ID,
    NODE_A,
    NODE_B,
    RUNNER_1_ID,
    RUNNER_2_ID,
)
from exo.worker.tests.unittests.conftest import (
    FakeRunnerSupervisor,
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)


def test_plan_requests_download_when_waiting_and_shard_not_downloaded():
    """
    When a runner is waiting for a model and its shard is not in the
    local download_status map, plan() should emit DownloadModel.
    """

    shard = get_pipeline_shard_metadata(model_id=MODEL_A_ID, device_rank=0)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID},
        runner_to_shard={RUNNER_1_ID: shard},
    )
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    runner = FakeRunnerSupervisor(bound_instance=bound_instance, status=RunnerIdle())

    runners = {RUNNER_1_ID: runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {RUNNER_1_ID: RunnerIdle()}

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert isinstance(result, plan_mod.DownloadModel)
    assert result.instance_id == INSTANCE_1_ID
    assert result.shard_metadata == shard


def test_plan_loads_model_when_all_shards_downloaded_and_waiting():
    """
    When all shards for an instance are DownloadCompleted (globally) and
    all runners are in waiting/loading/loaded states, plan() should emit
    LoadModel once.
    """
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard2 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: shard1, RUNNER_2_ID: shard2},
    )
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerConnected()
    )

    runners = {RUNNER_1_ID: local_runner}
    instances = {INSTANCE_1_ID: instance}

    all_runners = {
        RUNNER_1_ID: RunnerConnected(),
        RUNNER_2_ID: RunnerConnected(),
    }

    global_download_status = {
        NODE_A: [
            DownloadCompleted(shard_metadata=shard1, node_id=NODE_A, total=Memory())
        ],
        NODE_B: [
            DownloadCompleted(shard_metadata=shard2, node_id=NODE_B, total=Memory())
        ],
    }

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        global_download_status=global_download_status,
        instances=instances,
        all_runners=all_runners,
        tasks={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert isinstance(result, LoadModel)
    assert result.instance_id == INSTANCE_1_ID


def test_plan_does_not_request_download_when_shard_already_downloaded():
    """
    If the local shard already has a DownloadCompleted entry, plan()
    should not re-emit DownloadModel while global state is still catching up.
    """
    shard = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID},
        runner_to_shard={RUNNER_1_ID: shard},
    )
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    runner = FakeRunnerSupervisor(bound_instance=bound_instance, status=RunnerIdle())

    runners = {RUNNER_1_ID: runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {RUNNER_1_ID: RunnerIdle()}

    # Global state shows shard is downloaded for NODE_A
    global_download_status: dict[NodeId, list[DownloadProgress]] = {
        NODE_A: [
            DownloadCompleted(shard_metadata=shard, node_id=NODE_A, total=Memory())
        ],
        NODE_B: [],
    }

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        global_download_status=global_download_status,
        instances=instances,
        all_runners=all_runners,
        tasks={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert not isinstance(result, plan_mod.DownloadModel)


def test_plan_does_not_load_model_until_all_shards_downloaded_globally():
    """
    LoadModel should not be emitted while some shards are still missing from
    the global_download_status.
    """
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard2 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: shard1, RUNNER_2_ID: shard2},
    )

    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerConnected()
    )

    runners = {RUNNER_1_ID: local_runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {
        RUNNER_1_ID: RunnerConnected(),
        RUNNER_2_ID: RunnerConnected(),
    }

    global_download_status = {
        NODE_A: [
            DownloadCompleted(shard_metadata=shard1, node_id=NODE_A, total=Memory())
        ],
        NODE_B: [],  # NODE_B has no downloads completed yet
    }

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        global_download_status=global_download_status,
        instances=instances,
        all_runners=all_runners,
        tasks={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert result is None

    global_download_status = {
        NODE_A: [
            DownloadCompleted(shard_metadata=shard1, node_id=NODE_A, total=Memory())
        ],
        NODE_B: [
            DownloadCompleted(shard_metadata=shard2, node_id=NODE_B, total=Memory())
        ],  # NODE_B has no downloads completed yet
    }

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        global_download_status=global_download_status,
        instances=instances,
        all_runners=all_runners,
        tasks={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert result is not None


def _connected_runner_of_two_node_instance() -> tuple[
    FakeRunnerSupervisor, dict[InstanceId, Instance], ShardMetadata
]:
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard2 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: shard1, RUNNER_2_ID: shard2},
    )
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerConnected()
    )
    return runner, {INSTANCE_1_ID: instance}, shard1


def _plan_with(
    runner: FakeRunnerSupervisor,
    instances: dict[InstanceId, Instance],
    local: list[DownloadProgress],
    failed_backoff: KeyedBackoff[ModelId],
) -> object:
    return plan_mod.plan(
        node_id=NODE_A,
        runners={RUNNER_1_ID: runner},  # type: ignore
        global_download_status={NODE_A: local},
        instances=instances,
        all_runners={RUNNER_1_ID: RunnerConnected(), RUNNER_2_ID: RunnerConnected()},
        tasks={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
        failed_download_backoff=failed_backoff,
    )


def test_plan_asks_again_for_a_stopped_download_whatever_the_runner_is_doing():
    """A multi-node instance's runners connect before the downloads finish; a
    download that stops then must still be asked for again."""
    runner, instances, shard = _connected_runner_of_two_node_instance()
    stopped = DownloadPending(shard_metadata=shard, node_id=NODE_A)
    result = _plan_with(runner, instances, [stopped], KeyedBackoff())
    assert isinstance(result, plan_mod.DownloadModel)


def test_plan_asks_again_for_a_failed_download_with_its_own_backoff():
    runner, instances, shard = _connected_runner_of_two_node_instance()
    failed = DownloadFailed(shard_metadata=shard, node_id=NODE_A, error_message="x")
    failed_backoff: KeyedBackoff[ModelId] = KeyedBackoff(base=30.0, cap=300.0)
    assert isinstance(
        _plan_with(runner, instances, [failed], failed_backoff), plan_mod.DownloadModel
    )
    failed_backoff.record_attempt(MODEL_A_ID)
    assert _plan_with(runner, instances, [failed], failed_backoff) is None


def test_plan_leaves_a_running_download_alone():
    runner, instances, shard = _connected_runner_of_two_node_instance()
    running = DownloadOngoing(
        shard_metadata=shard,
        node_id=NODE_A,
        download_progress=DownloadProgressData(
            total=Memory.from_bytes(2),
            downloaded=Memory.from_bytes(1),
            downloaded_this_session=Memory.from_bytes(1),
            completed_files=0,
            total_files=1,
            speed=0,
            eta_ms=0,
            files={},
        ),
    )
    assert _plan_with(runner, instances, [running], KeyedBackoff()) is None


def test_plan_asks_for_nothing_no_local_runner_needs():
    _, _, shard = _connected_runner_of_two_node_instance()
    failed = DownloadFailed(shard_metadata=shard, node_id=NODE_A, error_message="x")
    result = plan_mod.plan(
        node_id=NODE_A,
        runners={},
        global_download_status={NODE_A: [failed]},
        instances={},
        all_runners={},
        tasks={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
        failed_download_backoff=KeyedBackoff(),
    )
    assert result is None
