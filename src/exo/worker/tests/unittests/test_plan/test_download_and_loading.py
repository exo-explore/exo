import exo.worker.plan as plan_mod
from exo.shared.types.common import NodeId
from exo.shared.types.tasks import LoadModel
from exo.shared.types.worker.downloads import DownloadCompleted, DownloadProgress
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runners import (
    RunnerWaitingForModel,
)
from exo.shared.types.worker.shards import ShardMetadata
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
    runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerWaitingForModel()
    )

    runners = {RUNNER_1_ID: runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {RUNNER_1_ID: RunnerWaitingForModel()}

    # No entry for this shard -> should trigger DownloadModel
    download_status: dict[ShardMetadata, DownloadProgress] = {}

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        download_status=download_status,
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
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
        bound_instance=bound_instance, status=RunnerWaitingForModel()
    )

    runners = {RUNNER_1_ID: local_runner}
    instances = {INSTANCE_1_ID: instance}

    all_runners = {
        RUNNER_1_ID: RunnerWaitingForModel(),
        RUNNER_2_ID: RunnerWaitingForModel(),
    }

    # Local node has already marked its shard as downloaded (not actually used by _load_model)
    local_download_status = {
        shard1: DownloadCompleted(shard_metadata=shard1, node_id=NODE_A)  # type: ignore[reportUnhashable]
    }

    # Global view has completed downloads for both nodes
    global_download_status = {
        NODE_A: [DownloadCompleted(shard_metadata=shard1, node_id=NODE_A)],
        NODE_B: [DownloadCompleted(shard_metadata=shard2, node_id=NODE_B)],
    }

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        download_status=local_download_status,
        global_download_status=global_download_status,
        instances=instances,
        all_runners=all_runners,
        tasks={},
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
    runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerWaitingForModel()
    )

    runners = {RUNNER_1_ID: runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {RUNNER_1_ID: RunnerWaitingForModel()}

    # Local status claims the shard is downloaded already
    local_download_status = {
        shard: DownloadCompleted(shard_metadata=shard, node_id=NODE_A)  # type: ignore[reportUnhashable]
    }

    # Global view hasn't caught up yet (no completed shards recorded for NODE_A)
    global_download_status: dict[NodeId, list[DownloadProgress]] = {
        NODE_A: [],
        NODE_B: [],
    }

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        download_status=local_download_status,
        global_download_status=global_download_status,
        instances=instances,
        all_runners=all_runners,
        tasks={},
    )

    assert result is None


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
        bound_instance=bound_instance, status=RunnerWaitingForModel()
    )

    runners = {RUNNER_1_ID: local_runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {
        RUNNER_1_ID: RunnerWaitingForModel(),
        RUNNER_2_ID: RunnerWaitingForModel(),
    }

    # Only NODE_A's shard is recorded as downloaded globally
    local_download_status = {
        shard1: DownloadCompleted(shard_metadata=shard1, node_id=NODE_A)  # type: ignore[reportUnhashable]
    }
    global_download_status = {
        NODE_A: [DownloadCompleted(shard_metadata=shard1, node_id=NODE_A)],
        NODE_B: [],  # NODE_B has no downloads completed yet
    }

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        download_status=local_download_status,
        global_download_status=global_download_status,
        instances=instances,
        all_runners=all_runners,
        tasks={},
    )

    assert result is None
