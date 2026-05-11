from typing import Any

import exo.worker.plan as plan_mod
from exo.shared.types.tasks import Shutdown
from exo.shared.types.worker.instances import BoundInstance, Instance, InstanceId
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerIdle,
    RunnerReady,
    RunnerRunning,
    RunnerStatus,
)
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


def test_plan_kills_runner_when_instance_missing():
    """
    If a local runner's instance is no longer present in state,
    plan() should return a Shutdown for that runner.
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
    runner = FakeRunnerSupervisor(bound_instance=bound_instance, status=RunnerReady())

    runners = {RUNNER_1_ID: runner}
    instances: dict[InstanceId, Instance] = {}
    all_runners = {RUNNER_1_ID: RunnerReady()}

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore[arg-type]
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert isinstance(result, Shutdown)
    assert result.instance_id == INSTANCE_1_ID
    assert result.runner_id == RUNNER_1_ID


def test_plan_kills_runner_when_sibling_failed():
    """
    If a sibling runner in the same instance has failed, the local runner
    should be shut down.
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
    runner = FakeRunnerSupervisor(bound_instance=bound_instance, status=RunnerReady())

    runners = {RUNNER_1_ID: runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {
        RUNNER_1_ID: RunnerReady(),
        RUNNER_2_ID: RunnerFailed(error_message="boom"),
    }

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore[arg-type]
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert isinstance(result, Shutdown)
    assert result.instance_id == INSTANCE_1_ID
    assert result.runner_id == RUNNER_1_ID


def test_plan_creates_runner_when_missing_for_node():
    """
    If shard_assignments specify a runner for this node but we don't have
    a local supervisor yet, plan() should emit a CreateRunner.
    """
    shard = get_pipeline_shard_metadata(model_id=MODEL_A_ID, device_rank=0)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID},
        runner_to_shard={RUNNER_1_ID: shard},
    )

    runners: dict[Any, Any] = {}  # nothing local yet
    instances = {INSTANCE_1_ID: instance}
    all_runners: dict[Any, Any] = {}

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    # We patched plan_mod.CreateRunner → CreateRunner
    assert isinstance(result, plan_mod.CreateRunner)
    assert result.instance_id == INSTANCE_1_ID
    assert isinstance(result.bound_instance, BoundInstance)
    assert result.bound_instance.instance is instance
    assert result.bound_instance.bound_runner_id == RUNNER_1_ID


def test_plan_does_not_create_runner_when_supervisor_already_present():
    """
    If we already have a local supervisor for the runner assigned to this node,
    plan() should not emit a CreateRunner again.
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
    runner = FakeRunnerSupervisor(bound_instance=bound_instance, status=RunnerReady())

    runners = {RUNNER_1_ID: runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {RUNNER_1_ID: RunnerReady()}

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore[arg-type]
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert result is None


def test_plan_kills_local_when_peer_cycled_back_to_idle():
    """
    Restart-cascade regression: a peer rank crashed mid-task, its supervisor
    immediately respawned a fresh process which emitted ``RunnerIdle``, and
    the transient ``RunnerFailed`` window was too short for our plan loop to
    observe. The local rank is still ``RunnerRunning`` from before the peer
    crash. Without this rule the bootstrap predicate (``all_runners_connecting``
    in ``_init_distributed_backend``) never fires and the respawned peer is
    stuck in ``RunnerIdle`` forever.

    See PR #15 (regression: aborted K=8 sweep at 14:35:05).
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
    runner = FakeRunnerSupervisor(bound_instance=bound_instance, status=RunnerRunning())

    runners = {RUNNER_1_ID: runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners: dict[RunnerId, RunnerStatus] = {
        RUNNER_1_ID: RunnerRunning(),
        # Peer just respawned: process is up but hasn't initialized
        # the distributed backend yet.
        RUNNER_2_ID: RunnerIdle(),
    }

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore[arg-type]
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert isinstance(result, Shutdown)
    assert result.instance_id == INSTANCE_1_ID
    assert result.runner_id == RUNNER_1_ID


def test_plan_does_not_kill_local_when_peer_idle_but_local_only_loaded():
    """
    During initial bootstrap a peer can legitimately sit at ``RunnerIdle``
    while we have completed our own ``LoadModel`` (loading is per-rank
    without a collective barrier; see ``runner.py`` case ``LoadModel``).
    The restart-cascade rule must NOT fire here -- only ``RunnerRunning``
    on the local rank guarantees we previously cleared warmup with all
    peers, which is the precondition that makes a peer ``RunnerIdle``
    a process-restart signal.
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
    runner = FakeRunnerSupervisor(bound_instance=bound_instance, status=RunnerReady())

    runners = {RUNNER_1_ID: runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners: dict[RunnerId, RunnerStatus] = {
        RUNNER_1_ID: RunnerReady(),
        RUNNER_2_ID: RunnerIdle(),
    }

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore[arg-type]
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert not isinstance(result, Shutdown), (
        "RunnerReady + peer=Idle is normal initial bootstrap; cascade "
        "rule must only fire after the local rank has been observed in "
        "RunnerRunning (proving warmup completed for all ranks)"
    )


def test_plan_does_not_kill_single_rank_instance_on_idle_self():
    """
    The restart-cascade rule must only fire on multi-rank instances. For a
    single-rank instance the local runner cycling through ``RunnerIdle``
    on its own is a normal transient (initial spawn) and there is no peer
    that needs to re-bootstrap.
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
    runner = FakeRunnerSupervisor(bound_instance=bound_instance, status=RunnerRunning())

    runners = {RUNNER_1_ID: runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners: dict[RunnerId, RunnerStatus] = {RUNNER_1_ID: RunnerRunning()}

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore[arg-type]
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert not isinstance(result, Shutdown)


def test_plan_does_not_create_runner_for_unassigned_node():
    """
    If this node does not appear in shard_assignments.node_to_runner,
    plan() should not try to create a runner on this node.
    """
    shard = get_pipeline_shard_metadata(model_id=MODEL_A_ID, device_rank=0)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_2_ID: shard},
    )

    runners: dict[RunnerId, FakeRunnerSupervisor] = {}  # no local runners
    instances = {INSTANCE_1_ID: instance}
    all_runners: dict[RunnerId, RunnerStatus] = {}

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )

    assert result is None
