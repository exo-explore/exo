from typing import cast

import exo.worker.plan as plan_mod
from exo.shared.types.api import ChatCompletionTaskParams
from exo.shared.types.tasks import ChatCompletion, Task, TaskId, TaskStatus
from exo.shared.types.worker.instances import BoundInstance, InstanceId
from exo.shared.types.worker.runners import (
    RunnerIdle,
    RunnerReady,
    RunnerRunning,
)
from exo.worker.tests.constants import (
    COMMAND_1_ID,
    INSTANCE_1_ID,
    MODEL_A_ID,
    NODE_A,
    NODE_B,
    RUNNER_1_ID,
    RUNNER_2_ID,
    TASK_1_ID,
)
from exo.worker.tests.unittests.conftest import (
    FakeRunnerSupervisor,
    OtherTask,
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)


def test_plan_forwards_pending_chat_completion_when_runner_ready():
    """
    When there is a pending ChatCompletion for the local instance and all
    runners are Ready/Running, plan() should forward that task.
    """
    shard0 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: shard0, RUNNER_2_ID: shard1},
    )
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerReady()
    )

    runners = {RUNNER_1_ID: local_runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {
        RUNNER_1_ID: RunnerReady(),
        RUNNER_2_ID: RunnerReady(),
    }

    task = ChatCompletion(
        task_id=TASK_1_ID,
        instance_id=INSTANCE_1_ID,
        task_status=TaskStatus.Pending,
        command_id=COMMAND_1_ID,
        task_params=ChatCompletionTaskParams(model=MODEL_A_ID, messages=[]),
    )

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        download_status={},
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={TASK_1_ID: task},
    )

    assert result is task


def test_plan_does_not_forward_chat_completion_if_any_runner_not_ready():
    """
    Even with a pending ChatCompletion, plan() should not forward it unless
    all runners for the instance are Ready/Running.
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
        bound_instance=bound_instance, status=RunnerReady()
    )

    runners = {RUNNER_1_ID: local_runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {
        RUNNER_1_ID: RunnerReady(),
        RUNNER_2_ID: RunnerIdle(),
    }

    task = ChatCompletion(
        task_id=TASK_1_ID,
        instance_id=INSTANCE_1_ID,
        task_status=TaskStatus.Pending,
        command_id=COMMAND_1_ID,
        task_params=ChatCompletionTaskParams(model=MODEL_A_ID, messages=[]),
    )

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        download_status={},
        global_download_status={NODE_A: [], NODE_B: []},
        instances=instances,
        all_runners=all_runners,
        tasks={TASK_1_ID: task},
    )

    assert result is None


def test_plan_does_not_forward_tasks_for_other_instances():
    """
    plan() should ignore pending ChatCompletion tasks whose instance_id does
    not match the local instance.
    """
    shard = get_pipeline_shard_metadata(model_id=MODEL_A_ID, device_rank=0)
    local_instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID},
        runner_to_shard={RUNNER_1_ID: shard},
    )
    bound_instance = BoundInstance(
        instance=local_instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerReady()
    )

    runners = {RUNNER_1_ID: local_runner}
    instances = {INSTANCE_1_ID: local_instance}
    all_runners = {RUNNER_1_ID: RunnerReady()}

    other_instance_id = InstanceId("instance-2")
    foreign_task = ChatCompletion(
        task_id=TaskId("other-task"),
        instance_id=other_instance_id,
        task_status=TaskStatus.Pending,
        command_id=COMMAND_1_ID,
        task_params=ChatCompletionTaskParams(model=MODEL_A_ID, messages=[]),
    )

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        download_status={},
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={foreign_task.task_id: foreign_task},
    )

    assert result is None


def test_plan_ignores_non_pending_or_non_chat_tasks():
    """
    _pending_tasks should not forward tasks that are either not ChatCompletion
    or not in Pending/Running states.
    """
    shard0 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: shard0, RUNNER_2_ID: shard1},
    )
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )

    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerReady()
    )

    runners = {RUNNER_1_ID: local_runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {
        RUNNER_1_ID: RunnerReady(),
        RUNNER_2_ID: RunnerReady(),
    }

    completed_task = ChatCompletion(
        task_id=TASK_1_ID,
        instance_id=INSTANCE_1_ID,
        task_status=TaskStatus.Complete,
        command_id=COMMAND_1_ID,
        task_params=ChatCompletionTaskParams(model=MODEL_A_ID, messages=[]),
    )

    other_task_id = TaskId("other-task")

    other_task = cast(
        Task,
        cast(
            object,
            OtherTask(
                task_id=other_task_id,
                instance_id=INSTANCE_1_ID,
                task_status=TaskStatus.Pending,
            ),
        ),
    )

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        download_status={},
        global_download_status={NODE_A: [], NODE_B: []},
        instances=instances,
        all_runners=all_runners,
        tasks={TASK_1_ID: completed_task, other_task_id: other_task},
    )

    assert result is None


def test_plan_returns_none_when_nothing_to_do():
    """
    If there are healthy runners, no downloads needed, and no pending tasks,
    plan() should return None (steady state).
    """
    shard0 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: shard0, RUNNER_2_ID: shard1},
    )
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerRunning()
    )

    runners = {RUNNER_1_ID: local_runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {
        RUNNER_1_ID: RunnerRunning(),
        RUNNER_2_ID: RunnerRunning(),
    }

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,  # type: ignore
        download_status={},
        global_download_status={NODE_A: [], NODE_B: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
    )

    assert result is None
