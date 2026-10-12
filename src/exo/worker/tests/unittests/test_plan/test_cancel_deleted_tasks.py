"""A generation task deleted from the state while its runner works on it is cancelled."""

from exo.shared.types.tasks import (
    CancelTask,
    LoadModel,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runners import RunnerReady
from exo.utils.keyed_backoff import KeyedBackoff
from exo.worker.plan import plan
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
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)

INSTANCE = get_mlx_ring_instance(
    instance_id=INSTANCE_1_ID,
    model_id=MODEL_A_ID,
    node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
    runner_to_shard={
        RUNNER_1_ID: get_pipeline_shard_metadata(MODEL_A_ID, 0, 2),
        RUNNER_2_ID: get_pipeline_shard_metadata(MODEL_A_ID, 1, 2),
    },
)
GENERATION = TextGeneration(
    task_id=TASK_1_ID,
    instance_id=INSTANCE_1_ID,
    task_status=TaskStatus.Running,
    command_id=COMMAND_1_ID,
    task_params=TextGenerationTaskParams(
        model=MODEL_A_ID,
        input=[InputMessage(role="user", content=InputMessageContent("hi"))],
    ),
)


def plan_with(runner: FakeRunnerSupervisor, tasks: dict[TaskId, TextGeneration]):
    return plan(
        node_id=NODE_A,
        runners={RUNNER_1_ID: runner},  # type: ignore
        global_download_status={NODE_A: []},
        instances={INSTANCE_1_ID: INSTANCE},
        all_runners={RUNNER_1_ID: RunnerReady(), RUNNER_2_ID: RunnerReady()},
        tasks=tasks,
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )


def ready_runner() -> FakeRunnerSupervisor:
    return FakeRunnerSupervisor(
        bound_instance=BoundInstance(
            instance=INSTANCE, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
        ),
        status=RunnerReady(),
    )


def test_generation_task_deleted_while_running_is_cancelled() -> None:
    runner = ready_runner()
    runner.in_progress[TASK_1_ID] = GENERATION

    result = plan_with(runner, tasks={})

    assert isinstance(result, CancelTask)
    assert result.cancelled_task_id == TASK_1_ID
    assert result.runner_id == RUNNER_1_ID


def test_running_generation_task_still_in_the_state_is_left_alone() -> None:
    runner = ready_runner()
    runner.in_progress[TASK_1_ID] = GENERATION

    assert plan_with(runner, tasks={TASK_1_ID: GENERATION}) is None


def test_deleted_task_is_only_cancelled_once() -> None:
    runner = ready_runner()
    runner.in_progress[TASK_1_ID] = GENERATION
    runner.cancelled.add(TASK_1_ID)

    assert plan_with(runner, tasks={}) is None


def test_lifecycle_task_not_yet_in_the_state_is_not_cancelled() -> None:
    # The worker starts its own lifecycle tasks before the master has indexed them
    runner = ready_runner()
    load = LoadModel(instance_id=INSTANCE_1_ID, task_status=TaskStatus.Running)
    runner.in_progress[load.task_id] = load

    assert plan_with(runner, tasks={}) is None
