"""Finished worker bookkeeping tasks, and those of a deleted instance, leave the state;
generation tasks wait for TaskDeleted, and are never added for an instance that is gone."""

from exo.shared.apply import apply
from exo.shared.types.common import CommandId, ModelId, NodeId
from exo.shared.types.events import (
    Event,
    IndexedEvent,
    InstanceCreated,
    InstanceDeleted,
    TaskCreated,
    TaskStatusUpdated,
)
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    LoadModel,
    Shutdown,
    Task,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runners import RunnerId
from exo.worker.tests.unittests.conftest import (
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)

INSTANCE = InstanceId()
MODEL = ModelId("some/model")
RUNNER = RunnerId()
INSTANCE_CREATED = InstanceCreated(
    instance=get_mlx_ring_instance(
        instance_id=INSTANCE,
        model_id=MODEL,
        node_to_runner={NodeId("node"): RUNNER},
        runner_to_shard={RUNNER: get_pipeline_shard_metadata(MODEL, device_rank=0)},
    )
)


def run(*events: Event) -> State:
    state = State()
    for idx, event in enumerate(events):
        state = apply(state, IndexedEvent(idx=idx, event=event))
    return state


def generation() -> TextGeneration:
    return TextGeneration(
        instance_id=INSTANCE,
        task_status=TaskStatus.Pending,
        command_id=CommandId(),
        task_params=TextGenerationTaskParams(
            model=MODEL,
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        ),
    )


def created(task: Task) -> TaskCreated:
    return TaskCreated(task_id=task.task_id, task=task)


def test_completed_worker_task_leaves_the_state() -> None:
    load = LoadModel(instance_id=INSTANCE, task_status=TaskStatus.Pending)
    state = run(
        created(load),
        TaskStatusUpdated(task_id=load.task_id, task_status=TaskStatus.Running),
        TaskStatusUpdated(task_id=load.task_id, task_status=TaskStatus.Complete),
    )
    assert load.task_id not in state.tasks


def test_running_worker_task_stays() -> None:
    load = LoadModel(instance_id=INSTANCE, task_status=TaskStatus.Pending)
    state = run(
        created(load),
        TaskStatusUpdated(task_id=load.task_id, task_status=TaskStatus.Running),
    )
    assert state.tasks[load.task_id].task_status == TaskStatus.Running


def test_failed_worker_task_stays_for_diagnosis() -> None:
    load = LoadModel(instance_id=INSTANCE, task_status=TaskStatus.Pending)
    state = run(
        created(load),
        TaskStatusUpdated(task_id=load.task_id, task_status=TaskStatus.Failed),
    )
    assert state.tasks[load.task_id].task_status == TaskStatus.Failed


def test_completed_generation_task_stays_until_its_request_finishes() -> None:
    task = generation()
    state = run(
        INSTANCE_CREATED,
        created(task),
        TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Complete),
    )
    assert state.tasks[task.task_id].task_status == TaskStatus.Complete


def test_deleting_an_instance_drops_its_bookkeeping_tasks() -> None:
    failed_load = LoadModel(instance_id=INSTANCE, task_status=TaskStatus.Pending)
    request = generation()
    other = LoadModel(instance_id=InstanceId(), task_status=TaskStatus.Pending)
    state = run(
        INSTANCE_CREATED,
        created(failed_load),
        TaskStatusUpdated(task_id=failed_load.task_id, task_status=TaskStatus.Failed),
        created(request),
        created(other),
        InstanceDeleted(instance_id=INSTANCE),
    )
    assert failed_load.task_id not in state.tasks
    # The API still needs the request's task to end its stream
    assert request.task_id in state.tasks
    assert other.task_id in state.tasks


def test_a_request_for_a_deleted_instance_is_not_added() -> None:
    """The master can send a request to an instance it has just deleted: its own state
    changes only when its events come back. Nothing would run such a task, and the deletion
    has already ended the instance's requests, so it must not be added."""
    request = generation()
    state = run(
        INSTANCE_CREATED,
        InstanceDeleted(instance_id=INSTANCE),
        created(request),
    )
    assert request.task_id not in state.tasks


def test_a_request_for_an_unknown_instance_is_not_added() -> None:
    request = generation()
    state = run(created(request))
    assert request.task_id not in state.tasks


def test_a_worker_task_for_a_deleted_instance_is_still_added() -> None:
    """A worker shuts down the runner of a deleted instance with a task of that instance."""
    shutdown = Shutdown(
        instance_id=INSTANCE, runner_id=RUNNER, task_status=TaskStatus.Pending
    )
    state = run(
        INSTANCE_CREATED,
        InstanceDeleted(instance_id=INSTANCE),
        created(shutdown),
    )
    assert shutdown.task_id in state.tasks
