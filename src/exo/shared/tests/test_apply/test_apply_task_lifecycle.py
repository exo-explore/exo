"""Finished worker bookkeeping tasks leave the state; generation tasks wait for TaskDeleted."""

from exo.shared.apply import apply
from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.events import Event, IndexedEvent, TaskCreated, TaskStatusUpdated
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    LoadModel,
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

INSTANCE = InstanceId()


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
            model=ModelId("some/model"),
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
        created(task),
        TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Complete),
    )
    assert state.tasks[task.task_id].task_status == TaskStatus.Complete
