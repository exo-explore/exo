"""Finished worker bookkeeping tasks, and those of a deleted instance, leave the state, and
none is added for an instance that is gone."""

import pytest
from loguru import logger

from exo.shared.apply import apply
from exo.shared.types.common import ModelId, NodeId
from exo.shared.types.events import (
    Event,
    IndexedEvent,
    InstanceCreated,
    InstanceDeleted,
    NodeTimedOut,
    TaskCreated,
    TaskStatusUpdated,
)
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    LoadModel,
    Shutdown,
    Task,
    TaskStatus,
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

AT = "2026-11-03T12:00:00+00:00"


def run(*events: Event) -> State:
    state = State()
    for idx, event in enumerate(events):
        state = apply(state, IndexedEvent(idx=idx, event=event))
    return state


def created(task: Task) -> TaskCreated:
    return TaskCreated(task_id=task.task_id, task=task)


def test_completed_worker_task_leaves_the_state() -> None:
    load = LoadModel(instance_id=INSTANCE, task_status=TaskStatus.Pending)
    state = run(
        INSTANCE_CREATED,
        created(load),
        TaskStatusUpdated(task_id=load.task_id, task_status=TaskStatus.Running),
        TaskStatusUpdated(task_id=load.task_id, task_status=TaskStatus.Complete),
    )
    assert load.task_id not in state.tasks


def test_running_worker_task_stays() -> None:
    load = LoadModel(instance_id=INSTANCE, task_status=TaskStatus.Pending)
    state = run(
        INSTANCE_CREATED,
        created(load),
        TaskStatusUpdated(task_id=load.task_id, task_status=TaskStatus.Running),
    )
    assert state.tasks[load.task_id].task_status == TaskStatus.Running


def test_failed_worker_task_stays_for_diagnosis() -> None:
    load = LoadModel(instance_id=INSTANCE, task_status=TaskStatus.Pending)
    state = run(
        INSTANCE_CREATED,
        created(load),
        TaskStatusUpdated(task_id=load.task_id, task_status=TaskStatus.Failed),
    )
    assert state.tasks[load.task_id].task_status == TaskStatus.Failed


def test_deleting_an_instance_drops_its_bookkeeping_tasks() -> None:
    failed_load = LoadModel(instance_id=INSTANCE, task_status=TaskStatus.Pending)
    other_runner = RunnerId()
    other_instance = get_mlx_ring_instance(
        instance_id=InstanceId(),
        model_id=MODEL,
        node_to_runner={NodeId("node"): other_runner},
        runner_to_shard={
            other_runner: get_pipeline_shard_metadata(MODEL, device_rank=0)
        },
    )
    other = LoadModel(
        instance_id=other_instance.instance_id, task_status=TaskStatus.Pending
    )
    state = run(
        INSTANCE_CREATED,
        InstanceCreated(instance=other_instance),
        created(failed_load),
        TaskStatusUpdated(task_id=failed_load.task_id, task_status=TaskStatus.Failed),
        created(other),
        InstanceDeleted(instance_id=INSTANCE, reason="UserStopped", at=AT),
    )
    assert failed_load.task_id not in state.tasks
    assert other.task_id in state.tasks


def test_a_runner_shutdown_for_a_deleted_instance_is_not_added_and_its_end_is_harmless() -> (
    None
):
    """A worker stops the runner of a deleted instance with a task of that instance. It does
    so whether or not the task is in the state, and its report that it finished then finds
    nothing to update."""
    shutdown = Shutdown(
        instance_id=INSTANCE, runner_id=RUNNER, task_status=TaskStatus.Pending
    )
    state = run(
        INSTANCE_CREATED,
        InstanceDeleted(instance_id=INSTANCE, reason="UserStopped", at=AT),
        created(shutdown),
        TaskStatusUpdated(task_id=shutdown.task_id, task_status=TaskStatus.Complete),
    )
    assert not state.tasks


def levels_logged_while(*events: Event) -> list[str]:
    """The levels of the records apply logs for these events, in order."""
    levels: list[str] = []
    sink_id = logger.add(
        lambda message: levels.append(message.record["level"].name), level="DEBUG"
    )
    try:
        run(*events)
    finally:
        logger.remove(sink_id)
    return levels


def test_a_normal_stops_shutdown_is_not_a_warning() -> None:
    """Every normal stop leaves a Shutdown for an instance already gone (the worker plans it
    because the instance left the state), on every node, so it is logged at debug."""
    shutdown = Shutdown(
        instance_id=INSTANCE, runner_id=RUNNER, task_status=TaskStatus.Pending
    )
    assert levels_logged_while(
        INSTANCE_CREATED,
        InstanceDeleted(instance_id=INSTANCE, reason="UserStopped", at=AT),
        created(shutdown),
    ) == ["DEBUG"]


def runner_tasks() -> list[Task]:
    """What a worker creates for its runner as the runner's instance goes away: the runner's
    shutdown."""
    return [
        Shutdown(
            instance_id=INSTANCE, runner_id=RUNNER, task_status=TaskStatus.Pending
        ),
    ]


@pytest.mark.parametrize(
    "order", ["created_after_the_deletion", "created_before_the_deletion"]
)
def test_a_departed_nodes_runner_tasks_never_stay_pending(order: str) -> None:
    """A node leaves while its runner is serving: the cluster removes the node, then deletes
    its instance, and the node's worker, still shutting down, creates its runner's tasks just
    before or just after that and stops before it reports them finished. Neither order leaves
    a task in the state."""
    tasks = runner_tasks()
    departure: list[Event] = [
        NodeTimedOut(node_id=NodeId("node")),
        InstanceDeleted(instance_id=INSTANCE, reason="NodeLost", at=AT),
    ]
    created_tasks: list[Event] = [created(task) for task in tasks]
    events = (
        [INSTANCE_CREATED, *departure, *created_tasks]
        if order == "created_after_the_deletion"
        else [INSTANCE_CREATED, *created_tasks, *departure]
    )

    state = run(*events)

    assert not state.tasks
