from exo.shared.apply import apply_remote_prefill_ready
from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.events import RemotePrefillReady
from exo.shared.types.state import State
from exo.shared.types.tasks import RemotePrefill, TaskId
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.instances import InstanceId


def _make_prefill_task(task_id: TaskId | None = None) -> RemotePrefill:
    tid = task_id or TaskId()
    return RemotePrefill(
        task_id=tid,
        instance_id=InstanceId(),
        command_id=CommandId(),
        model_id=ModelId("m"),
        task_params=TextGenerationTaskParams(model=ModelId("m"), input=[]),
        paired_task_id=TaskId(),
    )


def test_apply_remote_prefill_ready_updates_task() -> None:
    task = _make_prefill_task()
    state = State(tasks={task.task_id: task})

    new_state = apply_remote_prefill_ready(
        RemotePrefillReady(
            task_id=task.task_id,
            endpoint="10.0.0.1:55123",
            request_id="req-1",
            num_tokens=4096,
        ),
        state,
    )

    updated = new_state.tasks[task.task_id]
    assert isinstance(updated, RemotePrefill)
    assert updated.ready_endpoint == "10.0.0.1:55123"
    assert updated.ready_request_id == "req-1"
    assert updated.ready_num_tokens == 4096


def test_apply_remote_prefill_ready_ignores_unknown_task() -> None:
    state = State()
    new_state = apply_remote_prefill_ready(
        RemotePrefillReady(
            task_id=TaskId(),
            endpoint="a:1",
            request_id="r",
            num_tokens=1,
        ),
        state,
    )
    assert new_state == state


def test_apply_remote_prefill_ready_overwrites_existing() -> None:
    task = _make_prefill_task()
    state = State(tasks={task.task_id: task})

    s1 = apply_remote_prefill_ready(
        RemotePrefillReady(task_id=task.task_id, endpoint="a:1", request_id="r1", num_tokens=10),
        state,
    )
    s2 = apply_remote_prefill_ready(
        RemotePrefillReady(task_id=task.task_id, endpoint="b:2", request_id="r2", num_tokens=20),
        s1,
    )

    updated = s2.tasks[task.task_id]
    assert isinstance(updated, RemotePrefill)
    assert updated.ready_endpoint == "b:2"
    assert updated.ready_request_id == "r2"
    assert updated.ready_num_tokens == 20
