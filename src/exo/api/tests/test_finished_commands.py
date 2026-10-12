from exo.api.finished_commands import FORGET_AFTER_SECONDS, finishes_to_resend
from exo.shared.models.model_cards import ModelId
from exo.shared.types.common import CommandId
from exo.shared.types.state import State
from exo.shared.types.tasks import TaskId, TaskStatus, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId

STILL_RUNNING = CommandId("still-running")
DONE = CommandId("done")
DONE_LONG_AGO = CommandId("done-long-ago")
NOW = 1000.0


def _state_with_tasks_for(*command_ids: CommandId) -> State:
    tasks = {
        TaskId(f"task-of-{command_id}"): TextGeneration(
            task_id=TaskId(f"task-of-{command_id}"),
            command_id=command_id,
            instance_id=InstanceId("instance"),
            task_status=TaskStatus.Running,
            task_params=TextGenerationTaskParams(
                model=ModelId("test/model"),
                input=[InputMessage(role="user", content=InputMessageContent("hi"))],
            ),
        )
        for command_id in command_ids
    }
    return State(tasks=tasks)


def test_resends_while_state_still_has_the_task() -> None:
    resend, forget = finishes_to_resend(
        {STILL_RUNNING: NOW - 15, DONE: NOW - 15},
        _state_with_tasks_for(STILL_RUNNING),
        NOW,
    )
    assert resend == [STILL_RUNNING]
    assert forget == []


def test_keeps_resending_however_long_the_task_stays() -> None:
    resend, forget = finishes_to_resend(
        {STILL_RUNNING: NOW - FORGET_AFTER_SECONDS * 10},
        _state_with_tasks_for(STILL_RUNNING),
        NOW,
    )
    assert resend == [STILL_RUNNING]
    assert forget == []


def test_forgets_a_finish_only_long_after_its_task_is_gone() -> None:
    # A replaced master can still come back with the task for a while
    resend, forget = finishes_to_resend(
        {DONE: NOW - 15, DONE_LONG_AGO: NOW - FORGET_AFTER_SECONDS - 1},
        _state_with_tasks_for(),
        NOW,
    )
    assert resend == []
    assert forget == [DONE_LONG_AGO]
