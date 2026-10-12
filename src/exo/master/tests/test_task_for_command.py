from exo.master.main import task_for_command
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


def test_master_finds_a_commands_task_in_state() -> None:
    # As a new master must, for requests that its predecessor started
    task = TextGeneration(
        task_id=TaskId("task"),
        command_id=CommandId("command"),
        instance_id=InstanceId("instance"),
        task_status=TaskStatus.Running,
        task_params=TextGenerationTaskParams(
            model=ModelId("test/model"),
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        ),
    )
    state = State(tasks={task.task_id: task})
    assert task_for_command(state, CommandId("command")) == TaskId("task")
    assert task_for_command(state, CommandId("another command")) is None
