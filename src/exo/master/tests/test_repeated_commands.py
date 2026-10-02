"""The master acts on a command only once, however often it arrives."""

from collections import deque

import pytest

import exo.master.main as master_main
from exo.master.main import Master
from exo.shared.models.model_cards import ModelId
from exo.shared.types.common import CommandId
from exo.shared.types.state import State
from exo.shared.types.tasks import TaskId, TaskStatus
from exo.shared.types.tasks import TextGeneration as TextGenerationTask
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId


def fresh_master() -> Master:
    master = object.__new__(Master)
    master.state = State()
    master._processed_commands = set()  # pyright: ignore[reportPrivateUsage]
    master._processed_order = deque()  # pyright: ignore[reportPrivateUsage]
    return master


def test_a_repeated_command_is_only_processed_once() -> None:
    master = fresh_master()
    command_id = CommandId()

    assert master._first_time(command_id)  # pyright: ignore[reportPrivateUsage]
    assert not master._first_time(command_id)  # pyright: ignore[reportPrivateUsage]
    assert master._first_time(CommandId())  # pyright: ignore[reportPrivateUsage]


def test_only_recent_commands_are_remembered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(master_main, "PROCESSED_COMMANDS_KEPT", 2)
    master = fresh_master()
    oldest, *rest = [CommandId() for _ in range(3)]
    for command_id in (oldest, *rest):
        master._first_time(command_id)  # pyright: ignore[reportPrivateUsage]

    assert len(master._processed_commands) == 2  # pyright: ignore[reportPrivateUsage]
    assert master._first_time(oldest)  # pyright: ignore[reportPrivateUsage]


def test_a_request_the_previous_master_accepted_is_not_processed_again() -> None:
    master = fresh_master()
    command_id = CommandId()
    task_id = TaskId()
    master.state = State(
        tasks={
            task_id: TextGenerationTask(
                task_id=task_id,
                command_id=command_id,
                instance_id=InstanceId(),
                task_status=TaskStatus.Running,
                task_params=TextGenerationTaskParams(
                    model=ModelId("test-model"),
                    input=[
                        InputMessage(role="user", content=InputMessageContent("hi"))
                    ],
                ),
            )
        }
    )

    assert not master._first_time(command_id)  # pyright: ignore[reportPrivateUsage]
