from exo.shared.apply import apply
from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import InputImageChunk
from exo.shared.types.common import CommandId
from exo.shared.types.events import (
    IndexedEvent,
    InputChunkReceived,
    TaskCreated,
    TaskDeleted,
)
from exo.shared.types.state import State
from exo.shared.types.tasks import TaskId, TaskStatus, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId


def test_apply_input_chunk_received_stores_chunk_in_state() -> None:
    command_id = CommandId("command-1")
    chunk = InputImageChunk(
        model=ModelId("mlx-community/test-model"),
        command_id=command_id,
        data="abc",
        chunk_index=0,
        total_chunks=1,
        image_index=0,
    )

    state = apply(
        State(),
        IndexedEvent(
            idx=0,
            event=InputChunkReceived(command_id=command_id, chunk=chunk),
        ),
    )

    assert state.input_chunks == {command_id: {0: chunk}}


def test_apply_task_deleted_removes_chunks_for_generation_command() -> None:
    command_id = CommandId("command-1")
    task_id = TaskId("task-1")
    chunk = InputImageChunk(
        model=ModelId("mlx-community/test-model"),
        command_id=command_id,
        data="abc",
        chunk_index=0,
        total_chunks=1,
        image_index=0,
    )
    task = TextGeneration(
        task_id=task_id,
        instance_id=InstanceId("instance-1"),
        task_status=TaskStatus.Pending,
        command_id=command_id,
        task_params=TextGenerationTaskParams(
            model=ModelId("mlx-community/test-model"),
            input=[
                InputMessage(role="user", content=InputMessageContent("hello")),
            ],
        ),
    )

    state = State()
    state = apply(
        state,
        IndexedEvent(
            idx=0,
            event=InputChunkReceived(command_id=command_id, chunk=chunk),
        ),
    )
    state = apply(
        state,
        IndexedEvent(idx=1, event=TaskCreated(task_id=task_id, task=task)),
    )
    state = apply(
        state,
        IndexedEvent(idx=2, event=TaskDeleted(task_id=task_id)),
    )

    assert state.tasks == {}
    assert state.input_chunks == {}
