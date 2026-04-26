from collections import deque
from typing import Any, cast

import pytest

from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.events import ChunkGenerated, Event
from exo.shared.types.tasks import TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runner_response import FinishedResponse
from exo.utils.channels import MpSender
from exo.worker.engines.mlx.generator.batch_generate import ExoBatchGenerator
from exo.worker.runner.llm_inference import batch_generator as batch_generator_module
from exo.worker.runner.llm_inference.batch_generator import BatchGenerator


class _FakeBatchEngine:
    has_work: bool = False


class _FakeEventSender:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def send(self, event: Event) -> None:
        self.events.append(event)


def test_batch_generator_finishes_task_when_prompt_template_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sender = _FakeEventSender()
    model_id = ModelId("mlx-community/gpt-oss-120b-MXFP4-Q8")
    task = TextGeneration(
        instance_id=InstanceId("instance"),
        command_id=CommandId("command"),
        task_params=TextGenerationTaskParams(
            model=model_id,
            input=[
                InputMessage(
                    role="user",
                    content=InputMessageContent("hello"),
                )
            ],
        ),
    )

    # We bypass the dataclass __init__ because constructing a real BatchGenerator
    # requires a full inference engine, tokenizer, and MP queue stack. The test
    # only exercises the prompt-templating error path inside step(), so we wire
    # in fakes for just the attributes that path touches.
    generator = object.__new__(BatchGenerator)
    generator.model_id = model_id
    generator.device_rank = 0
    generator.tokenizer = cast(Any, object())
    generator.event_sender = cast(MpSender[Event], cast(object, sender))
    generator._queue = deque([task])  # pyright: ignore[reportPrivateUsage]
    generator._active_tasks = {}  # pyright: ignore[reportPrivateUsage]
    generator._cancelled_tasks = set()  # pyright: ignore[reportPrivateUsage]
    generator._gen = cast(ExoBatchGenerator, cast(object, _FakeBatchEngine()))  # pyright: ignore[reportPrivateUsage]

    def fail_template(*_args: object, **_kwargs: object) -> None:
        raise ValueError("bad tool history")

    monkeypatch.setattr(
        batch_generator_module,
        "apply_chat_template",
        fail_template,
    )

    results = list(generator.step())

    assert len(results) == 1
    assert results[0][0] == task.task_id
    assert isinstance(results[0][1], FinishedResponse)
    assert generator._active_tasks == {}  # pyright: ignore[reportPrivateUsage]
    assert len(sender.events) == 1
    assert isinstance(sender.events[0], ChunkGenerated)
    assert isinstance(sender.events[0].chunk, ErrorChunk)
    assert sender.events[0].chunk.error_message == "bad tool history"
