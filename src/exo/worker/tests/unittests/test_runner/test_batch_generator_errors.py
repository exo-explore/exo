from collections import deque

from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.events import ChunkGenerated
from exo.shared.types.tasks import TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId
from exo.worker.runner.llm_inference import batch_generator as batch_generator_module
from exo.worker.runner.llm_inference.batch_generator import BatchGenerator, Finished


class _FakeBatchEngine:
    has_work = False


class _FakeEventSender:
    def __init__(self) -> None:
        self.events = []

    def send(self, event) -> None:
        self.events.append(event)


def test_batch_generator_finishes_task_when_prompt_template_fails(
    monkeypatch,
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

    generator = object.__new__(BatchGenerator)
    generator.model_id = model_id
    generator.device_rank = 0
    generator.tokenizer = object()
    generator.event_sender = sender
    generator._queue = deque([task])
    generator._active_tasks = {}
    generator._cancelled_tasks = set()
    generator._mlx_gen = _FakeBatchEngine()

    def fail_template(*_args, **_kwargs):
        raise ValueError("bad tool history")

    monkeypatch.setattr(
        batch_generator_module,
        "apply_chat_template",
        fail_template,
    )

    results = list(generator.step())

    assert len(results) == 1
    assert results[0][0] == task.task_id
    assert isinstance(results[0][1], Finished)
    assert generator._active_tasks == {}
    assert len(sender.events) == 1
    assert isinstance(sender.events[0], ChunkGenerated)
    assert isinstance(sender.events[0].chunk, ErrorChunk)
    assert sender.events[0].chunk.error_message == "bad tool history"
