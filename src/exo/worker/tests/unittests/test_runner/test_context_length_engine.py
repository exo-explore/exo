"""The sequential engine ends a request the model can't take with its error and goes on: the
runner isn't brought down, and the next request runs. (Needs MLX.)"""

from collections.abc import Generator
from typing import cast, final

import pytest
from jinja2 import TemplateError

import exo.worker.runner.llm_inference.batch_generator as batch_generator
from exo.shared.models.model_cards import ModelId
from exo.shared.request_errors import generation_error_code
from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.common import CommandId
from exo.shared.types.events import ChunkGenerated, Event
from exo.shared.types.tasks import TaskId, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runner_response import FinishedResponse, GenerationResponse
from exo.utils.channels import MpSender, mp_channel
from exo.worker.runner.context_length import ContextLengthExceededError
from exo.worker.runner.llm_inference.batch_generator import SequentialGenerator

MODEL = ModelId("test-model")
TOO_LONG = (
    "This model's maximum context length is 32 tokens. However, your messages "
    "resulted in 40 tokens. Please reduce the length of the messages."
)


@final
class _Events:
    def __init__(self) -> None:
        self.sent: list[Event] = []

    def send(self, event: Event) -> None:
        self.sent.append(event)


def _request(text: str) -> TextGeneration:
    # The engine's ranks agree on tasks by their (UUID) ids
    return TextGeneration(
        instance_id=InstanceId("instance"),
        command_id=CommandId(),
        task_params=TextGenerationTaskParams(
            model=MODEL,
            input=[InputMessage(role="user", content=InputMessageContent(text))],
        ),
    )


def _engine(events: _Events) -> SequentialGenerator:
    _, cancel_receiver = mp_channel[TaskId]()
    return SequentialGenerator(
        model=cast(batch_generator.Model, cast(object, None)),
        tokenizer=cast(batch_generator.TokenizerWrapper, cast(object, None)),
        group=None,
        kv_prefix_cache=None,
        tool_parser=None,
        model_id=MODEL,
        device_rank=0,
        cancel_receiver=cancel_receiver,
        event_sender=cast(MpSender[Event], cast(object, events)),
    )


def _refusing(
    self: SequentialGenerator, task: TextGeneration, prompt: str
) -> Generator[GenerationResponse]:
    """mlx_generate for a prompt that doesn't fit: it raises as it tokenizes the prompt"""
    del self, task
    if prompt == "long":
        raise ContextLengthExceededError(TOO_LONG)
    yield GenerationResponse(text="hi", token=1, finish_reason="stop", usage=None)


def _template(
    tokenizer: object, params: TextGenerationTaskParams, model_type: str | None
) -> str:
    del tokenizer, model_type
    content = str(params.input[0].content)
    if content == "bad":
        raise TemplateError("Conversation roles must alternate user/assistant")
    return content


def _no_model_type(model: object) -> str | None:
    del model
    return None


def _no_parsing(
    responses: Generator[GenerationResponse | None], *_: object
) -> Generator[None]:
    for _response in responses:
        yield None


@pytest.fixture(autouse=True)
def fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(batch_generator, "apply_chat_template", _template)
    monkeypatch.setattr(batch_generator, "model_type_of", _no_model_type)
    monkeypatch.setattr(SequentialGenerator, "_build_generator", _refusing)
    monkeypatch.setattr(batch_generator, "apply_all_parsers", _no_parsing)


def _errors(events: _Events) -> dict[str, str]:
    return {
        event.command_id: event.chunk.error_message
        for event in events.sent
        if isinstance(event, ChunkGenerated) and isinstance(event.chunk, ErrorChunk)
    }


@pytest.mark.parametrize("refused", ["long", "bad"])
def test_a_refused_request_ends_with_its_error_and_the_next_one_runs(
    refused: str,
) -> None:
    events = _Events()
    engine = _engine(events)
    bad, fine = _request(refused), _request("fine")
    engine.submit(bad)
    engine.submit(fine)

    finished: list[TaskId] = []
    for _ in range(5):
        finished.extend(
            task_id
            for task_id, result in engine.step()
            if isinstance(result, FinishedResponse)
        )

    assert sorted(finished) == sorted([bad.task_id, fine.task_id])
    errors = _errors(events)
    assert list(errors) == [bad.command_id]
    assert generation_error_code(errors[bad.command_id]) == (
        "context_length_exceeded" if refused == "long" else "invalid_request"
    )
