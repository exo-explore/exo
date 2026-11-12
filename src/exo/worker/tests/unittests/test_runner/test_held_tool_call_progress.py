"""A tool call streams only once it is complete. While the model writes it, the engine says for
each held token that the request is still making progress (HeldResponse), and the runner
passes that on to its supervisor now and then (RunnerProgressed), so that a large tool call at
long context isn't taken for a stalled runner. (Needs MLX.)"""

from collections.abc import Generator
from types import SimpleNamespace
from typing import cast, final

import pytest

import exo.worker.runner.llm_inference.batch_generator as batch_generator
import exo.worker.runner.runner as runner_module
from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import TokenChunk, ToolCallChunk
from exo.shared.types.common import CommandId
from exo.shared.types.events import Event, RunnerEvent, RunnerProgressed
from exo.shared.types.tasks import TaskId, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runner_response import (
    FinishedResponse,
    GenerationResponse,
    HeldResponse,
)
from exo.utils.channels import MpSender, mp_channel
from exo.worker.runner.llm_inference.batch_generator import SequentialGenerator
from exo.worker.runner.llm_inference.tool_parsers import make_json_parser
from exo.worker.runner.runner import Runner

MODEL = ModelId("test-model")
# What the model writes: a word, then a tool call over several tokens
TOKENS = [
    "Sure",
    "<tool_call>",
    '{"name": "write_file", ',
    '"arguments": {"text": "a long file"}}',
    "</tool_call>",
]


@final
class _Events:
    def __init__(self) -> None:
        self.sent: list[Event | RunnerEvent] = []

    def send(self, event: Event | RunnerEvent) -> None:
        self.sent.append(event)


def _writing_a_tool_call(
    self: SequentialGenerator, task: TextGeneration, prompt: str
) -> Generator[GenerationResponse]:
    del self, task, prompt
    for i, text in enumerate(TOKENS):
        last = i == len(TOKENS) - 1
        yield GenerationResponse(
            text=text, token=i + 1, finish_reason="stop" if last else None, usage=None
        )


def _template(*_: object) -> str:
    return "write the file"


def _no_model_type(_: object) -> str | None:
    return None


@pytest.fixture(autouse=True)
def fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(batch_generator, "apply_chat_template", _template)
    monkeypatch.setattr(batch_generator, "model_type_of", _no_model_type)
    monkeypatch.setattr(SequentialGenerator, "_build_generator", _writing_a_tool_call)


def test_held_tool_call_tokens_are_reported_as_progress() -> None:
    _, cancel_receiver = mp_channel[TaskId]()
    engine = SequentialGenerator(
        model=cast(batch_generator.Model, cast(object, None)),
        tokenizer=cast(
            batch_generator.TokenizerWrapper,
            cast(object, SimpleNamespace(has_thinking=False)),
        ),
        group=None,
        kv_prefix_cache=None,
        tool_parser=make_json_parser(),
        model_id=MODEL,
        device_rank=0,
        cancel_receiver=cancel_receiver,
        event_sender=cast(MpSender[RunnerEvent], cast(object, _Events())),
    )
    task = TextGeneration(
        instance_id=InstanceId("instance"),
        command_id=CommandId(),
        task_params=TextGenerationTaskParams(
            model=MODEL,
            input=[InputMessage(role="user", content=InputMessageContent("go"))],
            tools=[{"type": "function", "function": {"name": "write_file"}}],
        ),
    )
    engine.submit(task)

    steps = [[result for _, result in engine.step()] for _ in range(len(TOKENS) + 1)]
    kinds = [[type(result).__name__ for result in step] for step in steps]

    # the word streams, the tool call's tokens are held (and reported as progress), and the
    # tool call goes out whole at the end
    assert isinstance(steps[0][0], TokenChunk)
    assert all(step == [HeldResponse()] for step in steps[1:-2]), kinds
    assert any(isinstance(result, ToolCallChunk) for result in steps[-2]), kinds
    assert any(
        isinstance(result, FinishedResponse) for step in steps[-2:] for result in step
    ), kinds


def test_the_runner_reports_held_progress_at_most_every_few_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [100.0]
    monkeypatch.setattr(runner_module.time, "monotonic", lambda: clock[0])
    events = _Events()
    runner = cast(
        Runner,
        cast(object, SimpleNamespace(_last_progress_report=0.0, event_sender=events)),
    )
    task_id = TaskId("task")
    for _ in range(50):  # 50 held tokens within a second
        Runner.report_progress(runner, task_id)
        clock[0] += 0.02
    assert [(type(e), getattr(e, "task_id", None)) for e in events.sent] == [
        (RunnerProgressed, task_id)
    ]
    clock[0] += runner_module.PROGRESS_REPORT_INTERVAL
    Runner.report_progress(runner, task_id)
    assert len(events.sent) == 2
