# Check tasks are complete before runner is ever ready.
from collections.abc import Iterable
from typing import Callable

import pytest

import exo.worker.runner.runner as mlx_runner
from exo.shared.types.api import ChatCompletionMessage
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.tasks import (
    ChatCompletion,
    ChatCompletionTaskParams,
    ConnectToGroup,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskStatus,
)
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerWarmingUp,
)
from exo.utils.channels import mp_channel

from ...constants import (
    CHAT_COMPLETION_TASK_ID,
    COMMAND_1_ID,
    INITIALIZATION_TASK_ID,
    INSTANCE_1_ID,
    LOAD_TASK_ID,
    MODEL_A_ID,
    NODE_A,
    RUNNER_1_ID,
    SHUTDOWN_TASK_ID,
    WARMUP_TASK_ID,
)
from ..conftest import get_bound_mlx_ring_instance


def make_nothin[T, U, V](res: T) -> Callable[[], T]:
    def nothin(*_1: U, **_2: V) -> T:
        return res

    return nothin


nothin = make_nothin(None)


INIT_TASK = ConnectToGroup(
    task_id=INITIALIZATION_TASK_ID,
    instance_id=INSTANCE_1_ID,
)

LOAD_TASK = LoadModel(
    task_id=LOAD_TASK_ID,
    instance_id=INSTANCE_1_ID,
)

WARMUP_TASK = StartWarmup(
    task_id=WARMUP_TASK_ID,
    instance_id=INSTANCE_1_ID,
)

SHUTDOWN_TASK = Shutdown(
    task_id=SHUTDOWN_TASK_ID,
    instance_id=INSTANCE_1_ID,
    runner_id=RUNNER_1_ID,
)

CHAT_PARAMS = ChatCompletionTaskParams(
    model=str(MODEL_A_ID),
    messages=[ChatCompletionMessage(role="user", content="hello")],
    stream=True,
    max_tokens=4,
    temperature=0.0,
)

CHAT_TASK = ChatCompletion(
    task_id=CHAT_COMPLETION_TASK_ID,
    command_id=COMMAND_1_ID,
    task_params=CHAT_PARAMS,
    instance_id=INSTANCE_1_ID,
)


def assert_events_equal(test_events: Iterable[Event], true_events: Iterable[Event]):
    for test_event, true_event in zip(test_events, true_events, strict=True):
        test_event.event_id = true_event.event_id
        assert test_event == true_event, f"{test_event} != {true_event}"


@pytest.fixture
def patch_out_mlx(monkeypatch: pytest.MonkeyPatch):
    # initialize_mlx returns a "group" equal to 1
    monkeypatch.setattr(mlx_runner, "initialize_mlx", make_nothin(1))
    monkeypatch.setattr(mlx_runner, "load_mlx_items", make_nothin((1, MockTokenizer)))
    monkeypatch.setattr(mlx_runner, "warmup_inference", make_nothin(1))
    monkeypatch.setattr(mlx_runner, "_check_for_debug_prompts", nothin)
    # Mock apply_chat_template since we're using a fake tokenizer (integer 1).
    # Returns a prompt without thinking tag so detect_thinking_prompt_suffix returns None.
    monkeypatch.setattr(mlx_runner, "apply_chat_template", make_nothin("test prompt"))
    monkeypatch.setattr(mlx_runner, "detect_thinking_prompt_suffix", make_nothin(False))

    def fake_generate(*_1: object, **_2: object):
        yield GenerationResponse(token=0, text="hi", finish_reason="stop")

    monkeypatch.setattr(mlx_runner, "mlx_generate", fake_generate)


# Use a fake event_sender to remove test flakiness.
class EventCollector:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def send(self, event: Event) -> None:
        self.events.append(event)

    def close(self) -> None:
        pass

    def join(self) -> None:
        pass


class MockTokenizer:
    tool_parser = None
    tool_call_start = None
    tool_call_end = None
    has_tool_calling = False


def _run(tasks: Iterable[Task]):
    bound_instance = get_bound_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        runner_id=RUNNER_1_ID,
        node_id=NODE_A,
    )

    task_sender, task_receiver = mp_channel[Task]()
    event_sender = EventCollector()

    with task_sender:
        for t in tasks:
            task_sender.send(t)

        # worst monkeypatch known to man
        # this is some c++ nonsense
        task_receiver.close = nothin
        task_receiver.join = nothin

        mlx_runner.main(bound_instance, event_sender, task_receiver)  # type: ignore[arg-type]

        return event_sender.events


def test_events_processed_in_correct_order(patch_out_mlx: pytest.MonkeyPatch):
    events = _run([INIT_TASK, LOAD_TASK, WARMUP_TASK, CHAT_TASK, SHUTDOWN_TASK])

    expected_chunk = ChunkGenerated(
        command_id=COMMAND_1_ID,
        chunk=TokenChunk(
            model=MODEL_A_ID,
            text="hi",
            token_id=0,
            finish_reason="stop",
        ),
    )

    assert_events_equal(
        events,
        [
            RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerIdle()),
            TaskStatusUpdated(
                task_id=INITIALIZATION_TASK_ID, task_status=TaskStatus.Running
            ),
            TaskAcknowledged(task_id=INITIALIZATION_TASK_ID),
            RunnerStatusUpdated(
                runner_id=RUNNER_1_ID, runner_status=RunnerConnecting()
            ),
            TaskStatusUpdated(
                task_id=INITIALIZATION_TASK_ID, task_status=TaskStatus.Complete
            ),
            RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerConnected()),
            TaskStatusUpdated(task_id=LOAD_TASK_ID, task_status=TaskStatus.Running),
            TaskAcknowledged(task_id=LOAD_TASK_ID),
            RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerLoading()),
            TaskStatusUpdated(task_id=LOAD_TASK_ID, task_status=TaskStatus.Complete),
            RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerLoaded()),
            TaskStatusUpdated(task_id=WARMUP_TASK_ID, task_status=TaskStatus.Running),
            TaskAcknowledged(task_id=WARMUP_TASK_ID),
            RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerWarmingUp()),
            TaskStatusUpdated(task_id=WARMUP_TASK_ID, task_status=TaskStatus.Complete),
            RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerReady()),
            TaskStatusUpdated(
                task_id=CHAT_COMPLETION_TASK_ID, task_status=TaskStatus.Running
            ),
            TaskAcknowledged(task_id=CHAT_COMPLETION_TASK_ID),
            RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerRunning()),
            expected_chunk,
            TaskStatusUpdated(
                task_id=CHAT_COMPLETION_TASK_ID, task_status=TaskStatus.Complete
            ),
            # CHAT COMPLETION TASK SHOULD COMPLETE BEFORE RUNNER READY
            RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerReady()),
            TaskStatusUpdated(task_id=SHUTDOWN_TASK_ID, task_status=TaskStatus.Running),
            TaskAcknowledged(task_id=SHUTDOWN_TASK_ID),
            RunnerStatusUpdated(
                runner_id=RUNNER_1_ID, runner_status=RunnerShuttingDown()
            ),
            TaskStatusUpdated(
                task_id=SHUTDOWN_TASK_ID, task_status=TaskStatus.Complete
            ),
            # SPECIAL EXCEPTION FOR RUNNER SHUTDOWN
            RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerShutdown()),
        ],
    )
