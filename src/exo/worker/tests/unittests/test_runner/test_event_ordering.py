# Check tasks are complete before runner is ever ready.
from collections.abc import Iterable
from typing import Callable

import pytest

import exo.worker.runner.runner as mlx_runner
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.common import CommandId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.tasks import (
    ConnectToGroup,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
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
from exo.worker.engines.mlx.generator.batch_engine import BatchedGenerationResponse

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

CHAT_PARAMS = TextGenerationTaskParams(
    model=MODEL_A_ID,
    input=[InputMessage(role="user", content="hello")],
    stream=True,
    max_output_tokens=4,
    temperature=0.0,
)

CHAT_TASK = TextGeneration(
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
    # initialize_mlx returns a mock group
    monkeypatch.setattr(mlx_runner, "initialize_mlx", make_nothin(MockGroup()))
    monkeypatch.setattr(mlx_runner, "load_mlx_items", make_nothin((1, MockTokenizer)))
    monkeypatch.setattr(mlx_runner, "warmup_inference", make_nothin(1))
    monkeypatch.setattr(mlx_runner, "_check_for_debug_prompts", nothin)
    monkeypatch.setattr(mlx_runner, "BatchGenerationEngine", FakeBatchEngine)


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
    has_thinking = False


class MockGroup:
    def rank(self) -> int:
        return 0

    def size(self) -> int:
        return 1


class FakeBatchEngine:
    """Fake batch engine that generates a single 'hi' token per request."""

    def __init__(self, *_args: object, **_kwargs: object):
        self._active_requests: dict[int, tuple[CommandId, TaskId]] = {}
        self._pending_inserts: list[tuple[CommandId, TaskId, object]] = []
        self._uid_counter = 0
        self.rank = 0

    def queue_request(
        self, command_id: CommandId, task_id: TaskId, task_params: object
    ) -> str:
        self._pending_inserts.append((command_id, task_id, task_params))
        return ""

    def sync_and_insert_pending(self) -> list[int]:
        uids: list[int] = []
        for cmd_id, task_id, _params in self._pending_inserts:
            uid = self._uid_counter
            self._uid_counter += 1
            self._active_requests[uid] = (cmd_id, task_id)
            uids.append(uid)
        self._pending_inserts.clear()
        return uids

    def step(self) -> list[BatchedGenerationResponse]:
        results: list[BatchedGenerationResponse] = []
        for _uid, (cmd_id, task_id) in list(self._active_requests.items()):
            results.append(
                BatchedGenerationResponse(
                    command_id=cmd_id,
                    task_id=task_id,
                    response=GenerationResponse(
                        token=0, text="hi", finish_reason="stop", usage=None
                    ),
                )
            )
        self._active_requests.clear()
        return results

    def sync_completions(self) -> None:
        pass

    @property
    def has_active_requests(self) -> bool:
        return bool(self._active_requests)

    @property
    def has_pending_inserts(self) -> bool:
        return bool(self._pending_inserts)

    @property
    def pending_insert_count(self) -> int:
        return len(self._pending_inserts)

    @property
    def active_count(self) -> int:
        return len(self._active_requests)

    @property
    def is_distributed(self) -> bool:
        return False


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
            usage=None,
            stats=None,
        ),
    )

    assert_events_equal(
        events,
        [
            RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerIdle()),
            TaskStatusUpdated(
                task_id=INITIALIZATION_TASK_ID, task_status=TaskStatus.Running
            ),
            RunnerStatusUpdated(
                runner_id=RUNNER_1_ID, runner_status=RunnerConnecting()
            ),
            TaskAcknowledged(task_id=INITIALIZATION_TASK_ID),
            TaskStatusUpdated(
                task_id=INITIALIZATION_TASK_ID, task_status=TaskStatus.Complete
            ),
            RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerConnected()),
            TaskStatusUpdated(task_id=LOAD_TASK_ID, task_status=TaskStatus.Running),
            RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerLoading()),
            TaskAcknowledged(task_id=LOAD_TASK_ID),
            TaskStatusUpdated(task_id=LOAD_TASK_ID, task_status=TaskStatus.Complete),
            RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerLoaded()),
            TaskStatusUpdated(task_id=WARMUP_TASK_ID, task_status=TaskStatus.Running),
            RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerWarmingUp()),
            TaskAcknowledged(task_id=WARMUP_TASK_ID),
            TaskStatusUpdated(task_id=WARMUP_TASK_ID, task_status=TaskStatus.Complete),
            RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerReady()),
            # CHAT TASK: queued, tokens generated, then completed
            TaskStatusUpdated(
                task_id=CHAT_COMPLETION_TASK_ID, task_status=TaskStatus.Running
            ),
            TaskAcknowledged(task_id=CHAT_COMPLETION_TASK_ID),
            RunnerStatusUpdated(
                runner_id=RUNNER_1_ID,
                runner_status=RunnerRunning(active_requests=1),
            ),
            # Generation loop produces token and completes the task
            expected_chunk,
            TaskStatusUpdated(
                task_id=CHAT_COMPLETION_TASK_ID, task_status=TaskStatus.Complete
            ),
            RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerReady()),
            # SHUTDOWN
            TaskStatusUpdated(task_id=SHUTDOWN_TASK_ID, task_status=TaskStatus.Running),
            RunnerStatusUpdated(
                runner_id=RUNNER_1_ID, runner_status=RunnerShuttingDown()
            ),
            TaskAcknowledged(task_id=SHUTDOWN_TASK_ID),
            TaskStatusUpdated(
                task_id=SHUTDOWN_TASK_ID, task_status=TaskStatus.Complete
            ),
            RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerShutdown()),
        ],
    )
