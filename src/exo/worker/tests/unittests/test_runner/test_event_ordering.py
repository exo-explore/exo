# Check tasks are complete before runner is ever ready.

# pyright: reportAny=false

from collections.abc import Iterable
from typing import Any, Callable
from unittest.mock import MagicMock

import pytest

import exo.worker.runner.runner as mlx_runner
from exo.shared.types.api import ChatCompletionMessage
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
    ChatCompletion,
    ChatCompletionTaskParams,
    ConnectToGroup,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
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
from exo.worker.engines.mlx.generator.batch_engine import (
    BatchedGenerationResponse,
)

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


class FakeBatchEngine:
    """
    Fake batch engine for testing.

    Queues requests on insert, returns one token per step.
    The runner's non-blocking loop drains all tasks before running batch steps,
    so this engine queues requests and has_active_requests returns True only
    after at least one request has been inserted.
    """

    def __init__(self, *_args: Any, **_kwargs: Any):
        self._active_requests: dict[int, tuple[CommandId, TaskId]] = {}
        self._pending_inserts: list[
            tuple[CommandId, TaskId, ChatCompletionTaskParams]
        ] = []
        self._uid_counter = 0
        self.rank = 0  # Fake rank for testing

    def queue_request(
        self,
        command_id: CommandId,
        task_id: TaskId,
        task_params: ChatCompletionTaskParams,
    ) -> None:
        """Queue a request for insertion."""
        self._pending_inserts.append((command_id, task_id, task_params))

    def sync_and_insert_pending(self) -> list[int]:
        """Insert all pending requests."""
        uids: list[int] = []
        for command_id, task_id, _task_params in self._pending_inserts:
            uid = self._uid_counter
            self._uid_counter += 1
            self._active_requests[uid] = (command_id, task_id)
            uids.append(uid)
        self._pending_inserts.clear()
        return uids

    @property
    def has_pending_inserts(self) -> bool:
        return len(self._pending_inserts) > 0

    def step(self) -> list[BatchedGenerationResponse]:
        results: list[BatchedGenerationResponse] = []
        # Process all active requests - return one token and complete
        for uid, (command_id, task_id) in list(self._active_requests.items()):
            results.append(
                BatchedGenerationResponse(
                    command_id=command_id,
                    task_id=task_id,
                    response=GenerationResponse(
                        token=0,
                        text="hi",
                        finish_reason="stop",
                    ),
                )
            )
            del self._active_requests[uid]
        return results

    @property
    def has_active_requests(self) -> bool:
        return len(self._active_requests) > 0

    @property
    def active_count(self) -> int:
        return len(self._active_requests)

    @property
    def pending_insert_count(self) -> int:
        return len(self._pending_inserts)

    @property
    def is_distributed(self) -> bool:
        return False  # Non-distributed mode for testing


class FakeGroup:
    """Fake MLX distributed group for testing."""

    def size(self) -> int:
        return 1  # Single node (non-distributed)


@pytest.fixture
def patch_out_mlx(monkeypatch: pytest.MonkeyPatch):
    # initialize_mlx returns a fake "group" (non-None for state machine)
    monkeypatch.setattr(mlx_runner, "initialize_mlx", make_nothin(FakeGroup()))
    monkeypatch.setattr(
        mlx_runner, "load_mlx_items", make_nothin((MagicMock(), MagicMock()))
    )
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


def test_chat_completion_generates_and_completes(patch_out_mlx: pytest.MonkeyPatch):
    """Verify chat completion generates tokens, completes, and runner returns to Ready."""
    events = _run([INIT_TASK, LOAD_TASK, WARMUP_TASK, CHAT_TASK, SHUTDOWN_TASK])

    expected_chunk = ChunkGenerated(
        command_id=COMMAND_1_ID,
        chunk=TokenChunk(
            idx=0,
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
            RunnerStatusUpdated(
                runner_id=RUNNER_1_ID, runner_status=RunnerRunning(active_requests=1)
            ),
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
            RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerShutdown()),
        ],
    )
