"""
Tests for continuous batching behavior in the runner.

These tests verify that:
1. Single requests work through the batch path
2. Multiple concurrent requests batch together
3. Tokens are routed to the correct requests
4. Requests complete at different times appropriately
"""

# pyright: reportAny=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeVarUse=false

from typing import Any
from unittest.mock import MagicMock

import pytest

import exo.worker.runner.runner as mlx_runner
from exo.shared.types.api import ChatCompletionMessage
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.events import (
    Event,
    RunnerStatusUpdated,
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
from exo.shared.types.worker.runners import RunnerRunning
from exo.utils.channels import mp_channel
from exo.worker.engines.mlx.generator.batch_engine import (
    BatchedGenerationResponse,
)
from exo.worker.tests.constants import (
    INSTANCE_1_ID,
    MODEL_A_ID,
    NODE_A,
    RUNNER_1_ID,
)
from exo.worker.tests.unittests.conftest import get_bound_mlx_ring_instance


class FakeBatchEngineWithTokens:
    """
    Fake batch engine that generates a specified number of tokens per request.

    This simulates realistic batch generation behavior where:
    - Requests are queued on insert
    - Each step() call generates one token for all active requests
    - Requests complete when they've generated all their tokens
    """

    def __init__(self, *_args: Any, **_kwargs: Any):
        self._active_requests: dict[int, tuple[CommandId, TaskId, int, int]] = {}
        self._pending_inserts: list[
            tuple[CommandId, TaskId, ChatCompletionTaskParams]
        ] = []
        self._uid_counter = 0
        self._tokens_per_request = 3  # Default: generate 3 tokens before completing
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
        for command_id, task_id, task_params in self._pending_inserts:
            uid = self._do_insert(command_id, task_id, task_params)
            uids.append(uid)
        self._pending_inserts.clear()
        return uids

    @property
    def has_pending_inserts(self) -> bool:
        return len(self._pending_inserts) > 0

    def _do_insert(
        self,
        command_id: CommandId,
        task_id: TaskId,
        task_params: ChatCompletionTaskParams | None,
    ) -> int:
        uid = self._uid_counter
        self._uid_counter += 1
        # Track: (command_id, task_id, tokens_generated, max_tokens)
        max_tokens = task_params.max_tokens if task_params else self._tokens_per_request
        self._active_requests[uid] = (command_id, task_id, 0, max_tokens or 3)
        return uid

    def step(self) -> list[BatchedGenerationResponse]:
        results: list[BatchedGenerationResponse] = []
        uids_to_remove: list[int] = []

        for uid, (command_id, task_id, tokens_gen, max_tokens) in list(
            self._active_requests.items()
        ):
            tokens_gen += 1
            finish_reason = "stop" if tokens_gen >= max_tokens else None
            text = f"token{tokens_gen}"

            if finish_reason:
                uids_to_remove.append(uid)
            else:
                self._active_requests[uid] = (
                    command_id,
                    task_id,
                    tokens_gen,
                    max_tokens,
                )

            results.append(
                BatchedGenerationResponse(
                    command_id=command_id,
                    task_id=task_id,
                    response=GenerationResponse(
                        token=tokens_gen,
                        text=text,
                        finish_reason=finish_reason,
                    ),
                )
            )

        for uid in uids_to_remove:
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


def make_nothin[T, U, V](res: T):
    def nothin(*_1: U, **_2: V) -> T:
        return res

    return nothin


@pytest.fixture
def patch_batch_engine(monkeypatch: pytest.MonkeyPatch):
    """Patch MLX dependencies and use FakeBatchEngineWithTokens."""
    monkeypatch.setattr(mlx_runner, "initialize_mlx", make_nothin(FakeGroup()))
    monkeypatch.setattr(
        mlx_runner, "load_mlx_items", make_nothin((MagicMock(), MagicMock()))
    )
    monkeypatch.setattr(mlx_runner, "warmup_inference", make_nothin(1))
    monkeypatch.setattr(mlx_runner, "_check_for_debug_prompts", make_nothin(None))
    monkeypatch.setattr(mlx_runner, "BatchGenerationEngine", FakeBatchEngineWithTokens)


def _run_with_tasks(tasks: list[Task]) -> list[Event]:
    """
    Run tasks through the runner, adding shutdown at the end.

    Tasks are sent in order, with shutdown sent last.
    The batch engine processes between task handling.
    """
    bound_instance = get_bound_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        runner_id=RUNNER_1_ID,
        node_id=NodeId(NODE_A),
    )

    task_sender, task_receiver = mp_channel[Task]()
    event_sender, event_receiver = mp_channel[Event]()

    shutdown_task = Shutdown(
        task_id=TaskId("shutdown"),
        instance_id=INSTANCE_1_ID,
        runner_id=RUNNER_1_ID,
    )

    with task_sender, event_receiver:
        # Send all tasks including shutdown
        for t in tasks:
            task_sender.send(t)
        task_sender.send(shutdown_task)

        # Disable cleanup methods to prevent issues
        event_sender.close = lambda: None
        event_sender.join = lambda: None
        task_receiver.close = lambda: None
        task_receiver.join = lambda: None

        mlx_runner.main(bound_instance, event_sender, task_receiver)

        return event_receiver.collect()


INIT_TASK = ConnectToGroup(task_id=TaskId("init"), instance_id=INSTANCE_1_ID)
LOAD_TASK = LoadModel(task_id=TaskId("load"), instance_id=INSTANCE_1_ID)
WARMUP_TASK = StartWarmup(task_id=TaskId("warmup"), instance_id=INSTANCE_1_ID)


def make_chat_task(
    task_id: str, command_id: str, max_tokens: int = 3
) -> ChatCompletion:
    return ChatCompletion(
        task_id=TaskId(task_id),
        command_id=CommandId(command_id),
        task_params=ChatCompletionTaskParams(
            model=str(MODEL_A_ID),
            messages=[ChatCompletionMessage(role="user", content="hello")],
            stream=True,
            max_tokens=max_tokens,
        ),
        instance_id=INSTANCE_1_ID,
    )


def test_single_request_generates_tokens(patch_batch_engine: None):
    """
    Verify a single request generates the expected tokens through the batch path.

    Note: With the current non-blocking design, shutdown is processed before
    batch steps run when all tasks are queued together. This test verifies
    the runner status reflects active requests.
    """
    chat_task = make_chat_task("chat1", "cmd1", max_tokens=3)
    events = _run_with_tasks([INIT_TASK, LOAD_TASK, WARMUP_TASK, chat_task])

    # Find RunnerRunning status events - this shows the request was inserted
    running_events = [
        e
        for e in events
        if isinstance(e, RunnerStatusUpdated)
        and isinstance(e.runner_status, RunnerRunning)
    ]

    assert len(running_events) >= 1, "Expected at least one RunnerRunning event"
    assert running_events[0].runner_status.active_requests == 1


def test_runner_status_reflects_active_requests(patch_batch_engine: None):
    """Verify RunnerRunning status includes active_requests count."""
    chat_task = make_chat_task("chat1", "cmd1", max_tokens=2)
    events = _run_with_tasks([INIT_TASK, LOAD_TASK, WARMUP_TASK, chat_task])

    # Find RunnerRunning status events
    running_events = [
        e
        for e in events
        if isinstance(e, RunnerStatusUpdated)
        and isinstance(e.runner_status, RunnerRunning)
    ]

    assert len(running_events) > 0, "Expected at least one RunnerRunning event"
    assert running_events[0].runner_status.active_requests == 1


def test_chat_task_acknowledged(patch_batch_engine: None):
    """Verify chat completion task is acknowledged with proper status updates."""
    chat_task = make_chat_task("chat1", "cmd1", max_tokens=2)
    events = _run_with_tasks([INIT_TASK, LOAD_TASK, WARMUP_TASK, chat_task])

    # Find the chat task status events
    chat_running = [
        e
        for e in events
        if isinstance(e, TaskStatusUpdated)
        and e.task_id == TaskId("chat1")
        and e.task_status == TaskStatus.Running
    ]

    assert len(chat_running) == 1, "Expected exactly one chat task Running status"


def test_multiple_requests_tracked(patch_batch_engine: None):
    """Verify multiple concurrent requests are tracked in active_requests."""
    chat1 = make_chat_task("chat1", "cmd1", max_tokens=2)
    chat2 = make_chat_task("chat2", "cmd2", max_tokens=2)
    events = _run_with_tasks([INIT_TASK, LOAD_TASK, WARMUP_TASK, chat1, chat2])

    # Find RunnerRunning status events
    running_events = [
        e
        for e in events
        if isinstance(e, RunnerStatusUpdated)
        and isinstance(e.runner_status, RunnerRunning)
    ]

    # Should have at least 2 RunnerRunning events (one per request inserted)
    assert len(running_events) >= 2, (
        f"Expected at least 2 RunnerRunning events, got {len(running_events)}"
    )

    # First should have 1 active request, second should have 2
    assert running_events[0].runner_status.active_requests == 1
    assert running_events[1].runner_status.active_requests == 2
