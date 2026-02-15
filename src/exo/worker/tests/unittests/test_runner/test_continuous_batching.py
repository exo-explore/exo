"""
Tests for continuous batching behavior in the runner.

These tests verify that:
1. Single requests work through the batch path
2. Multiple concurrent requests batch together
3. Tokens are routed to the correct requests
4. Requests complete at different times appropriately

NOTE: These tests require the continuous-batching runner architecture
(BatchGenerationEngine) which is not yet integrated with main.
"""

# ruff: noqa: E402
# pyright: reportAny=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeVarUse=false

from typing import Any

import pytest

import exo.worker.runner.runner as mlx_runner
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
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
            tuple[CommandId, TaskId, TextGenerationTaskParams]
        ] = []
        self._uid_counter = 0
        self._tokens_per_request = 3  # Default: generate 3 tokens before completing
        self.rank = 0  # Fake rank for testing

    def queue_request(
        self,
        command_id: CommandId,
        task_id: TaskId,
        task_params: TextGenerationTaskParams,
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
        task_params: TextGenerationTaskParams | None,
    ) -> int:
        uid = self._uid_counter
        self._uid_counter += 1
        # Track: (command_id, task_id, tokens_generated, max_tokens)
        max_tokens = (
            task_params.max_output_tokens if task_params else self._tokens_per_request
        )
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
                        usage=None,
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

    def sync_completions(self) -> None:
        pass  # Completions already removed in step()

    @property
    def is_distributed(self) -> bool:
        return False  # Non-distributed mode for testing


class MockTokenizer:
    """Mock tokenizer with tool calling disabled."""

    tool_parser = None
    tool_call_start = None
    tool_call_end = None
    has_tool_calling = False


class FakeGroup:
    """Fake MLX distributed group for testing."""

    def rank(self) -> int:
        return 0

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
    monkeypatch.setattr(mlx_runner, "load_mlx_items", make_nothin((1, MockTokenizer)))
    monkeypatch.setattr(mlx_runner, "warmup_inference", make_nothin(1))
    monkeypatch.setattr(mlx_runner, "_check_for_debug_prompts", make_nothin(None))
    monkeypatch.setattr(mlx_runner, "BatchGenerationEngine", FakeBatchEngineWithTokens)


class EventCollector:
    """Collects events directly into a list to avoid mp_channel flakiness."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def send(self, event: Event) -> None:
        self.events.append(event)

    def close(self) -> None:
        pass

    def join(self) -> None:
        pass


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
    event_collector = EventCollector()

    shutdown_task = Shutdown(
        task_id=TaskId("shutdown"),
        instance_id=INSTANCE_1_ID,
        runner_id=RUNNER_1_ID,
    )

    with task_sender:
        # Send all tasks including shutdown
        for t in tasks:
            task_sender.send(t)
        task_sender.send(shutdown_task)

        # Disable cleanup methods to prevent issues
        task_receiver.close = lambda: None
        task_receiver.join = lambda: None

        mlx_runner.main(bound_instance, event_collector, task_receiver)  # type: ignore[arg-type]

        return event_collector.events


INIT_TASK = ConnectToGroup(task_id=TaskId("init"), instance_id=INSTANCE_1_ID)
LOAD_TASK = LoadModel(task_id=TaskId("load"), instance_id=INSTANCE_1_ID)
WARMUP_TASK = StartWarmup(task_id=TaskId("warmup"), instance_id=INSTANCE_1_ID)


def make_chat_task(
    task_id: str, command_id: str, max_tokens: int = 3
) -> TextGeneration:
    return TextGeneration(
        task_id=TaskId(task_id),
        command_id=CommandId(command_id),
        task_params=TextGenerationTaskParams(
            model=MODEL_A_ID,
            input=[InputMessage(role="user", content="hello")],
            stream=True,
            max_output_tokens=max_tokens,
        ),
        instance_id=INSTANCE_1_ID,
    )


def test_single_request_generates_tokens(patch_batch_engine: None):
    """
    Verify a single request generates the expected tokens through the batch path.

    Tokens are generated during the generation loop (not during shutdown drain).
    The task completes after all tokens are generated.
    """
    chat_task = make_chat_task("chat1", "cmd1", max_tokens=3)
    events = _run_with_tasks([INIT_TASK, LOAD_TASK, WARMUP_TASK, chat_task])

    # Verify ChunkGenerated events are emitted for all tokens
    chunk_events = [
        e
        for e in events
        if isinstance(e, ChunkGenerated) and e.command_id == CommandId("cmd1")
    ]
    assert len(chunk_events) == 3, (
        f"Expected 3 ChunkGenerated events, got {len(chunk_events)}"
    )

    # Last chunk should have finish_reason="stop"
    last_chunk = chunk_events[-1].chunk
    assert isinstance(last_chunk, TokenChunk)
    assert last_chunk.finish_reason == "stop"

    # Task should be marked complete after tokens are generated
    chat_complete = [
        e
        for e in events
        if isinstance(e, TaskStatusUpdated)
        and e.task_id == TaskId("chat1")
        and e.task_status == TaskStatus.Complete
    ]
    assert len(chat_complete) == 1, "Expected exactly one chat task Complete status"


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


def test_multiple_requests_generate_tokens(patch_batch_engine: None):
    """Verify multiple requests each generate their expected tokens."""
    chat1 = make_chat_task("chat1", "cmd1", max_tokens=2)
    chat2 = make_chat_task("chat2", "cmd2", max_tokens=2)
    events = _run_with_tasks([INIT_TASK, LOAD_TASK, WARMUP_TASK, chat1, chat2])

    # Both requests should generate their expected number of tokens
    cmd1_chunks = [
        e
        for e in events
        if isinstance(e, ChunkGenerated) and e.command_id == CommandId("cmd1")
    ]
    cmd2_chunks = [
        e
        for e in events
        if isinstance(e, ChunkGenerated) and e.command_id == CommandId("cmd2")
    ]

    assert len(cmd1_chunks) == 2, f"Expected 2 chunks for cmd1, got {len(cmd1_chunks)}"
    assert len(cmd2_chunks) == 2, f"Expected 2 chunks for cmd2, got {len(cmd2_chunks)}"

    # Both tasks should be completed
    completed_task_ids = {
        e.task_id
        for e in events
        if isinstance(e, TaskStatusUpdated)
        and e.task_status == TaskStatus.Complete
        and e.task_id in (TaskId("chat1"), TaskId("chat2"))
    }
    assert TaskId("chat1") in completed_task_ids
    assert TaskId("chat2") in completed_task_ids
