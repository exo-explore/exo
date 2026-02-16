"""
Edge-case tests for continuous batching in the runner.

Tests cover:
1. Concurrent requests with overlapping tool calls
2. Requests that finish mid-generation with 'length' reason
3. Multiple requests finishing on the same step() call
4. Batch of 5+ simultaneous completions
"""

# ruff: noqa: E402
# pyright: reportAny=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeVarUse=false
# pyright: reportPrivateUsage=false

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

import exo.worker.runner.runner as mlx_runner
from exo.shared.types.api import FinishReason
from exo.shared.types.chunks import TokenChunk, ToolCallChunk
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
from exo.shared.types.worker.runners import RunnerReady, RunnerRunning
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

# ---------------------------------------------------------------------------
# Fake batch engines
# ---------------------------------------------------------------------------


class ScriptedBatchEngine:
    """Batch engine driven by scripted per-request token sequences.

    Each request produces a predefined list of (text, finish_reason) pairs.
    One step() call pops one token per active request.
    """

    def __init__(self, *_args: Any, **_kwargs: Any):
        self._active: dict[
            int, tuple[CommandId, TaskId, list[tuple[str, FinishReason | None]]]
        ] = {}
        self._pending: list[tuple[CommandId, TaskId, TextGenerationTaskParams]] = []
        self._uid = 0
        self.rank = 0
        # map command_id -> scripted tokens, set externally before tasks arrive
        self.scripts: dict[str, list[tuple[str, FinishReason | None]]] = {}

    def queue_request(
        self,
        command_id: CommandId,
        task_id: TaskId,
        task_params: TextGenerationTaskParams,
    ) -> None:
        self._pending.append((command_id, task_id, task_params))

    def sync_and_insert_pending(self) -> list[int]:
        uids: list[int] = []
        for cmd_id, task_id, _params in self._pending:
            uid = self._uid
            self._uid += 1
            script = list(self.scripts.get(str(cmd_id), [("tok", "stop")]))
            self._active[uid] = (cmd_id, task_id, script)
            uids.append(uid)
        self._pending.clear()
        return uids

    @property
    def has_pending_inserts(self) -> bool:
        return bool(self._pending)

    def step(self) -> list[BatchedGenerationResponse]:
        results: list[BatchedGenerationResponse] = []
        done: list[int] = []
        for uid, (cmd_id, task_id, script) in self._active.items():
            if not script:
                continue
            text, finish_reason = script.pop(0)
            results.append(
                BatchedGenerationResponse(
                    command_id=cmd_id,
                    task_id=task_id,
                    response=GenerationResponse(
                        token=0, text=text, finish_reason=finish_reason, usage=None
                    ),
                )
            )
            if finish_reason is not None:
                done.append(uid)
        for uid in done:
            del self._active[uid]
        return results

    @property
    def has_active_requests(self) -> bool:
        return bool(self._active)

    @property
    def active_count(self) -> int:
        return len(self._active)

    def sync_completions(self) -> None:
        pass

    @property
    def is_distributed(self) -> bool:
        return False


class FakeBatchEngineWithTokens:
    """Generates N tokens per request (reused from the main test file)."""

    def __init__(self, *_args: Any, **_kwargs: Any):
        self._active_requests: dict[int, tuple[CommandId, TaskId, int, int]] = {}
        self._pending_inserts: list[
            tuple[CommandId, TaskId, TextGenerationTaskParams]
        ] = []
        self._uid_counter = 0
        self.rank = 0

    def queue_request(
        self,
        command_id: CommandId,
        task_id: TaskId,
        task_params: TextGenerationTaskParams,
    ) -> None:
        self._pending_inserts.append((command_id, task_id, task_params))

    def sync_and_insert_pending(self) -> list[int]:
        uids: list[int] = []
        for command_id, task_id, task_params in self._pending_inserts:
            uid = self._uid_counter
            self._uid_counter += 1
            max_tokens = task_params.max_output_tokens or 3
            self._active_requests[uid] = (command_id, task_id, 0, max_tokens)
            uids.append(uid)
        self._pending_inserts.clear()
        return uids

    @property
    def has_pending_inserts(self) -> bool:
        return bool(self._pending_inserts)

    def step(self) -> list[BatchedGenerationResponse]:
        results: list[BatchedGenerationResponse] = []
        done: list[int] = []
        for uid, (cmd_id, task_id, tokens_gen, max_tokens) in list(
            self._active_requests.items()
        ):
            tokens_gen += 1
            finish = "stop" if tokens_gen >= max_tokens else None
            results.append(
                BatchedGenerationResponse(
                    command_id=cmd_id,
                    task_id=task_id,
                    response=GenerationResponse(
                        token=tokens_gen,
                        text=f"token{tokens_gen}",
                        finish_reason=finish,
                        usage=None,
                    ),
                )
            )
            if finish:
                done.append(uid)
            else:
                self._active_requests[uid] = (cmd_id, task_id, tokens_gen, max_tokens)
        for uid in done:
            del self._active_requests[uid]
        return results

    @property
    def has_active_requests(self) -> bool:
        return bool(self._active_requests)

    @property
    def active_count(self) -> int:
        return len(self._active_requests)

    def sync_completions(self) -> None:
        pass

    @property
    def is_distributed(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Mock tokenizers
# ---------------------------------------------------------------------------


class MockTokenizer:
    tool_parser = None
    tool_call_start = None
    tool_call_end = None
    has_tool_calling = False


class MockToolTokenizer:
    """Tokenizer with tool calling enabled for testing."""

    has_tool_calling = True
    tool_call_start = "<tool>"
    tool_call_end = "</tool>"

    @staticmethod
    def _tool_parser(text: str) -> dict[str, Any]:
        return json.loads(text)


class FakeGroup:
    def rank(self) -> int:
        return 0

    def size(self) -> int:
        return 1


# ---------------------------------------------------------------------------
# Event collector & runner helper
# ---------------------------------------------------------------------------


class EventCollector:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def send(self, event: Event) -> None:
        self.events.append(event)

    def close(self) -> None:
        pass

    def join(self) -> None:
        pass


def make_nothin[T, U, V](res: T):
    def nothin(*_1: U, **_2: V) -> T:
        return res

    return nothin


INIT_TASK = ConnectToGroup(task_id=TaskId("init"), instance_id=INSTANCE_1_ID)
LOAD_TASK = LoadModel(task_id=TaskId("load"), instance_id=INSTANCE_1_ID)
WARMUP_TASK = StartWarmup(task_id=TaskId("warmup"), instance_id=INSTANCE_1_ID)
SETUP_TASKS: list[Task] = [INIT_TASK, LOAD_TASK, WARMUP_TASK]


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


def _run_with_tasks(
    tasks: list[Task],
    engine_cls: type = FakeBatchEngineWithTokens,
    tokenizer_cls: type = MockTokenizer,
    engine_instance: Any | None = None,
) -> list[Event]:
    """Run tasks through the runner with configurable engine and tokenizer."""
    bound = get_bound_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        runner_id=RUNNER_1_ID,
        node_id=NodeId(NODE_A),
    )
    task_sender, task_receiver = mp_channel[Task]()
    collector = EventCollector()
    shutdown = Shutdown(
        task_id=TaskId("shutdown"),
        instance_id=INSTANCE_1_ID,
        runner_id=RUNNER_1_ID,
    )

    import exo.worker.runner.runner as r

    orig_init_mlx = r.initialize_mlx
    orig_load = r.load_mlx_items
    orig_warmup = r.warmup_inference
    orig_check = r._check_for_debug_prompts
    orig_engine = r.BatchGenerationEngine

    r.initialize_mlx = make_nothin(FakeGroup())
    r.load_mlx_items = make_nothin((MagicMock(), tokenizer_cls))
    r.warmup_inference = make_nothin(1)
    r._check_for_debug_prompts = make_nothin(None)
    if engine_instance is not None:
        r.BatchGenerationEngine = lambda *_a, **_kw: engine_instance  # pyright: ignore[reportUnknownLambdaType]
    else:
        r.BatchGenerationEngine = engine_cls

    try:
        with task_sender:
            for t in tasks:
                task_sender.send(t)
            task_sender.send(shutdown)
            task_receiver.close = lambda: None
            task_receiver.join = lambda: None
            r.main(bound, collector, task_receiver)  # pyright: ignore[reportArgumentType]
    finally:
        r.initialize_mlx = orig_init_mlx
        r.load_mlx_items = orig_load
        r.warmup_inference = orig_warmup
        r._check_for_debug_prompts = orig_check
        r.BatchGenerationEngine = orig_engine

    return collector.events


# ---------------------------------------------------------------------------
# Helpers for querying events
# ---------------------------------------------------------------------------


def chunks_for(events: list[Event], command_id: str) -> list[ChunkGenerated]:
    return [
        e
        for e in events
        if isinstance(e, ChunkGenerated) and e.command_id == CommandId(command_id)
    ]


def completed_task_ids(events: list[Event]) -> set[TaskId]:
    return {
        e.task_id
        for e in events
        if isinstance(e, TaskStatusUpdated) and e.task_status == TaskStatus.Complete
    }


# ===========================================================================
# Test 1: Concurrent requests with overlapping tool calls
# ===========================================================================


def test_concurrent_tool_calls_and_normal_text():
    """Two concurrent requests: one emits normal text, the other a tool call.

    Verifies that:
    - The normal request produces TokenChunks with its text
    - The tool-call request produces a ToolCallChunk
    - Both tasks complete
    """
    engine = ScriptedBatchEngine()
    # cmd_normal: 2 normal tokens then stop
    engine.scripts["cmd_normal"] = [
        ("hello", None),
        (" world", "stop"),
    ]
    # cmd_tool: tool_start, body, tool_end (suppressed), then finish
    engine.scripts["cmd_tool"] = [
        ("<tool>", None),  # swallowed by tracker
        ('{"name":"get_weather","arguments":{"city":"SF"}}', None),  # accumulated
        ("</tool>", None),  # triggers ToolCallChunk emission
        ("done", "stop"),  # normal trailing token
    ]

    chat_normal = make_chat_task("t_normal", "cmd_normal", max_tokens=100)
    chat_tool = make_chat_task("t_tool", "cmd_tool", max_tokens=100)

    events = _run_with_tasks(
        [*SETUP_TASKS, chat_normal, chat_tool],
        tokenizer_cls=MockToolTokenizer,
        engine_instance=engine,
    )

    # Normal request: all chunks should be TokenChunk
    normal_chunks = chunks_for(events, "cmd_normal")
    assert len(normal_chunks) == 2
    assert all(isinstance(c.chunk, TokenChunk) for c in normal_chunks)
    assert normal_chunks[-1].chunk.finish_reason == "stop"

    # Tool-call request
    tool_chunks = chunks_for(events, "cmd_tool")
    # <tool> → swallowed, body → accumulated, </tool> → ToolCallChunk, "done" → TokenChunk
    tool_call_events = [c for c in tool_chunks if isinstance(c.chunk, ToolCallChunk)]
    token_events = [c for c in tool_chunks if isinstance(c.chunk, TokenChunk)]

    assert len(tool_call_events) == 1, (
        f"Expected 1 ToolCallChunk, got {len(tool_call_events)}"
    )
    tc_chunk = tool_call_events[0].chunk
    assert isinstance(tc_chunk, ToolCallChunk)
    assert tc_chunk.tool_calls[0].name == "get_weather"
    assert json.loads(tc_chunk.tool_calls[0].arguments) == {"city": "SF"}

    assert len(token_events) == 1, "Expected 1 trailing TokenChunk after tool call"
    assert token_events[0].chunk.finish_reason == "stop"

    # Both tasks should complete
    done = completed_task_ids(events)
    assert TaskId("t_normal") in done
    assert TaskId("t_tool") in done


def test_tool_call_interrupted_by_finish_reason():
    """Tool call in progress when finish_reason fires — partial text emitted."""
    engine = ScriptedBatchEngine()
    engine.scripts["cmd1"] = [
        ("<tool>", None),
        ('{"name":"f"', "stop"),  # finish while inside tool call
    ]

    chat = make_chat_task("t1", "cmd1", max_tokens=100)
    events = _run_with_tasks(
        [*SETUP_TASKS, chat],
        tokenizer_cls=MockToolTokenizer,
        engine_instance=engine,
    )

    chunks = chunks_for(events, "cmd1")
    assert len(chunks) == 1
    chunk = chunks[0].chunk
    assert isinstance(chunk, TokenChunk)
    # The interrupted tool call should be emitted as raw text
    assert "<tool>" in chunk.text
    assert '{"name":"f"' in chunk.text
    assert chunk.finish_reason == "stop"

    assert TaskId("t1") in completed_task_ids(events)


# ===========================================================================
# Test 2: Request finishing with 'length' reason (timeout mid-generation)
# ===========================================================================


def test_request_finishes_with_length_reason():
    """Request that hits max_tokens limit and finishes with 'length'."""
    engine = ScriptedBatchEngine()
    engine.scripts["cmd1"] = [
        ("tok1", None),
        ("tok2", None),
        ("tok3", "length"),  # hit the token limit
    ]

    chat = make_chat_task("t1", "cmd1", max_tokens=100)
    events = _run_with_tasks(
        [*SETUP_TASKS, chat],
        engine_instance=engine,
    )

    chunks = chunks_for(events, "cmd1")
    assert len(chunks) == 3

    # Last chunk should have finish_reason="length"
    assert isinstance(chunks[-1].chunk, TokenChunk)
    assert chunks[-1].chunk.finish_reason == "length"

    # Earlier chunks should have no finish_reason
    for c in chunks[:-1]:
        assert isinstance(c.chunk, TokenChunk)
        assert c.chunk.finish_reason is None

    assert TaskId("t1") in completed_task_ids(events)


def test_mixed_finish_reasons_across_requests():
    """Two requests finishing with different reasons: 'stop' and 'length'."""
    engine = ScriptedBatchEngine()
    engine.scripts["cmd_stop"] = [("a", None), ("b", "stop")]
    engine.scripts["cmd_len"] = [("x", None), ("y", "length")]

    chat1 = make_chat_task("t_stop", "cmd_stop", max_tokens=100)
    chat2 = make_chat_task("t_len", "cmd_len", max_tokens=100)

    events = _run_with_tasks(
        [*SETUP_TASKS, chat1, chat2],
        engine_instance=engine,
    )

    stop_chunks = chunks_for(events, "cmd_stop")
    len_chunks = chunks_for(events, "cmd_len")

    assert stop_chunks[-1].chunk.finish_reason == "stop"
    assert len_chunks[-1].chunk.finish_reason == "length"

    done = completed_task_ids(events)
    assert TaskId("t_stop") in done
    assert TaskId("t_len") in done


# ===========================================================================
# Test 3: Multiple finish reasons in rapid succession (same step)
# ===========================================================================


def test_all_requests_finish_on_same_step():
    """Three requests that all finish on the same step() call.

    This tests that the runner and _process_generation_results correctly
    handle multiple completions in a single step.
    """
    engine = ScriptedBatchEngine()
    # All three produce exactly 1 token and finish
    engine.scripts["cmd_a"] = [("alpha", "stop")]
    engine.scripts["cmd_b"] = [("beta", "stop")]
    engine.scripts["cmd_c"] = [("gamma", "stop")]

    tasks = [
        *SETUP_TASKS,
        make_chat_task("ta", "cmd_a", max_tokens=100),
        make_chat_task("tb", "cmd_b", max_tokens=100),
        make_chat_task("tc", "cmd_c", max_tokens=100),
    ]
    events = _run_with_tasks([*tasks], engine_instance=engine)

    for cmd_id, expected_text in [
        ("cmd_a", "alpha"),
        ("cmd_b", "beta"),
        ("cmd_c", "gamma"),
    ]:
        c = chunks_for(events, cmd_id)
        assert len(c) == 1, f"Expected 1 chunk for {cmd_id}, got {len(c)}"
        assert isinstance(c[0].chunk, TokenChunk)
        assert c[0].chunk.text == expected_text
        assert c[0].chunk.finish_reason == "stop"

    done = completed_task_ids(events)
    assert TaskId("ta") in done
    assert TaskId("tb") in done
    assert TaskId("tc") in done

    # Runner should be back to RunnerReady after all completions
    last_status = [
        e
        for e in events
        if isinstance(e, RunnerStatusUpdated)
        and not isinstance(e.runner_status, (RunnerRunning,))
    ]
    ready_after_gen = [
        e for e in last_status if isinstance(e.runner_status, RunnerReady)
    ]
    assert len(ready_after_gen) >= 2, (
        "Expected RunnerReady after warmup and after generation completes"
    )


def test_staggered_completions_in_batch():
    """Four requests with different token counts — they complete at different steps.

    Verifies each request gets the right number of chunks and the runner
    tracks active_requests correctly as requests drain.
    """
    engine = ScriptedBatchEngine()
    engine.scripts["c1"] = [("a", "stop")]  # finishes step 1
    engine.scripts["c2"] = [("a", None), ("b", "stop")]  # finishes step 2
    engine.scripts["c3"] = [("a", None), ("b", None), ("c", "stop")]  # finishes step 3
    engine.scripts["c4"] = [
        ("a", None),
        ("b", None),
        ("c", None),
        ("d", "stop"),
    ]  # finishes step 4

    tasks = [
        *SETUP_TASKS,
        make_chat_task("t1", "c1", max_tokens=100),
        make_chat_task("t2", "c2", max_tokens=100),
        make_chat_task("t3", "c3", max_tokens=100),
        make_chat_task("t4", "c4", max_tokens=100),
    ]
    events = _run_with_tasks([*tasks], engine_instance=engine)

    assert len(chunks_for(events, "c1")) == 1
    assert len(chunks_for(events, "c2")) == 2
    assert len(chunks_for(events, "c3")) == 3
    assert len(chunks_for(events, "c4")) == 4

    done = completed_task_ids(events)
    for tid in ["t1", "t2", "t3", "t4"]:
        assert TaskId(tid) in done, f"Task {tid} should be complete"


# ===========================================================================
# Test 4: Batch of 5+ simultaneous completions
# ===========================================================================


@pytest.fixture
def patch_batch_engine(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(mlx_runner, "initialize_mlx", make_nothin(FakeGroup()))
    monkeypatch.setattr(
        mlx_runner, "load_mlx_items", make_nothin((MagicMock(), MockTokenizer))
    )
    monkeypatch.setattr(mlx_runner, "warmup_inference", make_nothin(1))
    monkeypatch.setattr(mlx_runner, "_check_for_debug_prompts", make_nothin(None))
    monkeypatch.setattr(mlx_runner, "BatchGenerationEngine", FakeBatchEngineWithTokens)


def test_five_simultaneous_completions(patch_batch_engine: None):
    """Five requests submitted together, all generating tokens and completing."""
    chats = [make_chat_task(f"t{i}", f"cmd{i}", max_tokens=2) for i in range(5)]
    events = _run_with_tasks([*SETUP_TASKS, *chats])

    for i in range(5):
        c = chunks_for(events, f"cmd{i}")
        assert len(c) == 2, f"Expected 2 chunks for cmd{i}, got {len(c)}"
        assert c[-1].chunk.finish_reason == "stop"

    done = completed_task_ids(events)
    for i in range(5):
        assert TaskId(f"t{i}") in done


def test_eight_requests_staggered(patch_batch_engine: None):
    """Eight requests with varying token counts, verifying all complete correctly."""
    chats = [make_chat_task(f"t{i}", f"cmd{i}", max_tokens=i + 1) for i in range(8)]
    events = _run_with_tasks([*SETUP_TASKS, *chats])

    for i in range(8):
        c = chunks_for(events, f"cmd{i}")
        expected = i + 1
        assert len(c) == expected, (
            f"Expected {expected} chunks for cmd{i}, got {len(c)}"
        )
        assert c[-1].chunk.finish_reason == "stop"

    done = completed_task_ids(events)
    for i in range(8):
        assert TaskId(f"t{i}") in done

    # Verify runner transitions back to ready after all requests complete
    # Find the last RunnerReady before shutdown
    ready_events = [
        (idx, e)
        for idx, e in enumerate(events)
        if isinstance(e, RunnerStatusUpdated)
        and isinstance(e.runner_status, RunnerReady)
    ]
    shutdown_idx = next(
        idx
        for idx, e in enumerate(events)
        if isinstance(e, TaskStatusUpdated)
        and e.task_id == TaskId("shutdown")
        and e.task_status == TaskStatus.Running
    )
    # There should be a RunnerReady event between generation and shutdown
    ready_before_shutdown = [idx for idx, _ in ready_events if idx < shutdown_idx]
    assert len(ready_before_shutdown) >= 1, (
        "Expected RunnerReady between generation completion and shutdown"
    )


def test_ten_simultaneous_single_token():
    """Ten requests that each produce exactly one token — all finish on step 1."""
    engine = ScriptedBatchEngine()
    for i in range(10):
        engine.scripts[f"cmd{i}"] = [(f"word{i}", "stop")]

    chats = [make_chat_task(f"t{i}", f"cmd{i}", max_tokens=100) for i in range(10)]
    events = _run_with_tasks([*SETUP_TASKS, *chats], engine_instance=engine)

    for i in range(10):
        c = chunks_for(events, f"cmd{i}")
        assert len(c) == 1
        assert isinstance(c[0].chunk, TokenChunk)
        assert c[0].chunk.text == f"word{i}"
        assert c[0].chunk.finish_reason == "stop"

    done = completed_task_ids(events)
    assert len(done & {TaskId(f"t{i}") for i in range(10)}) == 10
