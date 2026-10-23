"""Unit tests for Continuous Batching Chunked Prefill (TODO #7).

Verifies that incoming new requests with long prompts are prefilled in chunks,
interleaving with active generation steps so existing decoding requests
never experience stutter or head-of-line blocking.
"""

import sys
import types
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from typing import Any


# Provide lightweight mock loader for platform-specific C-extensions (MLX/PIL) when run on non-Mac
class _MockModule(types.ModuleType):
    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        if name.endswith("_token") or name in (
            "BOS_TOKEN",
            "EOS_TOKEN",
            "DSML_TOKEN",
            "THINKING_START",
            "THINKING_END",
        ):
            return ""
        if name.endswith("Model"):

            class _DummyModel:
                pass

            return _DummyModel
        m = _MockModule(f"{self.__name__}.{name}")
        setattr(self, name, m)
        return m

    def __repr__(self) -> str:
        return ""

    def __str__(self) -> str:
        return ""

    def __getitem__(self, item: Any) -> str:
        return ""

    def __or__(self, other: Any) -> Any:
        return self

    def __ror__(self, other: Any) -> Any:
        return self

    def __add__(self, other: Any) -> str:
        return str(self) + str(other)

    def __radd__(self, other: Any) -> str:
        return str(other) + str(self)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self

    def __mro_entries__(self, bases: Any) -> tuple[type, ...]:
        return (object,)


class _MockLoader:
    def __init__(self, name: str) -> None:
        self.name = name

    def create_module(self, spec: Any) -> types.ModuleType:
        m = _MockModule(self.name)
        m.__path__ = []
        return m

    def exec_module(self, module: types.ModuleType) -> None:
        pass


class _MockFinder:
    def find_spec(
        self, fullname: str, path: Any, target: Any = None
    ) -> ModuleSpec | None:
        if any(
            fullname.startswith(p)
            for p in [
                "mlx",
                "mlx_lm",
                "PIL",
                "transformers",
                "safetensors",
                "openai_harmony",
            ]
        ):
            spec = ModuleSpec(fullname, None, is_package=True)
            spec.loader = _MockLoader(fullname)
            return spec
        return None


if not any(isinstance(f, _MockFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _MockFinder())

from exo.shared.types.chunks import TokenChunk
from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.tasks import TaskId, TaskStatus, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    GenerationResponse,
)
from exo.worker.runner.llm_inference.batch_generator import (
    BatchGenerator,
    GeneratorQueue,
)


def _make_text_generation(prompt: str = "hello") -> TextGeneration:
    return TextGeneration(
        task_id=TaskId(),
        command_id=CommandId(),
        instance_id=InstanceId(),
        task_status=TaskStatus.Pending,
        task_params=TextGenerationTaskParams(
            model=ModelId("test-model"),
            bench=True,
            input=[InputMessage(role="user", content=InputMessageContent(prompt))],
        ),
    )


@dataclass
class FakePrefillSession:
    task_params: TextGenerationTaskParams
    total_tokens: int
    processed: int = 0
    on_prefill_progress: Callable[[int, int], None] | None = None
    distributed_prompt_progress_callback: Callable[[], None] | None = None
    on_generation_token: Callable[[], None] | None = None
    uid: int = 0


class FakeChunkedExoBatchGenerator:
    """Mock generator simulating chunked prefill and interleaved decode steps."""

    def __init__(self) -> None:
        self.uid_counter = 0
        self.active_decode_uids: set[int] = set()
        self.prefill_sessions: dict[int, FakePrefillSession] = {}
        self.step_history: list[str] = []
        self.cancelled_uids: list[int] = []

    @property
    def has_work(self) -> bool:
        return bool(self.active_decode_uids)

    def prepare_prefill(
        self,
        task_params: TextGenerationTaskParams,
        prompt: str,
        on_prefill_progress: Callable[[int, int], None] | None = None,
        distributed_prompt_progress_callback: Callable[[], None] | None = None,
        on_generation_token: Callable[[], None] | None = None,
    ) -> FakePrefillSession:
        uid = self.uid_counter
        self.uid_counter += 1
        content = ""
        if task_params.input and hasattr(task_params.input[0], "content"):
            content = str(task_params.input[0].content)
        total_tokens = 2000 if "long" in content or "long" in str(prompt) else 100
        session = FakePrefillSession(
            task_params=task_params,
            total_tokens=total_tokens,
            processed=0,
            on_prefill_progress=on_prefill_progress,
            distributed_prompt_progress_callback=distributed_prompt_progress_callback,
            on_generation_token=on_generation_token,
            uid=uid,
        )
        self.prefill_sessions[uid] = session
        return session

    def step_prefill(
        self, session: FakePrefillSession, chunk_size: int = 512
    ) -> tuple[bool, int | None]:
        end = min(session.processed + chunk_size, session.total_tokens)
        session.processed = end
        self.step_history.append(
            f"prefill uid={session.uid} [{session.processed}/{session.total_tokens}]"
        )

        if session.on_prefill_progress is not None:
            session.on_prefill_progress(session.processed, session.total_tokens)

        if session.processed >= session.total_tokens:
            self.active_decode_uids.add(session.uid)
            return True, session.uid
        return False, None

    def step(self) -> list[tuple[int, GenerationResponse]]:
        results: list[tuple[int, GenerationResponse]] = []
        for uid in list(self.active_decode_uids):
            self.step_history.append(f"decode uid={uid}")
            results.append(
                (uid, GenerationResponse(text=f"tok_{uid} ", token=uid, usage=None))
            )
        return results

    def cancel(self, uids: list[int]) -> None:
        self.cancelled_uids.extend(uids)
        for uid in uids:
            self.active_decode_uids.discard(uid)
            self.prefill_sessions.pop(uid, None)

    def close(self) -> None:
        self.active_decode_uids.clear()
        self.prefill_sessions.clear()


def _setup_runner(gen: Any) -> BatchGenerator:
    runner = object.__new__(BatchGenerator)
    runner._gen = gen
    runner._queue = deque()
    runner._maybe_queue = []
    runner._active_tasks = {}
    runner._cancelled_tasks = set()
    runner._prefilling_task = None
    runner.prefill_chunk_size = 512
    runner.device_rank = 0
    runner.model_id = ModelId("test-model")
    runner.model = object()
    runner.tool_parser = None
    runner.sent_events = []
    runner.event_sender = _MockModule("event_sender")
    runner.event_sender.send = lambda evt: runner.sent_events.append(evt)
    runner.tokenizer = _MockModule("tokenizer")
    runner.check_for_cancel_every = 50
    runner.should_cancel = lambda tid: tid in runner._cancelled_tasks
    runner.agree_on_tasks = lambda: None
    runner.agree_on_cancellations = lambda: None
    runner._send_error = lambda task, e: None
    return runner


from exo.worker.runner.llm_inference.model_output_parsers import (
    map_responses_to_chunks,
)


def test_chunked_prefill_does_not_block_active_decode_batch() -> None:
    """When a new request with a 2,000 token prompt arrives while Task A is decoding,
    Task A must decode a token at EVERY step while Task B prefills in 512-token chunks.
    """
    gen = FakeChunkedExoBatchGenerator()
    runner = _setup_runner(gen)

    # 1. Start Task A and make it active (decoding)
    task_a = _make_text_generation(prompt="short prompt")
    queue_a = GeneratorQueue[GenerationResponse]()
    out_gen_a = (map_responses_to_chunks(r, runner.model_id) for r in queue_a.gen())
    uid_a = 0
    gen.active_decode_uids.add(uid_a)
    gen.uid_counter = 1
    runner._active_tasks[uid_a] = (task_a, queue_a, out_gen_a)

    # 2. Queue Task B with a long prompt (2,000 tokens => requires 4 chunks of 512)
    task_b = _make_text_generation(prompt="long prompt with many tokens")
    runner._queue.append(task_b)

    # 3. Step 1:
    # Task B starts chunked prefill (0 -> 512).
    # Task A MUST decode without being blocked!
    step1_results = list(runner.step())
    assert runner._prefilling_task is not None
    assert runner._prefilling_task[1].processed == 512
    # Verify Task A yielded its token chunk!
    task_a_chunks = [c for tid, c in step1_results if tid == task_a.task_id]
    assert len(task_a_chunks) == 1
    assert isinstance(task_a_chunks[0], TokenChunk)

    # 4. Step 2:
    # Task B prefills chunk 2 (512 -> 1024).
    # Task A decodes token 2!
    step2_results = list(runner.step())
    assert runner._prefilling_task[1].processed == 1024
    task_a_chunks = [c for tid, c in step2_results if tid == task_a.task_id]
    assert len(task_a_chunks) == 1

    # 5. Step 3:
    # Task B prefills chunk 3 (1024 -> 1536).
    # Task A decodes token 3!
    step3_results = list(runner.step())
    assert runner._prefilling_task[1].processed == 1536
    task_a_chunks = [c for tid, c in step3_results if tid == task_a.task_id]
    assert len(task_a_chunks) == 1

    # 6. Step 4:
    # Task B prefills chunk 4 (1536 -> 2000). Prefill completes!
    # Task B joins _active_tasks and decodes its first token!
    step4_results = list(runner.step())
    assert runner._prefilling_task is None  # Prefill completed
    assert len(runner._active_tasks) == 2  # Both Task A and Task B now active!
    task_a_chunks = [c for tid, c in step4_results if tid == task_a.task_id]
    assert len(task_a_chunks) == 1

    # 7. Step 5:
    # Now both Task A and Task B decode together in the batch!
    step5_results = list(runner.step())
    tids = [tid for tid, _ in step5_results]
    assert task_a.task_id in tids
    assert task_b.task_id in tids


def test_cancellation_during_chunked_prefill() -> None:
    """If a task is cancelled while its prompt is being prefilled, it cancels immediately
    and does not enter the generation batch.
    """
    gen = FakeChunkedExoBatchGenerator()
    runner = _setup_runner(gen)

    # Active decode task present so prefill chunk size is runner.prefill_chunk_size (512)
    task_a = _make_text_generation(prompt="short prompt")
    queue_a = GeneratorQueue[GenerationResponse]()
    out_gen_a = (map_responses_to_chunks(r, runner.model_id) for r in queue_a.gen())
    uid_a = 0
    gen.active_decode_uids.add(uid_a)
    gen.uid_counter = 1
    runner._active_tasks[uid_a] = (task_a, queue_a, out_gen_a)

    task_b = _make_text_generation(prompt="long prompt")
    runner._queue.append(task_b)

    # Step 1: Begins prefill (0 -> 512)
    step1_results = list(runner.step())
    assert runner._prefilling_task is not None
    assert runner._prefilling_task[1].processed == 512
    assert any(tid == task_a.task_id for tid, _ in step1_results)

    # Cancel task_b
    runner._cancelled_tasks.add(task_b.task_id)

    # Step 2: Should cancel the prefilling task without affecting Task A
    step2_results = list(runner.step())
    assert any(
        tid == task_b.task_id and isinstance(r, CancelledResponse)
        for tid, r in step2_results
    )
    assert runner._prefilling_task is None
    assert set(runner._active_tasks.keys()) == {uid_a}


def test_legacy_engine_fallback_without_chunking() -> None:
    """An engine without prepare_prefill falls back to synchronous submit cleanly."""

    class LegacyFakeGenerator:
        def __init__(self) -> None:
            self.has_work = False

        def submit(self, *args: Any, **kwargs: Any) -> int:
            return 99

        def step(self) -> list[Any]:
            return []

        def cancel(self, uids: list[int]) -> None:
            pass

    gen = LegacyFakeGenerator()
    runner = _setup_runner(gen)

    task = _make_text_generation()
    runner._queue.append(task)

    runner._start_task = lambda t: 99
    list(runner.step())
    assert 99 in runner._active_tasks
