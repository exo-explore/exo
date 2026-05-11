"""Resilience tests for :class:`SequentialGenerator`.

Regression coverage for PR #15: a per-task ``ValueError`` raised during
drafter construction (e.g. K above the transport's wire-protocol budget)
must not propagate out of ``step()`` and crash the runner subprocess.
The pre-fix behaviour was that ``_start_next`` re-raised after sending
the error chunk, which propagated through ``handle_generation_tasks``
and triggered ``RunnerFailed`` on the supervisor, leaving the peer rank
wedged in ``RunnerRunning`` while the respawned target sat in
``RunnerIdle`` forever.

These tests bypass the SequentialGenerator dataclass __init__ (which
needs a full MLX model + tokenizer stack) and patch only the failing
hot-spot, mirroring the pattern used by ``test_batch_generator_errors``.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Iterator
from typing import Any, cast

import pytest

from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.events import ChunkGenerated, Event
from exo.shared.types.tasks import TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId
from exo.utils.channels import MpSender
from exo.worker.runner.llm_inference.batch_generator import (
    FinishedResponse,
    GeneratorQueue,
    SequentialGenerator,
)


class _FakeEventSender:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def send(self, event: Event) -> None:
        self.events.append(event)


def _make_text_task(text: str = "hello", bench: bool = False) -> TextGeneration:
    return TextGeneration(
        instance_id=InstanceId("instance"),
        command_id=CommandId(f"command-{text}"),
        task_params=TextGenerationTaskParams(
            model=ModelId("mlx-community/test-model"),
            input=[
                InputMessage(role="user", content=InputMessageContent(text)),
            ],
            bench=bench,
        ),
    )


def _bare_sequential_generator(
    sender: _FakeEventSender,
    queue: deque[TextGeneration],
) -> SequentialGenerator:
    """Construct a :class:`SequentialGenerator` without running its dataclass init.

    Only the attributes touched by ``step()`` / ``_start_next()`` /
    ``_send_error()`` are wired in, so the test stays MLX-free and focused
    on the resilience contract.
    """
    generator = object.__new__(SequentialGenerator)
    generator.model_id = ModelId("mlx-community/test-model")
    generator.device_rank = 0
    generator.tokenizer = cast(Any, object())
    generator.event_sender = cast(MpSender[Event], cast(object, sender))
    generator.group = None
    generator._maybe_queue = []  # pyright: ignore[reportPrivateUsage]
    generator._maybe_cancel = []  # pyright: ignore[reportPrivateUsage]
    generator._all_tasks = {  # pyright: ignore[reportPrivateUsage]
        task.task_id: task for task in queue
    }
    generator._queue = queue  # pyright: ignore[reportPrivateUsage]
    generator._cancelled_tasks = set()  # pyright: ignore[reportPrivateUsage]
    generator._active_tasks = OrderedDict()  # pyright: ignore[reportPrivateUsage]
    generator._pending_failed = []  # pyright: ignore[reportPrivateUsage]
    generator._recent_acceptance = deque()  # pyright: ignore[reportPrivateUsage]
    generator.adaptive_draft_tokens = False
    generator.max_concurrent_tasks = 1
    return generator


def test_start_next_failure_emits_finished_and_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drafter construction failure must surface as ``FinishedResponse``."""
    sender = _FakeEventSender()
    task = _make_text_task("first")
    generator = _bare_sequential_generator(sender, deque([task]))

    def boom(_self: SequentialGenerator, _task: TextGeneration) -> None:
        raise ValueError("num_draft_tokens (8) exceeds transport's max (5)")

    def no_agree(_self: SequentialGenerator) -> None:
        return None

    monkeypatch.setattr(
        SequentialGenerator,
        "_build_generator",
        boom,
    )
    monkeypatch.setattr(
        SequentialGenerator,
        "agree_on_tasks",
        no_agree,
    )

    results = list(generator.step())

    assert len(results) >= 1
    assert results[0][0] == task.task_id
    assert isinstance(results[0][1], FinishedResponse)
    assert (
        len(generator._active_tasks) == 0  # pyright: ignore[reportPrivateUsage]
    ), "no active task should be set after failed _start_next"
    assert len(sender.events) == 1
    assert isinstance(sender.events[0], ChunkGenerated)
    assert isinstance(sender.events[0].chunk, ErrorChunk)
    assert "num_draft_tokens" in sender.events[0].chunk.error_message


def test_runner_survives_sequential_failure_and_serves_next_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a per-task failure the runner must still serve the next task.

    This is the core regression: pre-fix, the first task's failure
    propagated out of ``step()`` and tore down the runner subprocess, so
    the second task never got a chance to run. We use two failing tasks
    so the test stays MLX-free; what matters is that ``step()`` survives
    both failures and surfaces them as ``FinishedResponse`` rather than
    propagating an exception out of the runner loop.

    Post-concurrency-refactor (PR #15 round-robin), ``step`` drains the
    queue up to ``max_concurrent_tasks`` per tick rather than admitting
    one task per tick, so both failures may surface on tick 1. The
    contract that matters is unchanged: every queued task must reach
    ``_build_generator`` and surface a ``FinishedResponse`` without
    raising.
    """
    sender = _FakeEventSender()
    first = _make_text_task("first")
    second = _make_text_task("second")
    generator = _bare_sequential_generator(sender, deque([first, second]))

    call_log: list[str] = []

    def boom(_self: SequentialGenerator, task: TextGeneration) -> object:
        call_log.append(str(task.task_id))
        raise ValueError("num_draft_tokens (8) exceeds transport's max (5)")

    def no_agree(_self: SequentialGenerator) -> None:
        return None

    monkeypatch.setattr(
        SequentialGenerator,
        "_build_generator",
        boom,
    )
    monkeypatch.setattr(
        SequentialGenerator,
        "agree_on_tasks",
        no_agree,
    )

    finished_task_ids: set[Any] = set()
    while finished_task_ids != {first.task_id, second.task_id}:
        produced = list(generator.step())
        for task_id, response in produced:
            if isinstance(response, FinishedResponse):
                finished_task_ids.add(task_id)
        # Guard the loop: with max_concurrent_tasks=1 (helper default)
        # this finishes in one or two ticks; if step() ever loops without
        # progress the runner has regressed and we want a hard fail.
        if not produced and not generator._queue and not generator._pending_failed:  # pyright: ignore[reportPrivateUsage]
            break

    assert finished_task_ids == {first.task_id, second.task_id}, (
        "both tasks must surface as FinishedResponse"
    )
    assert call_log == [str(first.task_id), str(second.task_id)], (
        "both tasks must reach _build_generator -- pre-fix the first "
        "failure propagated and the second task never got a chance"
    )
    assert len(sender.events) == 2, "both failures must emit ErrorChunks"


def test_round_robin_advances_all_active_tasks_per_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``max_concurrent_tasks > 1`` must advance every active task per ``step``.

    Pre-fix, ``SequentialGenerator._active`` was a singular slot and slot
    1's TTFT equalled slot 0's *full* completion time -- the 14s figure
    measured in the PR #15 concurrency leg. The fix admits up to
    ``max_concurrent_tasks`` simultaneous in-flight tasks and round-
    robins one ``next(gen)`` per task per ``step``, so slot 1's TTFT is
    bounded by its own prefill plus a constant number of slot-0 token
    times. We assert the contract (both tasks make progress on the same
    tick) without standing up an MLX model.
    """
    sender = _FakeEventSender()
    # ``bench=True`` short-circuits the parser pipeline so ``_start_next``
    # never touches ``tokenizer.apply_chat_template`` -- the test stays
    # focused on the round-robin contract.
    first = _make_text_task("first", bench=True)
    second = _make_text_task("second", bench=True)
    generator = _bare_sequential_generator(sender, deque([first, second]))
    generator.max_concurrent_tasks = 2

    yielded_per_task: dict[Any, int] = {first.task_id: 0, second.task_id: 0}

    def fake_build(
        _self: SequentialGenerator, task: TextGeneration
    ) -> Iterator[object]:
        # Each generator yields a sentinel object three times so we can
        # observe round-robin progression without depending on MLX. The
        # parsed-output generator is an empty iterator -- ``step`` is
        # tested through its bookkeeping (``_active_tasks`` membership,
        # task progress), not through chunk emission.
        def gen() -> Iterator[object]:
            for _ in range(3):
                yielded_per_task[task.task_id] += 1
                yield object()

        return gen()

    def no_agree(_self: SequentialGenerator) -> None:
        return None

    monkeypatch.setattr(SequentialGenerator, "_build_generator", fake_build)
    monkeypatch.setattr(SequentialGenerator, "agree_on_tasks", no_agree)

    list(generator.step())

    assert yielded_per_task[first.task_id] == 1, (
        "first task must advance one token on tick 1"
    )
    assert yielded_per_task[second.task_id] == 1, (
        "second task must ALSO advance one token on tick 1 -- this is "
        "the round-robin contract; pre-fix it would have been 0 because "
        "the singular ``_active`` slot was held by the first task"
    )
    assert (
        len(generator._active_tasks) == 2  # pyright: ignore[reportPrivateUsage]
    ), "both tasks must be in the active set"


def test_round_robin_respects_max_concurrent_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``max_concurrent_tasks=1`` (asymmetric default) must stay singular.

    ``RemoteTransport``'s wire protocol is per-session, so the asymmetric
    placement leaves ``max_concurrent_tasks`` at 1 at builder time. This
    test asserts the cap is honoured in ``step``: with two queued tasks
    and a cap of 1, only the first is admitted; the second waits until
    the first retires.
    """
    sender = _FakeEventSender()
    first = _make_text_task("first", bench=True)
    second = _make_text_task("second", bench=True)
    generator = _bare_sequential_generator(sender, deque([first, second]))
    generator.max_concurrent_tasks = 1

    admitted_order: list[Any] = []

    def fake_build(
        _self: SequentialGenerator, task: TextGeneration
    ) -> Iterator[object]:
        admitted_order.append(task.task_id)

        # Generator yields once then exhausts on the next ``next()``.
        def gen() -> Iterator[object]:
            yield object()

        return gen()

    def no_agree(_self: SequentialGenerator) -> None:
        return None

    monkeypatch.setattr(SequentialGenerator, "_build_generator", fake_build)
    monkeypatch.setattr(SequentialGenerator, "agree_on_tasks", no_agree)

    # Tick 1: cap=1 admits only the first task; second remains queued.
    list(generator.step())
    assert admitted_order == [first.task_id], (
        "only the first task may be admitted when cap=1"
    )
    assert (
        first.task_id in generator._active_tasks  # pyright: ignore[reportPrivateUsage]
    ), "first task is mid-stream after one yield"
    assert len(generator._queue) == 1, (  # pyright: ignore[reportPrivateUsage]
        "second task must remain queued under cap=1"
    )

    # Tick 2: first generator exhausts (StopIteration on second ``next``)
    # and the slot frees up; the cap-respecting top-up admits second.
    list(generator.step())
    assert admitted_order == [first.task_id, second.task_id], (
        "second task must be admitted on tick 2 after first retires"
    )
    assert (
        first.task_id not in generator._active_tasks  # pyright: ignore[reportPrivateUsage]
    ), "first task must have retired"


def test_round_robin_per_task_error_does_not_kill_other_active_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A faulty generator must finish only its own task; siblings keep advancing.

    With ``max_concurrent_tasks > 1`` a single malformed request must
    not knock peer in-flight tasks off the runner. This is a strictly
    stronger version of the K=8-cancel resilience contract.
    """
    sender = _FakeEventSender()
    good = _make_text_task("good")
    bad = _make_text_task("bad")
    generator = _bare_sequential_generator(sender, deque())
    generator.max_concurrent_tasks = 2

    good_yields = [0]

    def good_gen() -> Iterator[object]:
        for _ in range(5):
            good_yields[0] += 1
            yield object()

    class _BoomError(Exception):
        pass

    def bad_gen() -> Iterator[object]:
        raise _BoomError("doomed mid-stream")
        yield  # pyright: ignore[reportUnreachable]

    # Use real ``GeneratorQueue`` instances per task so ``queue.push``
    # in ``step`` doesn't blow up; outputs are drained via per-task
    # ``output_generator`` iterators (empty here -- the contract under
    # test is task-membership in ``_active_tasks``, not chunk content).
    generator._active_tasks[good.task_id] = (  # pyright: ignore[reportPrivateUsage]
        good,
        cast(Any, good_gen()),
        GeneratorQueue(),
        iter([]),
    )
    generator._active_tasks[bad.task_id] = (  # pyright: ignore[reportPrivateUsage]
        bad,
        cast(Any, bad_gen()),
        GeneratorQueue(),
        iter([]),
    )

    # ``cast(Any, ...)`` above is required because ``_active_tasks``
    # expects ``Generator[GenerationResponse]`` and our test stubs yield
    # plain ``object()`` to keep the test MLX-free; the stubs satisfy the
    # iterator protocol that ``next(gen)`` relies on, which is the only
    # thing ``step`` actually requires.

    def no_agree(_self: SequentialGenerator) -> None:
        return None

    monkeypatch.setattr(SequentialGenerator, "agree_on_tasks", no_agree)

    results = list(generator.step())

    assert good_yields[0] == 1, "good task must still advance on the bad-task tick"
    bad_finished = any(
        r[0] == bad.task_id and isinstance(r[1], FinishedResponse) for r in results
    )
    assert bad_finished, "bad task must surface as FinishedResponse"
    assert (
        good.task_id in generator._active_tasks  # pyright: ignore[reportPrivateUsage]
    ), "good task must remain active after sibling failure"
    assert (
        bad.task_id not in generator._active_tasks  # pyright: ignore[reportPrivateUsage]
    ), "bad task must be evicted from the active set"
    assert len(sender.events) == 1
    assert isinstance(sender.events[0], ChunkGenerated)
    assert isinstance(sender.events[0].chunk, ErrorChunk)


def test_step_exception_during_next_does_not_raise() -> None:
    """An exception during ``next(gen)`` mid-stream must surface as Finished, not crash."""
    sender = _FakeEventSender()
    task = _make_text_task()
    generator = _bare_sequential_generator(sender, deque())

    class _BoomError(Exception):
        pass

    def faulty_gen() -> Iterator[object]:
        raise _BoomError("runtime fault inside spec loop")
        yield  # pyright: ignore[reportUnreachable]

    generator._active_tasks[task.task_id] = (  # pyright: ignore[reportPrivateUsage]
        task,
        cast(Any, faulty_gen()),
        GeneratorQueue(),
        iter([]),
    )

    results = list(generator.step())

    assert any(
        result[0] == task.task_id and isinstance(result[1], FinishedResponse)
        for result in results
    )
    assert (
        len(generator._active_tasks) == 0  # pyright: ignore[reportPrivateUsage]
    )
    assert len(sender.events) == 1
    assert isinstance(sender.events[0], ChunkGenerated)
    assert isinstance(sender.events[0].chunk, ErrorChunk)
    assert "runtime fault" in sender.events[0].chunk.error_message
