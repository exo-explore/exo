# pyright: reportPrivateUsage=false
"""The task agreement split into a queued count gather and a later read, and the
batch engine's overlapped use of it during decode."""

from collections import deque
from dataclasses import dataclass, field

import mlx.core as mx
import pytest

import exo.worker.runner.llm_inference.batch_generator as mlx_batch_generator
from exo.shared.types.tasks import TaskId, TextGeneration
from exo.worker.engines.mlx.utils_mlx import (
    TaskGather,
    finish_task_gather,
    mx_all_gather_tasks,
    start_task_gather,
)
from exo.worker.runner.llm_inference.batch_generator import BatchGenerator


@dataclass
class _Task:
    task_id: TaskId


def _task(n: int) -> TextGeneration:
    return _Task(TaskId(f"00000000-0000-0000-0000-00000000000{n}"))  # pyright: ignore[reportReturnType]


def test_single_rank_start_then_finish_matches_the_waited_gather() -> None:
    tasks = [_task(2), _task(1)]
    assert finish_task_gather(start_task_gather(tasks, None), None) == (
        mx_all_gather_tasks(tasks, None)
    )
    assert finish_task_gather(start_task_gather([], None), None) == ([], [])


@dataclass
class _Engine:
    """The fields BatchGenerator's agreement methods touch."""

    group: object = None
    _maybe_queue: list[TextGeneration] = field(default_factory=list)
    _queue: deque[TextGeneration] = field(default_factory=deque)
    _task_gather: TaskGather | None = None

    # The agreement methods under test, run against these fields only.
    agree_on_tasks = BatchGenerator.agree_on_tasks
    _overlap_agreement = BatchGenerator._overlap_agreement
    _finish_task_gather = BatchGenerator._finish_task_gather


@pytest.fixture
def one_rank_gathers(monkeypatch: pytest.MonkeyPatch) -> None:
    def start(tasks: list[TextGeneration], group: object) -> TaskGather:
        return TaskGather(tasks=list(tasks), counts=mx.array([len(tasks)]))

    def finish(
        gather: TaskGather, group: object
    ) -> tuple[list[TextGeneration], list[TextGeneration]]:
        return sorted(gather.tasks, key=lambda task: task.task_id), []

    monkeypatch.setattr(mlx_batch_generator, "start_task_gather", start)
    monkeypatch.setattr(mlx_batch_generator, "finish_task_gather", finish)

    def gather(
        tasks: list[TextGeneration], group: object
    ) -> tuple[list[TextGeneration], list[TextGeneration]]:
        return finish(start(tasks, group), group)

    monkeypatch.setattr(mlx_batch_generator, "mx_all_gather_tasks", gather)


@pytest.mark.usefixtures("one_rank_gathers")
def test_overlap_agrees_one_step_later_and_keeps_late_tasks_pending() -> None:
    engine = _Engine()
    first, late = _task(1), _task(2)
    engine._maybe_queue.append(first)
    engine._overlap_agreement()
    assert list(engine._queue) == []  # started, not applied yet
    engine._maybe_queue.append(late)  # arrives while the gather is in flight
    engine._overlap_agreement()
    assert list(engine._queue) == [first]
    assert engine._maybe_queue == [late]  # not in the finished gather: stays pending
    engine._overlap_agreement()
    assert list(engine._queue) == [first, late]


@pytest.mark.usefixtures("one_rank_gathers")
def test_waited_agreement_finishes_the_overlapped_one_first() -> None:
    engine = _Engine()
    first, second = _task(1), _task(2)
    engine._maybe_queue.append(first)
    engine._overlap_agreement()
    engine._maybe_queue.append(second)
    engine.agree_on_tasks()
    assert list(engine._queue) == [first, second]
    assert engine._task_gather is None
    assert engine._maybe_queue == []
