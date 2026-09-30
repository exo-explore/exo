"""Cancelling requests in the batch generator.

A cancelled request that is generating is finished by mlx_lm at its next step rather than
removed from the running batch (removing it between steps deadlocked pipeline-parallel
instances), and a cancelled request that hasn't started never starts.
"""

from collections import deque
from dataclasses import dataclass, field

from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.tasks import TaskId, TaskStatus, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runner_response import CancelledResponse
from exo.worker.engines.mlx.generator.batch_generate import ExoBatchGenerator
from exo.worker.runner.llm_inference.batch_generator import BatchGenerator


@dataclass
class FakeGenerationBatch:
    uids: list[int]
    max_tokens: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.max_tokens = [100] * len(self.uids)

    def __len__(self) -> int:
        return len(self.uids)


@dataclass
class FakeResponse:
    uid: int
    finish_reason: str | None


@dataclass
class FakeMlxBatchGenerator:
    generating: list[int]
    waiting: list[int]
    responses: list[FakeResponse] = field(default_factory=list)
    removed: list[int] = field(default_factory=list)
    _generation_batch: FakeGenerationBatch = field(init=False)
    _unprocessed_sequences: list[int] = field(init=False)
    _prompt_batch: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._generation_batch = FakeGenerationBatch(list(self.generating))
        self._unprocessed_sequences = list(self.waiting)

    def remove(self, uids: list[int]) -> None:
        self.removed.extend(uids)

    def next(self) -> tuple[list[object], list[FakeResponse]]:
        return [], self.responses


def engine(mlx_gen: FakeMlxBatchGenerator) -> ExoBatchGenerator:
    gen = object.__new__(ExoBatchGenerator)
    gen._mlx_gen = mlx_gen  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]
    active: dict[int, object] = {
        uid: object() for uid in [*mlx_gen.generating, *mlx_gen.waiting]
    }
    gen._active_tasks = active  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]
    gen._finishing = set()  # pyright: ignore[reportPrivateUsage]
    gen._step_count = 0  # pyright: ignore[reportPrivateUsage]
    return gen


def test_a_generating_request_is_finished_at_its_next_step_not_removed() -> None:
    mlx_gen = FakeMlxBatchGenerator(generating=[0, 1, 2], waiting=[7])
    gen = engine(mlx_gen)

    gen.cancel([1, 7])

    assert mlx_gen._generation_batch.uids == [0, 1, 2]  # pyright: ignore[reportPrivateUsage]
    assert mlx_gen._generation_batch.max_tokens == [100, 0, 100]  # pyright: ignore[reportPrivateUsage]
    assert mlx_gen.removed == [7]
    assert set(gen._active_tasks) == {0, 2}  # pyright: ignore[reportPrivateUsage]


def test_the_last_token_of_a_cancelled_request_is_discarded() -> None:
    mlx_gen = FakeMlxBatchGenerator(generating=[1], waiting=[])
    gen = engine(mlx_gen)
    gen.cancel([1])
    mlx_gen.responses = [FakeResponse(uid=1, finish_reason="length")]

    assert gen.step() == []
    assert gen._finishing == set()  # pyright: ignore[reportPrivateUsage]


def text_generation() -> TextGeneration:
    return TextGeneration(
        task_id=TaskId(),
        command_id=CommandId(),
        instance_id=InstanceId(),
        task_status=TaskStatus.Pending,
        task_params=TextGenerationTaskParams(
            model=ModelId("test-model"),
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        ),
    )


def test_a_cancelled_request_that_has_not_started_never_starts() -> None:
    queued, other = text_generation(), text_generation()
    runner_side = object.__new__(BatchGenerator)
    runner_side._queue = deque([queued, other])  # pyright: ignore[reportPrivateUsage]
    runner_side._maybe_queue = []  # pyright: ignore[reportPrivateUsage]
    runner_side._active_tasks = {}  # pyright: ignore[reportPrivateUsage]
    runner_side._cancelled_tasks = {queued.task_id}  # pyright: ignore[reportPrivateUsage]

    results = list(runner_side._apply_cancellations())  # pyright: ignore[reportPrivateUsage]

    assert results == [(queued.task_id, CancelledResponse())]
    assert list(runner_side._queue) == [other]  # pyright: ignore[reportPrivateUsage]
