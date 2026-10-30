"""A peer that falls behind by more than the GPU watchdog doesn't fail the ranks waiting on it.

macOS fails a Metal command buffer that runs for more than a few seconds. When a GPU op used the
result of an unfinished collective, its command buffer waited on the network, so a peer that fell
behind by that long (a stalled link, a busy Mac) failed the ranks waiting on it with "[METAL]
Command buffer execution failed: Caused GPU Timeout Error". Each rank now waits for a collective
on the CPU, which has no such limit, before the GPU uses its result.

Each case runs two ranks over the ring on this machine. Rank 1 holds back its part of one
all_gather for longer than the watchdog, and both ranks must come through with the right result.
"""

import multiprocessing as mp
import os
import time
from collections.abc import Callable
from typing import Any, cast

import mlx.core as mx
import pytest

from exo.shared.models.model_cards import ModelId
from exo.shared.types.tasks import TaskId, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.worker.engines.mlx.auto_parallel import PipelineFirstLayer, PipelineLastLayer
from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.mlx.utils_mlx import (
    mx_all_gather_tasks,
    mx_barrier,
    use_generation_stream_by_default,
)
from exo.worker.tests.constants import COMMAND_1_ID, INSTANCE_1_ID
from exo.worker.tests.unittests.test_mlx.conftest import MockLayer, create_hostfile

# Longer than the GPU watchdog, about 5 s on Apple silicon
PEER_DELAY = 8.0

TASK_A = TaskId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
TASK_B = TaskId("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
TASK_C = TaskId("cccccccc-cccc-4ccc-8ccc-cccccccccccc")


def _task(task_id: TaskId) -> TextGeneration:
    return TextGeneration(
        task_id=task_id,
        command_id=COMMAND_1_ID,
        instance_id=INSTANCE_1_ID,
        task_params=TextGenerationTaskParams(
            model=ModelId("test/model"),
            input=[InputMessage(role="user", content=InputMessageContent("hello"))],
        ),
    )


def _pipeline_decode_step(rank: int, group: mx.distributed.Group) -> object:
    """A decode step through a pipeline of one layer per rank, which doubles its input"""
    layer = PipelineLastLayer(
        PipelineFirstLayer(MockLayer(), r=rank, group=group),
        r=rank,
        s=group.size(),
        group=group,
    )
    output = layer(mx.ones((1, 1, 4)))
    mx.eval(output)
    return output.tolist()


def _task_agreement(rank: int, group: mx.distributed.Group) -> object:
    """Rank 0 has tasks a and b, rank 1 tasks b and c"""
    agreed, _ = mx_all_gather_tasks(
        [_task(TASK_A if rank == 0 else TASK_C), _task(TASK_B)], group
    )
    return [task.task_id for task in agreed]


def _prefix_cache_eviction_check(rank: int, group: mx.distributed.Group) -> object:
    del rank
    return KVPrefixCache(group)._over_limit(0)  # pyright: ignore[reportPrivateUsage]


CASES: dict[str, tuple[Callable[[int, mx.distributed.Group], object], int]] = {
    # The step, and which of its all_gathers rank 1 holds back
    "pipeline decode step": (_pipeline_decode_step, 1),
    # The first gather (the task counts) is read on the CPU already: hold back the second
    "task agreement": (_task_agreement, 2),
    "prefix cache eviction check": (_prefix_cache_eviction_check, 1),
}


def _hold_back_all_gather(call: int) -> None:
    """Make this rank's `call`-th all_gather join PEER_DELAY seconds late"""
    all_gather = mx.distributed.all_gather
    calls = 0

    def late_all_gather(
        x: mx.array, *, group: mx.distributed.Group | None = None
    ) -> mx.array:
        nonlocal calls
        calls += 1
        if calls == call:
            time.sleep(PEER_DELAY)
        return all_gather(x, group=group)

    setattr(mx.distributed, "all_gather", late_all_gather)  # noqa: B010


def _run_rank(
    rank: int,
    hostfile_path: str,
    case: str,
    results: Any,  # pyright: ignore[reportAny]
) -> None:
    os.environ["MLX_HOSTFILE"] = hostfile_path
    os.environ["MLX_RANK"] = str(rank)
    try:
        group = mx.distributed.init(backend="ring", strict=True)
        use_generation_stream_by_default()
        step, held_back = CASES[case]
        mx_barrier(group)
        if rank == 1:
            _hold_back_all_gather(held_back)
        results.put((rank, True, step(rank, group)))  # pyright: ignore[reportAny]
    except Exception as e:
        results.put((rank, False, repr(e)))  # pyright: ignore[reportAny]


@pytest.mark.slow
@pytest.mark.skipif(not mx.metal.is_available(), reason="the GPU watchdog is Metal's")
@pytest.mark.parametrize("case", list(CASES))
def test_a_peer_slower_than_the_gpu_watchdog_does_not_fail_the_other_rank(
    case: str,
) -> None:
    world_size = 2
    hostfile_path, _ = create_hostfile(world_size, 29520 + 2 * list(CASES).index(case))
    context = mp.get_context("spawn")
    results: Any = context.Queue()
    processes = [
        context.Process(target=_run_rank, args=(rank, hostfile_path, case, results))
        for rank in range(world_size)
    ]
    try:
        for process in processes:
            process.start()
        outcomes: dict[int, tuple[bool, object]] = {}
        for _ in range(world_size):
            rank, ok, value = cast(
                tuple[int, bool, object],
                results.get(timeout=60),  # pyright: ignore[reportAny]
            )
            outcomes[rank] = (ok, value)
    finally:
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()
        os.unlink(hostfile_path)

    failed = {rank: value for rank, (ok, value) in outcomes.items() if not ok}
    assert failed == {}
    values = [outcomes[rank][1] for rank in range(world_size)]
    # Both ranks come out with the same answer, and the right one where it is known
    assert values[0] == values[1]
    if case == "pipeline decode step":
        assert values[0] == [[[4.0, 4.0, 4.0, 4.0]]]
    if case == "task agreement":
        assert values[0] == [TASK_B]
