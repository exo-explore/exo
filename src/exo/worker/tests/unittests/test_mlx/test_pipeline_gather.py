"""Outside prefill, a pipeline stage gathers the last stage's output from every rank and
keeps its own copy. It must wait for the gather on the CPU, on its own: a GPU command
buffer that waited for it would wait for the slowest rank, and the GPU watchdog (about
5 s on Apple silicon) fails command buffers that wait longer ("Caused GPU Timeout
Error"), e.g. while the last stage is still compiling its kernels or computing a long
chunk.
"""

import multiprocessing as mp
import os
import time
from typing import Any, cast

import mlx.core as mx
import pytest

from exo.worker.engines.mlx.sharding.pipeline import (
    PipelineFirstLayer,
    PipelineLastLayer,
)
from exo.worker.engines.mlx.utils_mlx import (
    mx_barrier,
    use_generation_stream_by_default,
)
from exo.worker.tests.unittests.test_mlx.conftest import MockLayer, create_hostfile

# Longer than the GPU watchdog
LAST_STAGE_DELAY = 8.0


def test_a_stage_evaluates_the_gather_before_slicing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluated: list[tuple[int, ...]] = []
    real_eval = mx.eval

    def recording_eval(*arrays: object) -> None:
        evaluated.extend(tuple(a.shape) for a in arrays if isinstance(a, mx.array))
        real_eval(*arrays)  # pyright: ignore[reportArgumentType]

    def gather_two_ranks(x: mx.array, group: object = None) -> mx.array:
        # Rank 0's rows, then the last rank's: a stage keeps the last rank's.
        return mx.concatenate([x * 0, x + 1])

    def send(x: mx.array, dst: int, group: object = None) -> mx.array:
        return x

    monkeypatch.setattr(mx, "eval", recording_eval)
    monkeypatch.setattr(mx.distributed, "all_gather", gather_two_ranks)
    monkeypatch.setattr(mx.distributed, "send", send)

    layer = PipelineLastLayer(MockLayer(), r=0, s=2, group=mx.distributed.init())
    output = layer(mx.ones((1, 4)))

    assert output.tolist() == [[3.0, 3.0, 3.0, 3.0]]
    # The (2, 4) gather on its own, then only the (1, 4) slice of it.
    gathered = evaluated.index((2, 4))
    assert evaluated[gathered + 1 :] == [(1, 4)]


class _LateLayer(MockLayer):
    """The last stage, still busy when the first stage reaches the gather."""

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        time.sleep(LAST_STAGE_DELAY)
        return x * 2


def _run_rank(
    rank: int,
    hostfile_path: str,
    results: Any,  # pyright: ignore[reportAny]
) -> None:
    os.environ["MLX_HOSTFILE"] = hostfile_path
    os.environ["MLX_RANK"] = str(rank)
    try:
        group = mx.distributed.init(backend="ring", strict=True)
        use_generation_stream_by_default()
        inner = _LateLayer() if rank == group.size() - 1 else MockLayer()
        layer = PipelineLastLayer(
            PipelineFirstLayer(inner, r=rank, group=group),
            r=rank,
            s=group.size(),
            group=group,
        )
        mx_barrier(group)
        output = layer(mx.full((1, 1, 4), 1.0))
        mx.eval(output)
        results.put((rank, True, output.tolist()))  # pyright: ignore[reportAny]
    except Exception as e:
        results.put((rank, False, repr(e)))  # pyright: ignore[reportAny]


@pytest.mark.slow
@pytest.mark.skipif(not mx.metal.is_available(), reason="the GPU watchdog is Metal's")
def test_a_last_stage_slower_than_the_gpu_watchdog_does_not_fail_the_others() -> None:
    world_size = 2
    hostfile_path, _ = create_hostfile(world_size, 29570)
    context = mp.get_context("spawn")
    results: Any = context.Queue()
    processes = [
        context.Process(target=_run_rank, args=(rank, hostfile_path, results))
        for rank in range(world_size)
    ]
    try:
        for process in processes:
            process.start()
        outcomes: dict[int, tuple[bool, object]] = {}
        for _ in range(world_size):
            rank, ok, value = cast(
                tuple[int, bool, object],
                results.get(timeout=120),  # pyright: ignore[reportAny]
            )
            outcomes[rank] = (ok, value)
    finally:
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()
        os.unlink(hostfile_path)
    # Each rank doubles its input; every rank ends with the last rank's output.
    assert outcomes == {0: (True, [[[4.0] * 4]]), 1: (True, [[[4.0] * 4]])}
