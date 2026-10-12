import multiprocessing as mp
import os
import time
from typing import Any

import mlx.core as mx
import pytest
from mlx_lm.generate import generation_stream as mlx_lm_generation_stream

from exo.worker.engines.mlx.generator import generate
from exo.worker.engines.mlx.utils_mlx import use_generation_stream_by_default
from exo.worker.tests.unittests.test_mlx.conftest import create_hostfile

LAYERS = 64
WIDTH = 4096
CACHE_WIDTH = 1024
PROMPT_TOKENS = 58


def test_exo_runs_its_generation_ops_on_mlx_lm_generation_stream() -> None:
    assert generate.generation_stream is mlx_lm_generation_stream


def test_generation_stream_becomes_the_default_stream() -> None:
    previous = mx.default_stream(mx.default_device())
    try:
        use_generation_stream_by_default()
        with mx.stream(mlx_lm_generation_stream):
            generation = mx.default_stream(mx.default_device())
        assert mx.default_stream(mx.default_device()) == generation
    finally:
        mx.set_default_stream(previous)


def _prefill_from_exo_built_caches(
    rank: int,
    hostfile_path: str,
    steps: int,
    results: Any,  # pyright: ignore[reportAny]
) -> None:
    os.environ["MLX_HOSTFILE"] = hostfile_path
    os.environ["MLX_RANK"] = str(rank)
    group = mx.distributed.init(backend="ring", strict=True)
    use_generation_stream_by_default()
    weights = [
        mx.random.normal((WIDTH, WIDTH), key=mx.random.key(rank * 100 + i)) * 0.02
        for i in range(LAYERS)
    ]
    projections = [
        mx.random.normal((WIDTH, CACHE_WIDTH), key=mx.random.key(rank * 100 + 50 + i))
        * 0.02
        for i in range(LAYERS)
    ]
    cached = [
        mx.random.normal((1, 200, CACHE_WIDTH), key=mx.random.key(i))
        for i in range(LAYERS)
    ]
    prompt = mx.random.normal((PROMPT_TOKENS, WIDTH), key=mx.random.key(7))
    mx.eval(*weights, *projections, *cached, prompt)

    for _ in range(steps):
        # Work that mlx-lm left lazy on its generation stream, which exo then builds a request's
        # caches from with ops of its own (copies, trims, snapshot restores), outside any stream
        # context; the next tensor-parallel forward reads those caches on the generation stream.
        with mx.stream(mlx_lm_generation_stream):
            pending = [mx.zeros((1, 200, CACHE_WIDTH)) for _ in range(LAYERS)]
        caches = [pending[i] + cached[i] for i in range(LAYERS)]
        with mx.stream(mlx_lm_generation_stream):
            hidden = prompt
            new_caches: list[mx.array] = []
            for i in range(LAYERS):
                hidden = mx.distributed.all_sum(
                    mx.matmul(hidden, weights[i]), group=group
                )
                new_caches.append(
                    mx.concatenate(
                        [caches[i], mx.matmul(hidden, projections[i])[None]], axis=1
                    )
                )
            mx.eval(*new_caches)
    results.put(rank)  # pyright: ignore[reportAny]


@pytest.mark.slow
@pytest.mark.skipif(not mx.metal.is_available(), reason="fast synch is Metal only")
def test_tensor_parallel_prefill_from_exo_built_caches_does_not_deadlock() -> None:
    """With MLX_METAL_FAST_SYNCH, when exo's ops ran on the default stream and mlx-lm's on its
    generation stream, a forward reading caches built this way spun forever on both ranks
    (two GPU streams waiting on each other inside an eval full of collectives), on the first or
    second step."""
    world_size = 2
    hostfile_path, _ = create_hostfile(world_size, 29740)
    context = mp.get_context("spawn")
    results: Any = context.Queue()
    previous = os.environ.get("MLX_METAL_FAST_SYNCH")
    os.environ["MLX_METAL_FAST_SYNCH"] = "1"
    try:
        processes = [
            context.Process(
                target=_prefill_from_exo_built_caches,
                args=(rank, hostfile_path, 20, results),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        deadline = time.monotonic() + 120
        for process in processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        hung = [process.is_alive() for process in processes]
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join()
    finally:
        if previous is None:
            del os.environ["MLX_METAL_FAST_SYNCH"]
        else:
            os.environ["MLX_METAL_FAST_SYNCH"] = previous
        os.unlink(hostfile_path)

    assert not any(hung), f"ranks still running after 120 s: {hung}"
    assert sorted(results.get(timeout=5) for _ in range(world_size)) == [0, 1]  # pyright: ignore[reportAny]
