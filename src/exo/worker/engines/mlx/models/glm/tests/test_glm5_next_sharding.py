"""GLM-5.3-Flash split over two ranks on one machine (ring backend) with exo's tensor and
pipeline parallelism gives the same logits as one device, through chunked prefill (dense
and sparse attention) and cached decode."""

import json
import multiprocessing as mp
import os
import sys
import tempfile
from typing import Any, cast

import numpy as np
import pytest

from exo.worker.engines.mlx.models.glm.tests.glm5_next_tiny import (
    cached_logits,
    sharded_worker,
    tiny_model,
    tokens,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="MLX distributed requires Metal"
)

_SPLITS = (13, 30)  # two prefill chunks, then cached decode; sparse past 11 keys


@pytest.mark.parametrize(
    ("mode", "split_layer", "base_port"),
    [("tensor", 0, 29700), ("pipeline", 3, 29710), ("pipeline", 4, 29720)],
)
def test_two_ranks_match_one_device(
    mode: str, split_layer: int, base_port: int
) -> None:
    prompt = tokens(36, seed=5)
    expected = np.array(cached_logits(tiny_model(), prompt, _SPLITS))
    world_size = 2
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([f"127.0.0.1:{base_port + i}" for i in range(world_size)], f)
        hostfile = f.name
    try:
        ctx = mp.get_context("spawn")
        results: Any = ctx.Queue()
        processes = [
            ctx.Process(
                target=sharded_worker,
                args=(
                    mode,
                    rank,
                    world_size,
                    hostfile,
                    prompt[0].tolist(),
                    _SPLITS,
                    split_layer,
                    results,
                ),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        outcomes: dict[int, np.ndarray] = {}
        for _ in range(world_size):
            rank, ok, value = results.get(timeout=120)  # pyright: ignore[reportAny]
            assert ok, f"rank {rank}: {value}"
            outcomes[rank] = cast(np.ndarray, value)
        for process in processes:
            process.join(timeout=30)
    finally:
        os.unlink(hostfile)
    for rank in range(world_size):
        np.testing.assert_allclose(outcomes[rank], expected, rtol=2e-4, atol=2e-4)
