"""On MLX's CPU backend a half-precision quantized matmul is many times slower than a float32
one, so a rank there computes in float32. Its pipeline neighbours may still compute in the
checkpoint's half dtype (on a GPU), and recv_like sizes a receive from its template: so every
rank sends and receives hidden states in the checkpoint's dtype.
"""

import multiprocessing as mp
import os
from collections.abc import Iterator
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import pytest

from exo.worker.engines.mlx.sharding.pipeline import (
    PipelineFirstLayer,
    PipelineLastLayer,
)
from exo.worker.engines.mlx.utils_mlx import (
    mx_barrier,
    upcast_for_cpu,
    use_generation_stream_by_default,
)
from exo.worker.tests.unittests.test_mlx.conftest import MockLayer, create_hostfile


class _TinyModel(nn.Module):
    def __init__(self, dtype: mx.Dtype) -> None:
        super().__init__()
        self.projection = nn.QuantizedLinear(64, 64, group_size=64, bits=4)
        self.norm = nn.RMSNorm(64)
        self.set_dtype(dtype)  # pyright: ignore[reportUnknownMemberType]


@pytest.fixture
def on_the_cpu() -> Iterator[None]:
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(device)


@pytest.mark.parametrize("checkpoint_dtype", [mx.float16, mx.bfloat16])
def test_a_cpu_rank_computes_in_float32(
    on_the_cpu: None, checkpoint_dtype: mx.Dtype
) -> None:
    model = _TinyModel(checkpoint_dtype)

    assert upcast_for_cpu(model) == checkpoint_dtype
    assert cast(mx.array, model.projection.weight).dtype == mx.uint32  # still packed
    assert cast(mx.array, model.projection.scales).dtype == mx.float32
    assert model.norm.weight.dtype == mx.float32


def test_a_float32_checkpoint_is_left_alone(on_the_cpu: None) -> None:
    assert upcast_for_cpu(_TinyModel(mx.float32)) is None


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs a GPU to compare with")
def test_a_gpu_rank_is_left_alone() -> None:
    model = _TinyModel(mx.float16)

    assert upcast_for_cpu(model) is None
    assert model.norm.weight.dtype == mx.float16


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
        layer = PipelineLastLayer(
            PipelineFirstLayer(MockLayer(), r=rank, group=group),
            r=rank,
            s=group.size(),
            group=group,
        )
        # Rank 0 computes in float32, as on the CPU; rank 1 in the checkpoint's float16
        compute_dtype = mx.float32 if rank == 0 else mx.float16
        if rank == 0:
            layer.wire_dtype = mx.float16
            cast(PipelineFirstLayer, layer.original_layer).wire_dtype = mx.float16
        mx_barrier(group)
        output = layer(mx.full((1, 1, 4), 1.0, dtype=compute_dtype))
        mx.eval(output)
        results.put((rank, True, (str(output.dtype), output.tolist())))  # pyright: ignore[reportAny]
    except Exception as e:
        results.put((rank, False, repr(e)))  # pyright: ignore[reportAny]


def test_ranks_in_different_dtypes_exchange_hidden_states() -> None:
    world_size = 2
    hostfile_path, _ = create_hostfile(world_size, 29580)
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
    # Each rank doubles its input; every rank ends with the last rank's output, in the
    # dtype it computes in
    assert outcomes == {
        0: (True, ("mlx.core.float32", [[[4.0] * 4]])),
        1: (True, ("mlx.core.float16", [[[4.0] * 4]])),
    }
