import json
import multiprocessing as mp
import os
import tempfile
from typing import Any

import mlx.core as mx
import mlx.nn as mlx_nn
import pytest

from exo.worker.engines.mlx.auto_parallel import (
    CustomMlxLayer,
    PipelineFirstLayer,
    PipelineLastLayer,
    patch_distributed_model,
)
from exo.worker.tests.unittests.test_mlx.conftest import MockLayer


def run_pipeline_device(
    rank: int,
    world_size: int,
    hostfile_path: str,
    result_queue: Any,  # pyright: ignore[reportAny]
) -> None:
    import os

    os.environ["MLX_HOSTFILE"] = hostfile_path
    os.environ["MLX_RANK"] = str(rank)

    class MockLayerInner(mlx_nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.custom_attr = "test_value"

        def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
            return x * 2

    class MockModel(mlx_nn.Module):
        def __init__(self, layers: list[mlx_nn.Module]) -> None:
            super().__init__()
            self.layers = layers

        def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
            for layer in self.layers:
                x = layer(x, *args, **kwargs)  # pyright: ignore[reportUnknownVariableType]
            return x  # pyright: ignore[reportUnknownVariableType]

    try:
        group = mx.distributed.init(backend="ring", strict=True)

        mock = MockLayerInner()
        first = PipelineFirstLayer(mock, r=rank, group=group)
        composed = PipelineLastLayer(first, r=rank, s=world_size, group=group)

        # Wrap in a mock model, then wrap in PipelineParallelModel for all_gather
        inner_model = MockModel([composed])
        model = patch_distributed_model(inner_model)

        x = mx.ones((1, 4))
        result = model(x)
        mx.eval(result)
        success = result.shape == x.shape
        result_queue.put((rank, success, result))  # pyright: ignore[reportAny]
    except Exception as e:
        result_queue.put((rank, False, str(e)))  # pyright: ignore[reportAny]


def test_single_wrapper_delegates_attributes() -> None:
    mock = MockLayer()
    wrapped = CustomMlxLayer(mock)

    assert wrapped.custom_attr == "test_value"  # type: ignore[attr-defined]
    assert wrapped.use_sliding is True  # type: ignore[attr-defined]


def test_composed_wrappers_delegate_attributes() -> None:
    mock = MockLayer()
    group = mx.distributed.init()

    first = PipelineFirstLayer(mock, r=0, group=group)
    composed = PipelineLastLayer(first, r=0, s=1, group=group)

    assert composed.custom_attr == "test_value"  # type: ignore[attr-defined]
    assert composed.use_sliding is True  # type: ignore[attr-defined]


def test_missing_attribute_raises() -> None:
    mock = MockLayer()
    wrapped = CustomMlxLayer(mock)

    with pytest.raises(AttributeError):
        _ = wrapped.nonexistent_attr  # type: ignore[attr-defined]


def test_composed_call_works() -> None:
    ctx = mp.get_context("spawn")

    world_size = 2
    base_port = 29500

    hosts = [f"127.0.0.1:{base_port + i}" for i in range(world_size)]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(hosts, f)
        hostfile_path = f.name

    try:
        result_queue: Any = ctx.Queue()

        processes: list[Any] = []
        for rank in range(world_size):
            p = ctx.Process(
                target=run_pipeline_device,
                args=(rank, world_size, hostfile_path, result_queue),
            )
            p.start()
            processes.append(p)

        for p in processes:  # pyright: ignore[reportAny]
            p.join(timeout=10)  # pyright: ignore[reportAny]

        results: dict[int, Any] = {}
        errors: dict[int, str] = {}
        while not result_queue.empty():  # pyright: ignore[reportAny]
            rank, success, value = result_queue.get()  # pyright: ignore[reportAny]
            if success:
                results[rank] = value
            else:
                errors[rank] = value

        assert len(results) == world_size, (
            f"Expected {world_size} results, got {len(results)}. Errors: {errors}"
        )

        for rank in range(world_size):
            assert rank in results, (
                f"Device {rank} failed: {errors.get(rank, 'unknown')}"
            )
            result_array = results[rank]
            # Each device sees its local result: intermediate ranks return their
            # computed output (before sending), last rank returns the final result.
            # With world_size=2 and each layer doing x*2:
            #   - Rank 0: 1.0 * 2 = 2.0 (sends to rank 1)
            #   - Rank 1: 2.0 * 2 = 4.0 (last rank, final result)
            expected = 2.0 * (2**rank)  # 2.0 for rank 0, 4.0 for rank 1
            assert (result_array == expected).all(), (
                f"Device {rank}: expected {expected}, got {result_array}"
            )
    finally:
        os.unlink(hostfile_path)
