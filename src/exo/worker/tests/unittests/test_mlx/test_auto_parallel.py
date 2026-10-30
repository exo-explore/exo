import json
import multiprocessing as mp
import os
import tempfile
from typing import Any, cast

import mlx.core as mx
import mlx.nn as mlx_nn
import pytest

from exo.worker.engines.mlx.sharding.blocks import CustomMlxLayer
from exo.worker.engines.mlx.sharding.pipeline import (
    PipelineFirstLayer,
    PipelineLastLayer,
    patch_pipeline_model,
)
from exo.worker.tests.unittests.test_mlx.conftest import MockLayer


def run_pipeline_device(
    rank: int,
    world_size: int,
    hostfile_path: str,
    result_queue: Any,  # pyright: ignore[reportAny]
    layers_return_tuples: bool = False,
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

    class MockTupleLayerInner(mlx_nn.Module):
        """Returns (hidden state, extra), like GLM MoE DSA layers (hidden, top-k)."""

        def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
            return cast(mx.array, cast(object, (x * 2, mx.array([rank]))))

    class MockModel(mlx_nn.Module):
        def __init__(self, layers: list[mlx_nn.Module]) -> None:
            super().__init__()
            self.layers = layers

        def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
            for layer in self.layers:
                out = layer(x, *args, **kwargs)
                if layers_return_tuples:
                    x, extra = cast(tuple[mx.array, mx.array], cast(object, out))
                    assert extra.tolist() == [rank]
                else:
                    x = out
            return x

    try:
        group = mx.distributed.init(backend="ring", strict=True)

        mock = MockTupleLayerInner() if layers_return_tuples else MockLayerInner()
        first = PipelineFirstLayer(mock, r=rank, group=group)
        composed = PipelineLastLayer(first, r=rank, s=world_size, group=group)

        # Wrap in a mock model, then wrap in PipelineParallelModel for all_gather
        inner_model = MockModel([composed])
        model = patch_pipeline_model(inner_model, group)

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


@pytest.mark.parametrize(
    ("layers_return_tuples", "base_port"), [(False, 29500), (True, 29510)]
)
def test_composed_call_works(layers_return_tuples: bool, base_port: int) -> None:
    ctx = mp.get_context("spawn")

    world_size = 2

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
                args=(
                    rank,
                    world_size,
                    hostfile_path,
                    result_queue,
                    layers_return_tuples,
                ),
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
            # Both devices see the final result (4.0) after all_gather
            assert (result_array == 4.0).all(), (
                f"Device {rank}: expected 4.0, got {result_array}"
            )
    finally:
        os.unlink(hostfile_path)


def run_shared_topk_device(
    rank: int,
    world_size: int,
    hostfile_path: str,
    result_queue: Any,  # pyright: ignore[reportAny]
    tokens: int,
    topk_size: int,
) -> None:
    """Two ranks of a model whose layers share a DSA top-k selection (GLM-5.2+):
    rank 0's last layer returns a selection, rank 1's first layer must receive it."""
    import os

    os.environ["MLX_HOSTFILE"] = hostfile_path
    os.environ["MLX_RANK"] = str(rank)

    class FakeCache:
        offset = 0

    received: list[object] = []

    class SharedTopkLayer(mlx_nn.Module):
        """Called as layer(x, mask, cache, prev_topk), returns (x, topk)."""

        def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
            prev_topk = cast(mx.array | None, args[2])
            received.append(None if prev_topk is None else prev_topk.tolist())
            topk = (
                mx.full((1, 1, x.shape[1], topk_size), 7, mx.int32)
                if x.shape[1] > topk_size
                else None
            )
            return cast(mx.array, cast(object, (x * 2, topk)))

    try:
        group = mx.distributed.init(backend="ring", strict=True)
        first = PipelineFirstLayer(SharedTopkLayer(), r=rank, group=group)
        composed = PipelineLastLayer(first, r=rank, s=world_size, group=group)
        first.shared_topk_size = topk_size
        composed.shared_topk_size = topk_size
        out = cast(
            tuple[mx.array, object],
            cast(object, composed(mx.ones((1, tokens, 4)), None, FakeCache(), None)),
        )
        mx.eval(out[0])
        result_queue.put((rank, True, received))  # pyright: ignore[reportAny]
    except Exception as e:
        result_queue.put((rank, False, str(e)))  # pyright: ignore[reportAny]


@pytest.mark.parametrize(("tokens", "base_port"), [(3, 29520), (2, 29530)])
def test_shared_topk_is_forwarded_to_the_next_rank(tokens: int, base_port: int) -> None:
    ctx = mp.get_context("spawn")
    world_size, topk_size = 2, 2
    hosts = [f"127.0.0.1:{base_port + i}" for i in range(world_size)]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(hosts, f)
        hostfile_path = f.name
    try:
        result_queue: Any = ctx.Queue()
        processes: list[Any] = []
        for rank in range(world_size):
            p = ctx.Process(
                target=run_shared_topk_device,
                args=(rank, world_size, hostfile_path, result_queue, tokens, topk_size),
            )
            p.start()
            processes.append(p)
        for p in processes:  # pyright: ignore[reportAny]
            p.join(timeout=30)  # pyright: ignore[reportAny]
        results: dict[int, tuple[bool, object]] = {}
        while not result_queue.empty():  # pyright: ignore[reportAny]
            rank, success, value = cast(tuple[int, bool, object], result_queue.get())  # pyright: ignore[reportAny]
            results[rank] = (success, value)
        assert len(results) == world_size, results
        assert all(success for success, _ in results.values()), results
        assert results[0][1] == [None]
        expected = [[[[7] * topk_size] * tokens]] if tokens > topk_size else None
        assert results[1][1] == [expected]
    finally:
        os.unlink(hostfile_path)
