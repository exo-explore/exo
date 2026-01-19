import multiprocessing as mp
from typing import Any

import mlx.core as mx
import pytest

from exo.worker.engines.mlx.auto_parallel import (
    CustomMlxLayer,
    PipelineFirstLayer,
    PipelineLastLayer,
)
from exo.worker.tests.unittests.test_mlx.conftest import MockLayer, run_pipeline_device


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
    import json
    import os
    import tempfile

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
            # Both devices see the final result (4.0) after all_gather
            assert (result_array == 4.0).all(), (
                f"Device {rank}: expected 4.0, got {result_array}"
            )
    finally:
        os.unlink(hostfile_path)
