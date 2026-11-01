"""Row slicing of linear layers for Nemotron-H's Mamba-2 tensor parallelism."""

import gc
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten, tree_unflatten

from exo.worker.engines.mlx.models.nemotron_h.sharding import (
    hidden_dimension_bounds,
    shard_hidden_dimension,
    take_output_rows,
)


@pytest.mark.parametrize(
    ("mode", "group_size", "bits"),
    [
        ("affine", 32, 4),
        ("affine", 64, 8),
        ("mxfp4", 32, 4),
        ("nvfp4", 16, 4),
        ("mxfp8", 32, 8),
    ],
)
def test_take_output_rows_keeps_quantized_rows(
    mode: str, group_size: int, bits: int
) -> None:
    mx.random.seed(1)
    linear = nn.QuantizedLinear(
        64, 96, bias=True, group_size=group_size, bits=bits, mode=mode
    )
    linear.update({"bias": mx.random.normal((96,))})
    inputs = mx.random.normal((3, 64))
    rows = mx.concatenate([mx.arange(48, 72), mx.arange(0, 8), mx.arange(90, 96)])
    expected = linear(inputs)[:, rows]

    take_output_rows(linear, rows)

    assert mx.array_equal(linear(inputs), expected)


def test_take_output_rows_keeps_dense_rows() -> None:
    mx.random.seed(2)
    linear = nn.Linear(64, 96, bias=False)
    inputs = mx.random.normal((3, 64))
    rows = mx.concatenate([mx.arange(48, 72), mx.arange(0, 8)])
    expected = linear(inputs)[:, rows]

    take_output_rows(linear, rows)

    assert mx.array_equal(linear(inputs), expected)


def test_hidden_dimension_splits_on_quantization_groups() -> None:
    # Nemotron 3.5 Lightning's experts: 1856 hidden units = 29 groups of 64.
    assert hidden_dimension_bounds(1856, 64, 0, 2) == (0, 960)
    assert hidden_dimension_bounds(1856, 64, 1, 2) == (960, 1856)
    assert hidden_dimension_bounds(2688, 64, 1, 2) == (1344, 2688)
    assert [hidden_dimension_bounds(96, 32, rank, 4) for rank in range(4)] == [
        (0, 32),
        (32, 64),
        (64, 96),
        (96, 96),
    ]


def test_sharded_hidden_dimension_releases_the_other_ranks_slice(
    tmp_path: Path,
) -> None:
    mx.random.seed(3)
    up = nn.QuantizedLinear(1024, 8192, bias=False, group_size=64, bits=4)
    down = nn.QuantizedLinear(8192, 1024, bias=False, group_size=64, bits=4)
    # Load the weights from a file, as checkpoints are: a slice of such an array is a
    # view that keeps the whole file buffer alive unless it is copied.
    for name, layer in (("up", up), ("down", down)):
        mx.save_safetensors(  # pyright: ignore[reportUnknownMemberType]
            str(tmp_path / f"{name}.safetensors"),
            dict(cast(list[tuple[str, mx.array]], tree_flatten(layer.parameters()))),
        )
        loaded = cast(
            dict[str, mx.array], mx.load(str(tmp_path / f"{name}.safetensors"))
        )
        layer.update(cast(dict[str, Any], tree_unflatten(list(loaded.items()))))
        del loaded
    mx.eval(up.parameters(), down.parameters())
    mx.clear_cache()
    full = mx.get_active_memory()

    shard_hidden_dimension(up, down, rank=0, world_size=2)
    mx.eval(up.parameters(), down.parameters())
    gc.collect()
    mx.clear_cache()

    assert mx.get_active_memory() < 0.7 * full
