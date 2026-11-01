"""Pipeline and tensor sharding of a tiny Inkling against a single device.

Two ranks over MLX's ring backend on localhost. The prompt is longer than the sliding window
and the bias extents and is followed by decode steps, so the rotating caches wrap. The
pipeline split leaves rank 0 with sliding-window layers only, so each rank must build its
masks from its own layers.
"""

import json
import multiprocessing as mp
import os
import tempfile
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.memory import Memory
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.engines.mlx.auto_parallel import (
    pipeline_auto_parallel,
    tensor_auto_parallel,
)
from exo.worker.engines.mlx.models.inkling.model import Model, ModelArgs
from exo.worker.engines.mlx.models.inkling.tests.test_inkling_model import (
    _TEXT_CONFIG,  # pyright: ignore[reportPrivateUsage]
    ARGS,
    tiny_model,
)

_N_LAYERS = 4
_SPLIT = 2
_PROMPT = [[(7 * i + 3) % 90 for i in range(23)]]
_DECODE_STEPS = 6


def _drain(
    loader: Any,  # pyright: ignore[reportAny]
) -> nn.Module:
    while True:
        try:
            response = cast(ModelLoadingResponse, next(loader))  # pyright: ignore[reportAny]
            assert response.total > 0
        except StopIteration as stop:
            return cast(nn.Module, stop.value)


def _generate(model: nn.Module) -> list[list[float]]:
    """Logits of the last prompt position and of each greedy decode step."""
    cache = cast(Model, model).make_cache()
    logits = model(mx.array(_PROMPT), cache=cache)[:, -1, :]
    steps = [logits]
    for _ in range(_DECODE_STEPS):
        next_id = mx.argmax(logits, axis=-1, keepdims=True)
        logits = model(next_id, cache=cache)[:, -1, :]
        steps.append(logits)
    mx.eval(*steps)
    return [cast(list[float], step[0].tolist()) for step in steps]


def _card() -> ModelCard:
    return ModelCard(
        model_id=ModelId("test/inkling-tiny"),
        storage_size=Memory.from_mb(1),
        n_layers=_N_LAYERS,
        hidden_size=ARGS.text.hidden_size,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )


# Dimensions that split into whole quantization groups (of 32) across two ranks.
_QUANTIZED_ARGS = ModelArgs(
    model_type="inkling_mm_model",
    text_config=dict(
        _TEXT_CONFIG,
        hidden_size=64,
        head_dim=16,
        swa_head_dim=16,
        dense_intermediate_size=128,
        intermediate_size=64,
    ),
)


def _quantized_model() -> Model:
    """The experts, shared experts and dense MLPs at 4 bits, as the MLX conversions have
    them; attention stays in float32."""
    mx.random.seed(5)
    model = Model(_QUANTIZED_ARGS)
    model.set_dtype(mx.float32)  # pyright: ignore[reportUnknownMemberType]
    nn.quantize(
        model,
        group_size=32,
        bits=4,
        class_predicate=lambda path, module: (
            hasattr(module, "to_quantized") and ".mlp." in path
        ),
    )
    mx.eval(model.parameters())
    return model


def _build(quantized: bool) -> nn.Module:
    return _quantized_model() if quantized else tiny_model()


def _run_rank(
    mode: str,
    rank: int,
    hostfile_path: str,
    result_queue: Any,  # pyright: ignore[reportAny]
    quantized: bool = False,
) -> None:
    os.environ["MLX_HOSTFILE"] = hostfile_path
    os.environ["MLX_RANK"] = str(rank)
    try:
        group = mx.distributed.init(backend="ring", strict=True)
        model: nn.Module = _build(quantized)
        if mode == "pipeline":
            shard = PipelineShardMetadata(
                model_card=_card(),
                device_rank=rank,
                world_size=2,
                start_layer=0 if rank == 0 else _SPLIT,
                end_layer=_SPLIT if rank == 0 else _N_LAYERS,
                n_layers=_N_LAYERS,
            )
            model = _drain(pipeline_auto_parallel(model, group, shard))
        else:
            model = _drain(tensor_auto_parallel(model, group))
        result_queue.put((rank, True, _generate(model)))  # pyright: ignore[reportAny]
    except Exception as error:
        result_queue.put((rank, False, repr(error)))  # pyright: ignore[reportAny]


def _sharded_logits(
    mode: str, base_port: int, quantized: bool = False
) -> list[list[float]]:
    context = mp.get_context("spawn")
    hosts = [f"127.0.0.1:{base_port + rank}" for rank in range(2)]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(hosts, f)
        hostfile_path = f.name
    try:
        result_queue: Any = context.Queue()
        processes = [
            context.Process(
                target=_run_rank,
                args=(mode, rank, hostfile_path, result_queue, quantized),
            )
            for rank in range(2)
        ]
        for process in processes:
            process.start()
        results: dict[int, tuple[bool, object]] = {}
        for _ in processes:
            rank, ok, value = cast(
                tuple[int, bool, object],
                result_queue.get(timeout=120),  # pyright: ignore[reportAny]
            )
            results[rank] = (ok, value)
        for process in processes:
            process.join(timeout=30)
        assert all(ok for ok, _ in results.values()), results
        # Every rank ends with the full logits.
        assert results[0][1] == results[1][1]
        return cast(list[list[float]], results[0][1])
    finally:
        os.unlink(hostfile_path)


@pytest.mark.parametrize("base_port", [29710])
def test_pipeline_split_matches_a_single_device(base_port: int) -> None:
    single = _generate(tiny_model())
    sharded = _sharded_logits("pipeline", base_port)
    assert np.array_equal(np.array(sharded), np.array(single))


def _argmax(rows: list[list[float]]) -> list[int]:
    return [max(range(len(row)), key=row.__getitem__) for row in rows]


@pytest.mark.parametrize("base_port", [29720])
def test_tensor_split_matches_a_single_device(base_port: int) -> None:
    single = _generate(tiny_model())
    sharded = _sharded_logits("tensor", base_port)
    np.testing.assert_allclose(
        np.array(sharded), np.array(single), rtol=1e-4, atol=1e-4
    )
    assert _argmax(sharded) == _argmax(single)


@pytest.mark.parametrize(
    ("mode", "base_port"), [("tensor", 29730), ("pipeline", 29740)]
)
def test_quantized_experts_split_like_a_single_device(
    mode: str, base_port: int
) -> None:
    """Splitting 4-bit expert and MLP weights keeps their scales and biases with their
    rows, on group boundaries."""
    single = _generate(_quantized_model())
    sharded = _sharded_logits(mode, base_port, quantized=True)
    np.testing.assert_allclose(
        np.array(sharded), np.array(single), rtol=1e-4, atol=1e-4
    )
    assert _argmax(sharded) == _argmax(single)
