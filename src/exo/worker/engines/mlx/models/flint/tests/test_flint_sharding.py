"""Pipeline and tensor sharding of a tiny Flint against a single device.

Two ranks over MLX's ring backend on localhost. The prompt is longer than the sliding
window and is followed by decode steps, so the rotating caches wrap and both attention
masks are exercised. The learned sinks are random per head, so a rank holding another
rank's sinks would show.
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
from exo.worker.engines.mlx.models.flint.model import Model, ModelArgs

# Layer kinds chosen so that rank 1 of a split at layer 2 starts on a sliding layer:
# the whole model's first full layer is 0 and first sliding layer is 1, neither of which
# is right for rank 1's two layers.
_N_LAYERS = 4
_N_HEADS = 8
_ARGS = ModelArgs(
    model_type="flint",
    hidden_size=64,
    num_hidden_layers=_N_LAYERS,
    intermediate_size=96,
    num_attention_heads=_N_HEADS,
    num_key_value_heads=2,
    head_dim=16,
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    mlp_only_layers=[0],
    layer_types=[
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ],
    sliding_window=5,
    rms_norm_eps=1e-6,
    vocab_size=128,
)
_SPLIT = 2
_PROMPT = [[(7 * i + 3) % 128 for i in range(13)]]
_DECODE_STEPS = 4


def _model() -> Model:
    mx.random.seed(0)
    model = Model(_ARGS)
    # Random, non-zero sinks (mlx-lm initialises them to zero).
    for layer in model.layers:
        layer.self_attn.softmax_offset = mx.random.normal((_N_HEADS,))
    mx.eval(model.parameters())
    return model


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
        model_id=ModelId("test/flint-tiny"),
        storage_size=Memory.from_mb(1),
        n_layers=_N_LAYERS,
        hidden_size=_ARGS.hidden_size,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )


def _run_rank(
    mode: str,
    rank: int,
    hostfile_path: str,
    result_queue: Any,  # pyright: ignore[reportAny]
) -> None:
    os.environ["MLX_HOSTFILE"] = hostfile_path
    os.environ["MLX_RANK"] = str(rank)
    try:
        group = mx.distributed.init(backend="ring", strict=True)
        model: nn.Module = _model()
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


def _sharded_logits(mode: str, base_port: int) -> list[list[float]]:
    context = mp.get_context("spawn")
    hosts = [f"127.0.0.1:{base_port + rank}" for rank in range(2)]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(hosts, f)
        hostfile_path = f.name
    try:
        result_queue: Any = context.Queue()
        processes = [
            context.Process(
                target=_run_rank, args=(mode, rank, hostfile_path, result_queue)
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


@pytest.mark.parametrize("base_port", [29610])
def test_pipeline_split_matches_a_single_device(base_port: int) -> None:
    single = _generate(_model())
    sharded = _sharded_logits("pipeline", base_port)
    assert np.array_equal(np.array(sharded), np.array(single))


def _argmax(rows: list[list[float]]) -> list[int]:
    return [max(range(len(row)), key=row.__getitem__) for row in rows]


@pytest.mark.parametrize("base_port", [29620])
def test_tensor_split_matches_a_single_device(base_port: int) -> None:
    single = _generate(_model())
    sharded = _sharded_logits("tensor", base_port)
    np.testing.assert_allclose(
        np.array(sharded), np.array(single), rtol=1e-4, atol=1e-4
    )
    assert _argmax(sharded) == _argmax(single)
