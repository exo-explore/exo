"""Pipeline and tensor sharding of a tiny Laguna against a single device.

Two ranks over MLX's ring backend on localhost. The prompt is longer than the sliding
window and is followed by decode steps, so the rotating caches wrap and both attention
masks are exercised. Head counts differ per layer as in the real checkpoints, the
full-attention layers use YaRN on half of each head as Laguna S 2.1 does, and the
router's score-correction bias is random, so a rank holding the wrong heads, gates or
bias would show.
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
from mlx_lm.models.laguna import Model, ModelArgs

from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.memory import Memory
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.engines.mlx.auto_parallel import (
    pipeline_auto_parallel,
    tensor_auto_parallel,
)

_LAYER_TYPES = [
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
]
_N_LAYERS = len(_LAYER_TYPES)
_ARGS = ModelArgs(
    model_type="laguna",
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=_N_LAYERS,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    num_attention_heads_per_layer=[4, 6, 6, 4, 6],
    layer_types=_LAYER_TYPES,
    mlp_layer_types=["dense"] + ["sparse"] * (_N_LAYERS - 1),
    sliding_window=5,
    gating="per-head",
    rope_parameters={
        "full_attention": {
            "rope_type": "yarn",
            "rope_theta": 500000.0,
            "factor": 32.0,
            "original_max_position_embeddings": 8192,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "partial_rotary_factor": 0.5,
        },
        "sliding_attention": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
        },
    },
    num_experts=8,
    num_experts_per_tok=3,
    moe_intermediate_size=64,
    shared_expert_intermediate_size=64,
    moe_routed_scaling_factor=2.5,
    rms_norm_eps=1e-6,
)
_PROMPT = [[(7 * i + 3) % 128 for i in range(13)]]
_DECODE_STEPS = 4


def _model(quantized: bool) -> Model:
    mx.random.seed(0)
    model = Model(_ARGS)
    # Random score-correction biases (mlx-lm initialises them to zero), large enough to
    # change which experts are picked.
    for layer in model.layers[1:]:
        gate = cast(Any, layer.mlp).gate  # pyright: ignore[reportAny]
        gate.e_score_correction_bias = mx.random.uniform(
            low=-0.3, high=0.3, shape=(_ARGS.num_experts,)
        )
    if quantized:
        nn.quantize(model, group_size=32, bits=4)
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
    return [cast(list[float], step[0].astype(mx.float32).tolist()) for step in steps]


def _card() -> ModelCard:
    return ModelCard(
        model_id=ModelId("test/laguna-tiny"),
        storage_size=Memory.from_mb(1),
        n_layers=_N_LAYERS,
        hidden_size=_ARGS.hidden_size,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )


def _run_rank(
    mode: str,
    split: int,
    quantized: bool,
    rank: int,
    hostfile_path: str,
    result_queue: Any,  # pyright: ignore[reportAny]
) -> None:
    os.environ["MLX_HOSTFILE"] = hostfile_path
    os.environ["MLX_RANK"] = str(rank)
    try:
        group = mx.distributed.init(backend="ring", strict=True)
        model: nn.Module = _model(quantized)
        if mode == "pipeline":
            shard = PipelineShardMetadata(
                model_card=_card(),
                device_rank=rank,
                world_size=2,
                start_layer=0 if rank == 0 else split,
                end_layer=split if rank == 0 else _N_LAYERS,
                n_layers=_N_LAYERS,
            )
            model = _drain(pipeline_auto_parallel(model, group, shard))
        else:
            model = _drain(tensor_auto_parallel(model, group))
        result_queue.put((rank, True, _generate(model)))  # pyright: ignore[reportAny]
    except Exception as error:
        result_queue.put((rank, False, repr(error)))  # pyright: ignore[reportAny]


def _sharded_logits(
    mode: str, split: int, quantized: bool, base_port: int
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
                args=(mode, split, quantized, rank, hostfile_path, result_queue),
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


# split 2: rank 1 starts on a sliding layer, so the whole model's first full-attention
#   layer (0) is a sliding layer there.
# split 1: rank 0 holds only a full-attention layer (no sliding cache entry to index).
# split 4: rank 1 holds only a sliding layer (no full-attention layer).
@pytest.mark.parametrize(
    "split,quantized,base_port",
    [
        (2, False, 29710),
        (1, False, 29712),
        (4, False, 29714),
        (2, True, 29716),
    ],
)
def test_pipeline_split_matches_a_single_device(
    split: int, quantized: bool, base_port: int
) -> None:
    single = _generate(_model(quantized))
    sharded = _sharded_logits("pipeline", split, quantized, base_port)
    assert np.array_equal(np.array(sharded), np.array(single))


def _argmax(rows: list[list[float]]) -> list[int]:
    return [max(range(len(row)), key=row.__getitem__) for row in rows]


@pytest.mark.parametrize(
    "quantized,base_port",
    [(False, 29720), (True, 29722)],
)
def test_tensor_split_matches_a_single_device(quantized: bool, base_port: int) -> None:
    single = _generate(_model(quantized))
    sharded = _sharded_logits("tensor", 0, quantized, base_port)
    np.testing.assert_allclose(
        np.array(sharded), np.array(single), rtol=1e-4, atol=1e-4
    )
    assert _argmax(sharded) == _argmax(single)
