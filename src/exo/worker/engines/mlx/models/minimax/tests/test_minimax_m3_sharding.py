"""Pipeline and tensor sharding of a tiny MiniMax M3 against a single device.

Two ranks over MLX's ring backend on localhost. The index blocks are 4 keys and each query
keeps 2 of them, so the 13-token prompt and the decode steps after it run the sparse
attention's block selection (and its index-key cache). Layer 0 is a dense, full-attention
layer and the others are MoE layers with sparse attention, as in MiniMax M3.

The quantized variant stores the routed experts in nvfp4 with a tensor scale per expert,
as MiniMax M3.1's checkpoint does, and everything else in mxfp8.
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
from mlx_lm.models.minimax_m3_vl import Model, ModelArgs
from mlx_lm.models.switch_layers import SwitchLinear

from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.memory import Memory
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.engines.mlx.auto_parallel import (
    pipeline_auto_parallel,
    tensor_auto_parallel,
)

_N_LAYERS = 4
_TEXT_CONFIG: dict[str, Any] = {
    "hidden_size": 64,
    "intermediate_size": 32,
    # Each rank's slice of every quantized input dimension is whole quantization groups.
    "dense_intermediate_size": 128,
    "shared_intermediate_size": 64,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "rotary_dim": 8,
    "num_hidden_layers": _N_LAYERS,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
    "vocab_size": 128,
    "moe_layer_freq": [0, 1, 1, 1],
    "sparse_attention_config": {
        "sparse_index_dim": 16,
        "sparse_num_index_heads": 2,
        "sparse_topk_blocks": 2,
        "sparse_block_size": 4,
        "sparse_local_block": 1,
        "sparse_attention_freq": [0, 1, 1, 1],
    },
}
_SPLIT = 2
_PROMPT = [[(7 * i + 3) % 128 for i in range(13)]]
_DECODE_STEPS = 4


def _quantize(model: Model) -> None:
    def predicate(path: str, module: nn.Module) -> bool | dict[str, Any]:
        if isinstance(module, SwitchLinear):
            return {"group_size": 16, "bits": 4, "mode": "nvfp4", "global_scale": True}
        if isinstance(module, nn.Linear) and not path.endswith("block_sparse_moe.gate"):
            return {"group_size": 32, "bits": 8, "mode": "mxfp8"}
        return False

    nn.quantize(model, class_predicate=predicate)


def _model(quantized: bool) -> Model:
    mx.random.seed(0)
    model = Model(ModelArgs(text_config=dict(_TEXT_CONFIG)))
    # Non-zero norms and routing biases (mlx-lm initialises them to zero).
    params = cast(dict[str, Any], model.parameters())
    for layer in params["language_model"]["model"]["layers"]:  # pyright: ignore[reportAny]
        for norm in ("input_layernorm", "post_attention_layernorm"):
            layer[norm]["weight"] = 0.1 * mx.random.normal((64,))
        if "block_sparse_moe" in layer:
            layer["block_sparse_moe"]["e_score_correction_bias"] = 0.1 * (
                mx.random.normal((8,))
            )
    model.update(params)
    if quantized:
        _quantize(model)
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
        model_id=ModelId("test/minimax-m3-tiny"),
        storage_size=Memory.from_mb(1),
        n_layers=_N_LAYERS,
        hidden_size=64,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )


def _run_rank(
    mode: str,
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


def _sharded_logits(mode: str, quantized: bool, base_port: int) -> list[list[float]]:
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
                args=(mode, quantized, rank, hostfile_path, result_queue),
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


def _argmax(rows: list[list[float]]) -> list[int]:
    return [max(range(len(row)), key=row.__getitem__) for row in rows]


def test_the_prompt_runs_the_sparse_block_selection() -> None:
    model = _model(quantized=False)
    attention = model.layers[1].self_attn
    assert attention.is_sparse_attn
    assert not model.layers[0].self_attn.is_sparse_attn
    blocks = -(-len(_PROMPT[0]) // attention.index_block_size)
    assert blocks > attention.index_topk_blocks


@pytest.mark.parametrize("base_port", [29740])
def test_pipeline_split_matches_a_single_device(base_port: int) -> None:
    single = _generate(_model(quantized=False))
    sharded = _sharded_logits("pipeline", False, base_port)
    assert np.array_equal(np.array(sharded), np.array(single))


@pytest.mark.parametrize("base_port", [29750])
def test_tensor_split_matches_a_single_device(base_port: int) -> None:
    single = _generate(_model(quantized=False))
    sharded = _sharded_logits("tensor", False, base_port)
    np.testing.assert_allclose(
        np.array(sharded), np.array(single), rtol=1e-4, atol=1e-4
    )
    assert _argmax(sharded) == _argmax(single)


@pytest.mark.parametrize("base_port", [29760])
def test_tensor_split_of_nvfp4_experts_matches_a_single_device(base_port: int) -> None:
    """Each expert's nvfp4 tensor scale stays with every rank: split along the experts,
    rank 1 would scale its experts with the wrong expert's scales (or run out of them)."""
    single = _generate(_model(quantized=True))
    sharded = _sharded_logits("tensor", True, base_port)
    np.testing.assert_allclose(
        np.array(sharded), np.array(single), rtol=1e-3, atol=1e-3
    )
    assert _argmax(sharded) == _argmax(single)
