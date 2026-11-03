"""Pipeline and tensor sharding of a tiny Muse Spark against a single device.

Two ranks over MLX's ring backend on localhost. Layers 1 and 5 are full-attention NoPE
layers and the others sliding-window RoPE layers, as in Muse Spark Open. The 13-token
prompt is longer than the window, so the rotating caches wrap during the decode steps,
and it is longer than the 8 tokens up to which the tensor-parallel MoE splits its
projections' rows itself (decode steps take that path). The norms, gates and router are
random, so a rank holding the wrong heads, experts or norm weights would show.

The quantized variant stores the routed experts of every layer but the first and the last
in 4-bit, as mlx-lm's conversions of the real checkpoint do.
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
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.kernels import norm_add_norm, pre_norm
from mlx_lm.models.muse_spark import Model, ModelArgs, TransformerBlock

from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.memory import Memory
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.engines.mlx.auto_parallel import (
    pipeline_auto_parallel,
    tensor_auto_parallel,
)
from exo.worker.engines.mlx.models.muse_spark.sharding import (
    _PreNormedInput,  # pyright: ignore[reportPrivateUsage]
)
from exo.worker.engines.mlx.sharding.blocks import LayerCallable

_LAYER_TYPES = [
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
]
_N_LAYERS = len(_LAYER_TYPES)
_TEXT_CONFIG: dict[str, Any] = {
    "vocab_size": 256,
    "hidden_size": 64,
    "num_hidden_layers": _N_LAYERS,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "rms_norm_eps": 1e-5,
    "post_norm_eps": 1e-8,
    "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
    "layer_rope_theta": [
        0.0 if t == "full_attention" else 500000.0 for t in _LAYER_TYPES
    ],
    "layer_types": _LAYER_TYPES,
    "sliding_window": 8,
    "final_logit_softcapping": 20.0,
    "output_multiplier": 0.25,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
    # Each rank's slice of the experts' quantized input (the down projection's) is
    # whole quantization groups.
    "moe_hidden_size": 64,
    "moe_intermediate_size": 128,
}
_PROMPT = [[(7 * i + 3) % 256 for i in range(13)]]
_DECODE_STEPS = 4
# Keep the router, its bias and the residual-gate coefficients in float32, as the
# checkpoints do.
_FLOAT32_PARAMETERS = ("_alpha", "_beta", "mlp.gate.weight", "e_score_correction_bias")


def _randomize(module: nn.Module, dtype: mx.Dtype) -> None:
    params: list[tuple[str, mx.array]] = []
    for name, value in cast(
        list[tuple[str, mx.array]], tree_flatten(module.parameters())
    ):
        if name.endswith(("norm.weight", "_alpha", "_beta", "e_score_correction_bias")):
            value = value + 0.2 * mx.random.normal(value.shape)
        elif name.endswith("mlp.gate.weight"):
            value = mx.random.normal(value.shape)
        if not name.endswith(_FLOAT32_PARAMETERS):
            value = value.astype(dtype)
        params.append((name, value))
    module.update(cast(dict[str, Any], tree_unflatten(params)))


def _model(dtype: mx.Dtype, quantized: bool) -> Model:
    mx.random.seed(0)
    model = Model(ModelArgs(model_type="muse_spark", text_config=dict(_TEXT_CONFIG)))
    _randomize(model, dtype)
    if quantized:
        predicate = model.quant_predicate
        nn.quantize(
            model,
            group_size=64,
            bits=4,
            class_predicate=lambda path, module: predicate(path, module),
        )
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
        model_id=ModelId("test/muse-spark-tiny"),
        storage_size=Memory.from_mb(1),
        n_layers=_N_LAYERS,
        hidden_size=64,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )


def _run_rank(
    mode: str,
    dtype_name: str,
    quantized: bool,
    split: int,
    rank: int,
    hostfile_path: str,
    result_queue: Any,  # pyright: ignore[reportAny]
) -> None:
    os.environ["MLX_HOSTFILE"] = hostfile_path
    os.environ["MLX_RANK"] = str(rank)
    try:
        group = mx.distributed.init(backend="ring", strict=True)
        model: nn.Module = _model(getattr(mx, dtype_name), quantized)  # pyright: ignore[reportAny]
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
    mode: str, dtype_name: str, quantized: bool, split: int, base_port: int
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
                args=(mode, dtype_name, quantized, split, rank, hostfile_path)
                + (result_queue,),
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


def test_the_prompt_outgrows_the_window_and_the_pooled_split() -> None:
    model = _model(mx.bfloat16, quantized=False)
    window = model.args.text.sliding_window
    assert len(_PROMPT[0]) > window
    assert len(_PROMPT[0]) + _DECODE_STEPS > 2 * window - len(_PROMPT[0])
    assert [layer.use_sliding for layer in model.layers] == [
        t == "sliding_attention" for t in _LAYER_TYPES
    ]


# Split 2: both stages hold full and sliding layers, and the second starts with a sliding
# one. Split 1: the first stage is one sliding layer (no full attention). Split 5: the
# second stage is one full-attention layer (no sliding window).
@pytest.mark.parametrize(
    ("split", "quantized", "base_port"),
    [(2, False, 29840), (1, False, 29845), (5, False, 29850), (2, True, 29855)],
)
def test_pipeline_split_matches_a_single_device_exactly(
    split: int, quantized: bool, base_port: int
) -> None:
    single = _generate(_model(mx.bfloat16, quantized))
    sharded = _sharded_logits("pipeline", "bfloat16", quantized, split, base_port)
    assert np.array_equal(np.array(sharded), np.array(single))


@pytest.mark.parametrize(("quantized", "base_port"), [(False, 29860), (True, 29865)])
def test_tensor_split_matches_a_single_device(quantized: bool, base_port: int) -> None:
    single = _generate(_model(mx.float32, quantized))
    sharded = _sharded_logits("tensor", "float32", quantized, 0, base_port)
    # The model rounds its last hidden state to bfloat16 before the LM head.
    np.testing.assert_allclose(
        np.array(sharded), np.array(single), rtol=2e-2, atol=2e-2
    )
    assert _argmax(sharded) == _argmax(single)


def test_a_later_stage_norms_its_input_as_the_previous_layer_does() -> None:
    """On one device, each layer's attention input comes out of the previous layer's fused
    residual add and norm. A pipeline stage's first layer gets the stream alone and norms
    it with the same sums."""
    mx.random.seed(1)
    args = ModelArgs(model_type="muse_spark", text_config=dict(_TEXT_CONFIG)).text
    previous = TransformerBlock(args, use_rope=True, use_sliding=True)
    block = TransformerBlock(args, use_rope=True, use_sliding=True)
    _randomize(previous, mx.bfloat16)
    _randomize(block, mx.bfloat16)
    stream = 20 * mx.random.normal((1, 32, 64))

    hidden, fused_input = previous(stream, next_norm=block.input_layernorm)
    expected, _ = block(hidden, attn_input=fused_input)
    stage_first = _PreNormedInput(cast(LayerCallable, cast(object, block)))
    received = cast(tuple[mx.array, mx.array | None], cast(object, stage_first(hidden)))
    mx.eval(expected, received[0])
    assert mx.array_equal(received[0], expected).item()


def test_the_layers_own_input_norm_rounds_differently_at_full_width() -> None:
    """Why a later stage doesn't use the layer's own input norm: at Muse Spark's width it
    rounds a few values differently from the fused kernel, while pre_norm doesn't.

    mlx-lm launches its norm kernels with 1024 threads per row at this width. A GPU that
    can't run them that wide can't run Muse Spark at all, so the test skips there: the
    macOS CI runner's virtual GPU allows 704."""
    mx.random.seed(2)
    width = 8192
    norm = nn.RMSNorm(width, eps=1e-5)
    norm.weight = (1 + 0.1 * mx.random.normal((width,))).astype(mx.bfloat16)
    stream = 30 * mx.random.normal((4, 512, width))
    branch = mx.random.normal((4, 512, width)).astype(mx.bfloat16)
    new_stream, fused = norm_add_norm(
        branch,
        stream,
        alpha=mx.random.uniform(shape=(width,)),
        beta=mx.random.uniform(shape=(width,)),
        post_weight=None,
        post_eps=1e-8,
        next_norm=norm,
    )
    assert fused is not None
    stage_input = pre_norm(new_stream, norm, dtype=mx.bfloat16)
    own_norm = norm(new_stream).astype(mx.bfloat16)
    try:
        mx.eval(fused, stage_input, own_norm)
    except ValueError as error:
        if "maximum allowed threads per threadgroup" not in str(error):
            raise
        pytest.skip(f"this GPU can't run mlx-lm's Muse Spark norm kernels: {error}")
    assert mx.array_equal(stage_input, fused).item()
    assert not mx.array_equal(own_norm, fused).item()
