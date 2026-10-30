"""A tiny random GLM-5.3-Flash (glm5_next) shared by the model and sharding tests, and the
worker the sharding tests run on each distributed rank."""

import os
from collections.abc import Generator
from typing import Any, NamedTuple, cast

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from exo.worker.engines.mlx.models.glm.glm5_next import Model, ModelArgs

TEXT_CONFIG: dict[str, Any] = {
    "model_type": "glm5_next_text",
    "vocab_size": 97,
    "hidden_size": 32,
    "num_hidden_layers": 5,
    "rms_norm_eps": 1e-5,
    "layer_types": [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "deepseek_sparse_attention",
        "linear_attention",
    ],
    "mlp_layer_types": ["dense", "dense", "sparse", "sparse", "sparse"],
    "indexer_types": ["full"] * 5,
    "linear_attn_config": {
        "num_heads": 2,
        "head_dim": 32,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0,
    },
    "intermediate_size": 48,
    "moe_intermediate_size": 16,
    "n_routed_experts": 4,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "routed_scaling_factor": 2.5,
    "norm_topk_prob": True,
    "n_group": 1,
    "topk_group": 1,
    "topk_method": "noaux_tc",
    "scoring_func": "sigmoid",
    "swiglu_limit": 10.0,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "q_lora_rank": 16,
    "kv_lora_rank": 32,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 0,
    "v_head_dim": 16,
    "mla_use_nope": True,
    "index_n_heads": 2,
    "index_head_dim": 32,
    "index_topk": 8,
    "index_kpool": 4,
    "index_kpool_compress": True,
    "index_kpool_always_select_tail": True,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
}
CONFIG: dict[str, Any] = {
    "model_type": "glm5_next",
    "architectures": ["Glm5NextForConditionalGeneration"],
    "text_config": TEXT_CONFIG,
    "tie_word_embeddings": False,
}


def tiny_model() -> Model:
    """The same random float32 weights on every call (and every rank)."""
    mx.random.seed(0)
    model = Model(cast(ModelArgs, ModelArgs.from_dict(CONFIG)))  # pyright: ignore[reportUnknownMemberType]
    params: list[tuple[str, mx.array]] = []
    for name, value in cast(
        list[tuple[str, mx.array]], tree_flatten(model.parameters())
    ):
        params.append((name, mx.random.normal(value.shape) * 0.2))
    model.load_weights(params)
    mx.eval(model.parameters())
    return model


def tokens(n: int, seed: int) -> mx.array:
    return mx.array(np.random.default_rng(seed).integers(0, 97, size=(1, n)))


def cached_logits(model: Model, prompt: mx.array, splits: tuple[int, ...]) -> mx.array:
    """Prefill in chunks ending at each split, then decode the rest one token at a time."""
    cache = model.make_cache()
    pieces: list[mx.array] = []
    start = 0
    for end in splits:
        pieces.append(model(prompt[:, start:end], cache))
        start = end
    for i in range(start, prompt.shape[1]):
        pieces.append(model(prompt[:, i : i + 1], cache))
    return mx.concatenate(pieces, axis=1)


class _PipelineShard(NamedTuple):
    start_layer: int
    end_layer: int
    device_rank: int
    world_size: int
    n_layers: int


def _drain(generator: Generator[object, None, Model]) -> Model:
    while True:
        try:
            next(generator)
        except StopIteration as stop:
            return cast(Model, stop.value)


def sharded_worker(
    mode: str,
    rank: int,
    world_size: int,
    hostfile: str,
    prompt: list[int],
    splits: tuple[int, ...],
    split_layer: int,
    results: Any,  # pyright: ignore[reportAny]
) -> None:
    """One rank of a 2-rank ring: shard the tiny model with exo's tensor or pipeline
    parallelism and return its cached logits for the prompt."""
    os.environ["MLX_HOSTFILE"] = hostfile
    os.environ["MLX_RANK"] = str(rank)
    try:
        from exo.worker.engines.mlx.auto_parallel import (
            pipeline_auto_parallel,
            tensor_auto_parallel,
        )

        group = mx.distributed.init(backend="ring", strict=True)
        model = tiny_model()
        if mode == "tensor":
            model = _drain(
                cast(Generator[object, None, Model], tensor_auto_parallel(model, group))
            )
        else:
            n_layers = len(model.layers)
            bounds = [(0, split_layer), (split_layer, n_layers)][rank]
            shard = _PipelineShard(bounds[0], bounds[1], rank, world_size, n_layers)
            model = _drain(
                cast(
                    Generator[object, None, Model],
                    pipeline_auto_parallel(model, group, cast(Any, shard)),  # pyright: ignore[reportAny]
                )
            )
        logits = cached_logits(model, mx.array([prompt]), splits)
        mx.eval(logits)
        results.put((rank, True, np.array(logits)))  # pyright: ignore[reportAny]
    except Exception as e:  # noqa: BLE001 - reported to the parent process
        results.put((rank, False, repr(e)))  # pyright: ignore[reportAny]
