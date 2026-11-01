"""Batched decoding of a tiny MiniMax M3 doesn't build up a graph that grows at every step.

MiniMax M3's sparse-attention layers cache their index keys in a second cache beside the
attention cache, and position them with the attention cache's offsets. In a batch, a cache's
offsets are an array that moves on at every step, so unless something evaluates the index
cache's offsets, each step chains a new array onto the last and keeps every earlier one alive,
each holding a small Metal buffer: 60 more per step for MiniMax M3.1, which reached Metal's
limit on live buffers (499,000) after about 15 minutes of 8 concurrent requests.
"""

import io
from typing import Any, cast

import mlx.core as mx
from mlx_lm.generate import BatchGenerator
from mlx_lm.models.cache import BatchKVCache, CacheList
from mlx_lm.models.minimax_m3_vl import Model, ModelArgs

from exo.worker.engines.mlx.patches.opt_batch_gen import apply_batch_gen_patch

_N_LAYERS = 4
_TEXT_CONFIG: dict[str, Any] = {
    "hidden_size": 64,
    "intermediate_size": 32,
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
    # Blocks of 4 keys, 2 kept: the 3-token prompt starts below the budget (the index keys
    # are cached but not used yet), the 13-token one above it.
    "sparse_attention_config": {
        "sparse_index_dim": 16,
        "sparse_num_index_heads": 2,
        "sparse_topk_blocks": 2,
        "sparse_block_size": 4,
        "sparse_local_block": 1,
        "sparse_attention_freq": [0, 1, 1, 1],
    },
}


def _pending_edges(array: mx.array) -> int:
    dot = io.StringIO()
    mx.export_to_dot(dot, array)  # pyright: ignore[reportUnknownMemberType]
    return dot.getvalue().count("->")


def test_batched_decoding_evaluates_every_cache_offset() -> None:
    apply_batch_gen_patch()
    mx.random.seed(0)
    model = Model(ModelArgs(text_config=dict(_TEXT_CONFIG)))
    mx.eval(model.parameters())
    generator = BatchGenerator(model, prefill_step_size=64)
    try:
        # One sequence finishes early and another joins later, so the batch's caches are
        # also filtered and extended along the way.
        generator.insert(
            prompts=[[(7 * i + 3) % 128 for i in range(13)], [5, 9, 11]],
            max_tokens=[40, 200],
        )
        for step in range(96):
            if step == 20:
                generator.insert(prompts=[[17, 2, 30, 4, 8]], max_tokens=[200])
            generator.next()

        layer_caches = cast(list[CacheList], generator._generation_batch.prompt_cache)
        assert len(layer_caches) == _N_LAYERS
        offsets = [
            sub_cache.offset
            for layer_cache in layer_caches
            for sub_cache in cast(
                tuple[BatchKVCache, ...], getattr(layer_cache, "caches", ())
            )
        ]
        # One attention cache per layer and an index cache per sparse-attention layer
        assert len(offsets) == 2 * _N_LAYERS - 1
        # Each was evaluated at its step, so none chains back to earlier steps
        assert [_pending_edges(offset) for offset in offsets] == [0] * len(offsets)
    finally:
        generator.close()
