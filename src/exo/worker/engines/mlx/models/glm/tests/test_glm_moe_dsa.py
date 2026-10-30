"""The GLM-5.x model classes exo loads: mlx-lm's, with the reference indexer epsilon."""

from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn

from exo.worker.engines.mlx.models import registry
from exo.worker.engines.mlx.models.glm.glm_moe_dsa import (
    INDEXER_KEY_NORM_EPS,
    Float32RouterGate,
    Model,
    ModelArgs,
)

# A tiny GLM-5.3-style config: full/shared indexer pattern (as in GLM-5.2/5.3).
CONFIG: dict[str, Any] = {
    "model_type": "glm_moe_dsa",
    "vocab_size": 64,
    "hidden_size": 32,
    "index_head_dim": 16,
    "index_n_heads": 2,
    "index_topk": 8,
    "intermediate_size": 32,
    "moe_intermediate_size": 16,
    "num_hidden_layers": 5,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "n_shared_experts": 1,
    "n_routed_experts": 4,
    "routed_scaling_factor": 2.5,
    "kv_lora_rank": 16,
    "q_lora_rank": 16,
    "qk_rope_head_dim": 8,
    "v_head_dim": 16,
    "qk_nope_head_dim": 8,
    "topk_method": "noaux_tc",
    "scoring_func": "sigmoid",
    "norm_topk_prob": True,
    "n_group": 1,
    "topk_group": 1,
    "num_experts_per_tok": 2,
    "moe_layer_freq": 1,
    "first_k_dense_replace": 3,
    "max_position_embeddings": 1048576,
    "rms_norm_eps": 1e-5,
    "rope_parameters": {"rope_theta": 8000000, "rope_type": "default"},
    "attention_bias": False,
    "indexer_types": ["full", "full", "full", "shared", "full"],
}


def test_glm_moe_dsa_loads_with_the_family_model_classes() -> None:
    assert registry.get_model_classes(CONFIG) == (Model, ModelArgs)


def test_indexers_use_the_reference_key_norm_epsilon() -> None:
    model = Model(cast(ModelArgs, ModelArgs.from_dict(CONFIG)))  # pyright: ignore[reportUnknownMemberType]
    layers = cast(list[nn.Module], model.model.layers)  # pyright: ignore[reportUnknownMemberType]
    attentions = [cast(nn.Module, layer.self_attn) for layer in layers]
    indexers = [
        cast(nn.Module | None, getattr(attention, "indexer", None))
        for attention in attentions
    ]
    # The "shared" layer reuses the previous layer's top-k and has no indexer.
    assert [indexer is None for indexer in indexers] == [
        False,
        False,
        False,
        True,
        False,
    ]
    for indexer in indexers:
        if indexer is not None:
            key_norm = cast(nn.LayerNorm, indexer.k_norm)
            assert cast(float, key_norm.eps) == INDEXER_KEY_NORM_EPS


def test_moe_layers_route_in_float32() -> None:
    model = Model(cast(ModelArgs, ModelArgs.from_dict(CONFIG)))  # pyright: ignore[reportUnknownMemberType]
    layers = cast(list[nn.Module], model.model.layers)  # pyright: ignore[reportUnknownMemberType]
    gates = [getattr(cast(nn.Module, layer.mlp), "gate", None) for layer in layers]
    # first_k_dense_replace = 3: three dense MLPs, then MoE layers.
    assert [isinstance(gate, Float32RouterGate) for gate in gates] == [
        False,
        False,
        False,
        True,
        True,
    ]
    gate = cast(Float32RouterGate, gates[3])
    x = mx.random.normal((1, 3, CONFIG["hidden_size"])).astype(mx.bfloat16)
    indices, scores = gate(x)
    expected_logits = x.astype(mx.float32) @ gate.weight.T.astype(mx.float32)
    assert indices.shape == (1, 3, CONFIG["num_experts_per_tok"])
    assert scores.dtype == mx.float32
    assert expected_logits.dtype == mx.float32
