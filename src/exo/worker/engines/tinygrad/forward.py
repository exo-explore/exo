from typing import NamedTuple

from tinygrad.tensor import Tensor

from exo.shared.model_config import ModelConfig
from exo.worker.engines.tinygrad.cache import KVCache
from exo.worker.engines.tinygrad.layers.attention import grouped_query_attention
from exo.worker.engines.tinygrad.layers.embedding import apply_embedding, apply_lm_head
from exo.worker.engines.tinygrad.layers.mlp import swiglu_mlp
from exo.worker.engines.tinygrad.layers.normalization import rms_norm
from exo.worker.engines.tinygrad.layers.rotary import compute_rope_frequencies
from exo.worker.engines.tinygrad.weight_loader import LayerWeights, TransformerWeights

class TransfomerBlockBuilder(NamedTuple):
    cos_freqs: Tensor
    sin_freqs: Tensor
    layer: LayerWeights
    idx: int
    offset: int

def forward_pass(
    weights: TransformerWeights,
    input_ids: Tensor,
    cache: KVCache | None,
    position_offset: int = 0,
) -> tuple[Tensor, KVCache]:
    config = weights.config
    _batch, seq_len = input_ids.shape

    x = apply_embedding(weights.embed_tokens, input_ids)

    if cache is None:
       cache = KVCache(num_layers = len(weights.layers))

    for layer_idx, layer in enumerate(weights.layers):
        builder = TransfomerBlockBuilder(
           weights.rope_cos, weights.rope_sin, 
            layer, layer_idx, position_offset,
        )
        x = _transformer_block(x, config, cache, builder)

    x = rms_norm(x, weights.final_norm, config.rms_norm_eps)
    logits = apply_lm_head(x, weights.lm_head)

    return logits, cache

def _transformer_block(
    x: Tensor,
    config: ModelConfig,
    cache: KVCache,
    builder: TransfomerBlockBuilder,
) -> Tensor:
    residual = x
    layer = builder.layer
    x = rms_norm(x, layer.input_norm, config.rms_norm_eps)

    match config.architecture_spec.attention_type:
        case "grouped_query" | "multi_head":
            x = grouped_query_attention(
                x, q_proj = layer.q_proj, k_proj = layer.k_proj,
                v_proj = layer.v_proj, o_proj = layer.o_proj,
                cos_freqs = builder.cos_freqs,
                sin_freqs = builder.sin_freqs,
                cache = cache, layer_idx = builder.idx,
                position_offset = builder.offset,
                num_heads = config.num_attention_heads,
                num_kv_heads = config.num_key_value_heads,
                head_dim = config.head_dim,
                q_norm = layer.q_norm,
                k_norm = layer.k_norm,
                rms_norm_eps = config.rms_norm_eps,
            )
        case "multi_latent":
            raise NotImplementedError(
                "MLA attention: not yet been implemented"
            )

    x = x + residual
    residual = x

    x = rms_norm(x, layer.post_attn_norm, config.rms_norm_eps)
    match config.architecture_spec.mlp_type:
        case "swiglu":
            x = swiglu_mlp(
                x, layer.gate_proj, layer.up_proj, layer.down_proj
            )
        case "moe_top_k":
            raise NotImplementedError(
                "MoE MLP: not yet implemented"
            )

    x = x + residual
    return x
