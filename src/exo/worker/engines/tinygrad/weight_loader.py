from typing import NamedTuple, Any
from tinygrad import Tensor
from tinygrad.nn.state import safe_load
from exo.worker.engines.tinygrad.quantization.layers import (
    QuantizedLinear, QuantizedEmbedding
)
from exo.worker.engines.tinygrad.quantization.shapes import infer_weight_shape
from exo.shared.model_config import ModelConfig
from pathlib import Path

LinearWeight = Tensor | QuantizedLinear
EmbedWeight = Tensor | QuantizedEmbedding

class LayerWeights(NamedTuple):
    q_proj: LinearWeight
    k_proj: LinearWeight
    v_proj: LinearWeight
    o_proj: LinearWeight

    gate_proj: LinearWeight
    up_proj: LinearWeight
    down_proj: LinearWeight

    input_norm: Tensor
    post_attn_norm: Tensor

    # MoE (None for dense models)
    router_weight: Tensor | None = None
    expert_gate_projs: list[LinearWeight] | None = None
    expert_up_projs: list[LinearWeight] | None = None
    expert_down_projs: list[LinearWeight] | None = None

class TransformerWeights(NamedTuple):
    embed_tokens: EmbedWeight
    lm_head: LinearWeight
    final_norm: Tensor
    layers: list[LayerWeights]
    config: ModelConfig

def load_transformer_weights(
    model_path: Path,
    config: ModelConfig,
    start_layer: int = 0,
    end_layer: int | None = None,
) -> TransformerWeights:

    if end_layer is None:
        end_layer = config.num_hidden_layers

    spec = config.architecture_spec
    raw_weights = _load_all_safetensors(model_path)

    embed_tokens = _build_weight(raw_weights,
        f"{spec.embed_key}.weight",
        config, is_embedding = True)

    lm_head = embed_tokens if config.tie_word_embeddings else _build_weight(
        raw_weights, f"{spec.lm_head_key}.weight", config,
    )

    final_norm = raw_weights[f"{spec.final_norm_key}.weight"]

    layers = []

    for layer_idx in range(start_layer, end_layer):
        prefix = spec.layer_prefix.format(layer_idx=layer_idx)
        layers.append(_build_layer_weights(raw_weights, prefix, spec, config))

    return TransformerWeights(
        embed_tokens=embed_tokens, lm_head=lm_head,
        final_norm=final_norm, layers=layers, config=config
    )

def _build_weight(
    raw: Any,
    key: str,
    config: ModelConfig,
    is_embedding: bool = False,
) -> LinearWeight | EmbedWeight:
    qweight_key = key.replace(".weight", ".qweight")

    if qweight_key in raw and config.quantization_config is not None:
        qcfg = config.quantization_config
        scales_key = key.replace(".weight", ".scales")
        biases_key = key.replace(".weight", ".biases")

        packed = PackedTensor(
            tensor = raw[qweight_key],
            original_shape = infer_weight_shape(key, config),
            pack_factor = 32 // qcfg.bits,
            bits = qcfg.bits
        )

        if is_embedding:
            return QuantizedEmbedding(
                num_embeddings = config.vocab_size,
                embedding_dim = config.hidden_size,
                weight_q = packed,
                scales = raw[scales_key],
                biases = raw[biases_key],
                group_size = qcfg.group_size,
            )

        return QuantizedEmbedding(
            weight_q = packed,
            scales = raw[scales_key],
            biases = raw[biases_key],
            group_size = qcfg.group_size,
        )

        if key in raw:
            return raw[key]

    raise KeyError(f"Weight key '{key}' not found (also tried {qweight_key})")

def _load_all_safetensors(path: Path) -> dict[str, Tensor]:
    merged: dict[str, Tensor] = {}

    for safetensor_file in sorted(path.glob("*.safetensors")):
        shard = safe_load(str(safetensor_file))
        merged.update(shard)

    if not merged:
        raise FileNotFoundError(f"No .safetensors file found in {path}")

    return merged
