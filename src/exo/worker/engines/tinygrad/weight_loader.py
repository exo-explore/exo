from pathlib import Path
from typing import Literal, NamedTuple, overload

from tinygrad.nn.state import safe_load
from tinygrad.tensor import Tensor

from exo.shared.architecture import ArchitectureSpec
from exo.shared.model_config import ModelConfig
from exo.worker.engines.tinygrad.quantization.layers import (
    QuantizedEmbedding,
    QuantizedLinear,
)
from exo.worker.engines.tinygrad.quantization.packing import PackedTensor
from exo.worker.engines.tinygrad.quantization.shapes import infer_weight_shape

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

    lm_head: LinearWeight = embed_tokens if config.tie_word_embeddings else _build_weight(  # pyright: ignore[reportAssignmentType]
        raw_weights, f"{spec.lm_head_key}.weight", config,
    )

    final_norm = raw_weights[f"{spec.final_norm_key}.weight"]

    layers: list[LayerWeights] = []

    for layer_idx in range(start_layer, end_layer):
        prefix = spec.layer_prefix.format(layer_idx=layer_idx)
        layers.append(_build_layer_weights(raw_weights, prefix, spec, config))

    return TransformerWeights(
        embed_tokens=embed_tokens, lm_head=lm_head,
        final_norm=final_norm, layers=layers, config=config
    )

def _build_layer_weights(
    raw: dict[str, Tensor],
    prefix: str,
    spec: ArchitectureSpec,
    config: ModelConfig,
) -> LayerWeights:
    def key(suffix: str) -> str:
        return f"{prefix}.{suffix}.weight"

    return LayerWeights(
        q_proj=_build_weight(raw, key(spec.q_proj_key), config),
        k_proj=_build_weight(raw, key(spec.k_proj_key), config),
        v_proj=_build_weight(raw, key(spec.v_proj_key), config),
        o_proj=_build_weight(raw, key(spec.o_proj_key), config),
        gate_proj=_build_weight(raw, key(spec.gate_proj_key), config),
        up_proj=_build_weight(raw, key(spec.up_proj_key), config),
        down_proj=_build_weight(raw, key(spec.down_proj_key), config),
        input_norm=raw[f"{prefix}.{spec.input_norm_key}.weight"],
        post_attn_norm=raw[f"{prefix}.{spec.post_attn_norm_key}.weight"],
    )

@overload
def _build_weight(raw: dict[str, Tensor], key: str, config: ModelConfig, is_embedding: Literal[True]) -> EmbedWeight: ...
@overload
def _build_weight(raw: dict[str, Tensor], key: str, config: ModelConfig, is_embedding: Literal[False] = ...) -> LinearWeight: ...

def _build_weight(
    raw: dict[str, Tensor],
    key: str,
    config: ModelConfig,
    is_embedding: bool = False,
) -> LinearWeight | EmbedWeight:
    if key in raw:
        return raw[key]

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

        return QuantizedLinear(
            weight_q = packed,
            scales = raw[scales_key],
            biases = raw[biases_key],
            group_size = qcfg.group_size,
        )

    raise KeyError(f"Weight key '{key}' not found (also tried {qweight_key})")

def _load_all_safetensors(path: Path) -> dict[str, Tensor]:
    merged: dict[str, Tensor] = {}

    for safetensor_file in sorted(path.glob("*.safetensors")):
        shard = safe_load(str(safetensor_file))
        merged.update(shard)

    if not merged:
        raise FileNotFoundError(f"No .safetensors file found in {path}")

    return merged
