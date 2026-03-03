from pathlib import Path
from typing import Literal, NamedTuple, overload

from tinygrad.nn.state import safe_load
from tinygrad.tensor import Tensor
from tinygrad.device import Device

from exo.shared.architecture import ArchitectureSpec
from exo.shared.model_config import ModelConfig
from exo.worker.engines.tinygrad.quantization.layers import (
    QuantizedEmbedding,
    QuantizedLinear,
)
from exo.worker.engines.tinygrad.quantization.packing import PackedTensor
from exo.worker.engines.tinygrad.quantization.shapes import infer_weight_shape
from exo.worker.engines.tinygrad.layers.rotary import compute_rope_frequencies


LinearWeight = Tensor | QuantizedLinear
EmbedWeight = Tensor | QuantizedEmbedding

class LayerWeights(NamedTuple):
    qkv_proj: LinearWeight       # Merged Q+K+V
    o_proj: LinearWeight

    gate_up_proj: LinearWeight   # Merged gate+up
    down_proj: LinearWeight

    input_norm: Tensor
    post_attn_norm: Tensor

    # Optional layers
    q_norm: Tensor | None = None
    k_norm: Tensor | None = None

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
    rope_sin: Tensor
    rope_cos: Tensor

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

    if config.tie_word_embeddings:
        if isinstance(embed_tokens, QuantizedEmbedding):
            lm_head: LinearWeight = QuantizedLinear(
                weight_q = embed_tokens.weight_q,
                scales = embed_tokens.scales,
                biases = embed_tokens.biases,
                group_size = embed_tokens.group_size,
            )
        else:
            lm_head: LinearWeight = embed_tokens
    else:
        lm_head: LinearWeight = _build_weight(
            raw_weights, f"{spec.lm_head_key}.weight", config,
        )

    final_norm = raw_weights[f"{spec.final_norm_key}.weight"]

    layers: list[LayerWeights] = []

    for layer_idx in range(start_layer, end_layer):
        prefix = spec.layer_prefix.format(layer_idx=layer_idx)
        layers.append(_build_layer_weights(raw_weights, prefix, spec, config))

    rope_cos, rope_sin = compute_rope_frequencies(
        head_dim = config.head_dim,
        max_seq_len = config.max_position_embeddings,
        rope_theta = config.rope_theta,
    )

    return TransformerWeights(
        embed_tokens=embed_tokens, lm_head=lm_head,
        final_norm=final_norm, layers=layers, config=config,
        rope_cos = rope_cos.realize(), 
        rope_sin = rope_sin.realize(),
    )

def _merge_linear_weights(*weights: LinearWeight) -> LinearWeight:
    """Merge multiple LinearWeight objects by concatenating along the output dimension (dim 0).

    For quantized weights, concatenates packed uint32, scales, and biases directly —
    no extra memory from dequantization. For non-quantized or mixed weights, dequantizes
    first then concatenates plain Tensors.
    """
    if all(isinstance(w, QuantizedLinear) for w in weights):
        qls = [w for w in weights if isinstance(w, QuantizedLinear)]
        merged_tensor = qls[0].weight_q.tensor.cat(
            *[w.weight_q.tensor for w in qls[1:]], dim=0
        ).contiguous().realize()
        merged_scales = qls[0].scales.cat(
            *[w.scales for w in qls[1:]], dim=0
        ).contiguous().realize()
        merged_biases = qls[0].biases.cat(
            *[w.biases for w in qls[1:]], dim=0
        ).contiguous().realize()
        return QuantizedLinear(
            weight_q=PackedTensor(
                tensor=merged_tensor,
                original_shape=(
                    sum(w.weight_q.original_shape[0] for w in qls),
                    qls[0].weight_q.original_shape[1],
                ),
                pack_factor=qls[0].weight_q.pack_factor,
                bits=qls[0].weight_q.bits,
            ),
            scales=merged_scales,
            biases=merged_biases,
            group_size=qls[0].group_size,
        )
    # Non-quantized or mixed: dequantize if needed, then cat
    tensors: list[Tensor] = []
    for w in weights:
        if isinstance(w, QuantizedLinear):
            tensors.append(w.dequantize())
        else:
            tensors.append(w)
    return tensors[0].cat(*tensors[1:], dim=0).contiguous().realize()

def _build_layer_weights(
    raw: dict[str, Tensor],
    prefix: str,
    spec: ArchitectureSpec,
    config: ModelConfig,
) -> LayerWeights:
    def key(suffix: str) -> str:
        return f"{prefix}.{suffix}.weight"

    q_norm = raw.get(f"{prefix}.{spec.q_norm_key}.weight") if spec.q_norm_key else None
    k_norm = raw.get(f"{prefix}.{spec.k_norm_key}.weight") if spec.k_norm_key else None

    q_proj = _build_weight(raw, key(spec.q_proj_key), config)
    k_proj = _build_weight(raw, key(spec.k_proj_key), config)
    v_proj = _build_weight(raw, key(spec.v_proj_key), config)
    qkv_proj = _merge_linear_weights(q_proj, k_proj, v_proj)

    gate_proj = _build_weight(raw, key(spec.gate_proj_key), config)
    up_proj = _build_weight(raw, key(spec.up_proj_key), config)
    gate_up_proj = _merge_linear_weights(gate_proj, up_proj)

    return LayerWeights(
        qkv_proj=qkv_proj,
        o_proj=_build_weight(raw, key(spec.o_proj_key), config),
        gate_up_proj=gate_up_proj,
        down_proj=_build_weight(raw, key(spec.down_proj_key), config),
        input_norm=raw[f"{prefix}.{spec.input_norm_key}.weight"],
        post_attn_norm=raw[f"{prefix}.{spec.post_attn_norm_key}.weight"],
        q_norm=q_norm,
        k_norm=k_norm,
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
    scales_key = key.replace(".weight", ".scales")
    biases_key = key.replace(".weight", ".biases")

    # MLX quantized: .weight (packed uint32) + .scales + .biases + quantization_config
    if key in raw and config.quantization_config is not None and scales_key in raw and biases_key in raw:
        qcfg = config.quantization_config
        packed = PackedTensor(
            tensor = raw[key],
            original_shape = infer_weight_shape(key, config),
            pack_factor = 32 // qcfg.bits,
            bits = qcfg.bits,
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

    # Plain unquantized: .weight only
    if key in raw:
        return raw[key]

    # Legacy .qweight format
    qweight_key = key.replace(".weight", ".qweight")

    if qweight_key in raw and config.quantization_config is not None:
        qcfg = config.quantization_config
        packed = PackedTensor(
            tensor = raw[qweight_key],
            original_shape = infer_weight_shape(key, config),
            pack_factor = 32 // qcfg.bits,
            bits = qcfg.bits,
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
    from tinygrad.helpers import Context

    merged: dict[str, Tensor] = {}

    # Disable BEAM during weight loading. Copy kernels (DISK -> GPU) have unique
    # shapes per tensor and don't benefit from beam search optimisation.
    with Context(BEAM=0):
        for safetensor_file in sorted(path.glob("*.safetensors")):
            shard = safe_load(str(safetensor_file))
            for key, tensor in shard.items():
                merged[key] = tensor.to(Device.DEFAULT).contiguous().realize()

    if not merged:
        raise FileNotFoundError(f"No .safetensors file found in {path}")

    return merged
