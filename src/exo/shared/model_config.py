import json

from pathlib import Path
from pydantic import BaseModel, PositiveInt
from exo.shared.architecture import ArchitectureSpec, detect_architecture

class QuantizationConfig(BaseModel, frozen=True, strict=True):
    bits: int
    group_size: int

class ModelConfig(BaseModel, frozen=True, strict=True):
    architecture_spec: ArchitectureSpec
    num_hidden_layers: PositiveInt
    hidden_size: PositiveInt
    intermediate_size: PositiveInt
    num_attention_heads: PositiveInt
    num_key_value_heads: PositiveInt
    vocab_size: PositiveInt
    head_dim: PositiveInt
    rope_theta: float
    rope_scaling: dict | None
    max_position_embeddings: PositiveInt
    rms_norm_eps: float
    tie_word_embeddings: bool
    quantization_config: QuantizationConfig | None

def parse_model_config(config_path: Path) -> ModelConfig:
    with open(config_path) as f:
        raw = json.load(f)

    arch_spec = detect_architecture(raw)
    hidden_size = raw["hidden_size"]
    num_attention_heads = raw["num_attention_heads"]
    num_key_value_heads = raw.get("num_key_value_heads", num_attention_heads)

    intermediate_size = raw.get("intermediate_size", hidden_size * 4)

    head_dim = raw.get("head_dim", hidden_size // num_attention_heads)
    num_hidden_layers = raw.get("num_hidden_layers") or raw.get("num_heads") or 32

    rope_theta = raw.get("rope_theta", arch_spec.rope_theta)

    max_position_embeddings = raw.get("max_position_embeddings", 4096)
    rms_norm_eps = raw.get("rms_norm_eps", 1e-6)
    tie_word_embeddings = raw.get("tie_word_embeddings", False)

    quant_raw = raw.get("quantization_config")
    quantization_config = None

    if quant_raw and "bits" in quant_raw and "group_size" in quant_raw:
        quantization_config = QuantizationConfig(
            bits = quant_raw["bits"],
            group_size = quant_raw["group_size"]
        )

    return ModelConfig(
        architecture_spec = arch_spec,
        num_attention_heads = num_attention_heads,
        num_hidden_layers = num_hidden_layers,
        num_key_value_heads = num_key_value_heads,
        hidden_size = hidden_size,
        intermediate_size = intermediate_size,
        vocab_size = raw["vocab_size"],
        head_dim = head_dim,
        rope_theta = rope_theta,
        rope_scaling = raw.get("rope_scaling"),
        max_position_embeddings = max_position_embeddings,
        rms_norm_eps = rms_norm_eps,
        tie_word_embeddings = tie_word_embeddings,
        quantization_config = quantization_config,
    )
