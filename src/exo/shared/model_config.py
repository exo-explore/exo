import json
from pathlib import Path
from typing import Any

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
    rope_scaling: dict[str, Any] | None
    max_position_embeddings: PositiveInt
    rms_norm_eps: float
    tie_word_embeddings: bool
    quantization_config: QuantizationConfig | None

def parse_model_config(config_path: Path) -> ModelConfig:  # noqa: C901
    with open(config_path) as f:
        raw: dict[str, Any] = json.load(f)  # pyright: ignore[reportAny]

    arch_spec = detect_architecture(raw)
    hidden_size: int = raw["hidden_size"]  # pyright: ignore[reportAny]
    num_attention_heads: int = raw["num_attention_heads"]  # pyright: ignore[reportAny]
    num_key_value_heads: int = raw.get("num_key_value_heads", num_attention_heads)  # pyright: ignore[reportAny]

    intermediate_size: int = raw.get("intermediate_size", hidden_size * 4)  # pyright: ignore[reportAny]

    head_dim: int = raw.get("head_dim", hidden_size // num_attention_heads)  # pyright: ignore[reportAny]
    num_hidden_layers: int = raw.get("num_hidden_layers") or raw.get("num_heads") or 32

    rope_theta: float = raw.get("rope_theta", arch_spec.rope_theta)  # pyright: ignore[reportAny]

    max_position_embeddings: int = raw.get("max_position_embeddings", 4096)  # pyright: ignore[reportAny]
    rms_norm_eps: float = raw.get("rms_norm_eps", 1e-6)  # pyright: ignore[reportAny]
    tie_word_embeddings: bool = raw.get("tie_word_embeddings", False)  # pyright: ignore[reportAny]

    quant_raw: dict[str, Any] | None = raw.get("quantization_config")
    quantization_config: QuantizationConfig | None = None

    if quant_raw and "bits" in quant_raw and "group_size" in quant_raw:
        quantization_config = QuantizationConfig(
            bits = int(quant_raw["bits"]),  # pyright: ignore[reportAny]
            group_size = int(quant_raw["group_size"]),  # pyright: ignore[reportAny]
        )

    vocab_size: int = raw["vocab_size"]  # pyright: ignore[reportAny]
    rope_scaling: dict[str, Any] | None = raw.get("rope_scaling")

    return ModelConfig(
        architecture_spec = arch_spec,
        num_attention_heads = num_attention_heads,
        num_hidden_layers = num_hidden_layers,
        num_key_value_heads = num_key_value_heads,
        hidden_size = hidden_size,
        intermediate_size = intermediate_size,
        vocab_size = vocab_size,
        head_dim = head_dim,
        rope_theta = rope_theta,
        rope_scaling = rope_scaling,
        max_position_embeddings = max_position_embeddings,
        rms_norm_eps = rms_norm_eps,
        tie_word_embeddings = tie_word_embeddings,
        quantization_config = quantization_config,
    )
