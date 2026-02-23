from typing import Any, Literal

from pydantic import BaseModel

AttentionType = Literal["multi_head", "grouped_query", "multi_latent"]
MLPType = Literal["swiglu", "moe_top_k"]
NormType = Literal["rms_norm", "layer_norm"]
RoPEType = Literal["standard", "ntk_aware", "yarn"]

class ArchitectureSpec(BaseModel, frozen=True, strict=True):
    name: str
    attention_type: AttentionType
    mlp_type: MLPType
    norm_type: NormType
    rope_type: RoPEType

    # Weight key templates ({layer_idx} per layer)
    layer_prefix: str
    q_proj_key: str
    k_proj_key: str
    v_proj_key: str
    o_proj_key: str
    gate_proj_key: str
    up_proj_key: str
    down_proj_key: str
    input_norm_key: str
    post_attn_norm_key: str
    final_norm_key: str
    lm_head_key: str

    # Optional normalised keys

    q_norm_key: str | None = None
    k_norm_key: str | None = None

    # MoE (optional, follow-up)
    router_key: str | None = None
    expert_prefix: str | None = None

    # MLA (optional, follow-up)
    kv_lora_rank: int | None = None
    q_lora_rank: int | None = None

    # Embedding key
    embed_key: str | None = None

    # RoPE default
    rope_theta: float = 10000.0

    # Tokenizer
    tokenizer_type: Literal["huggingface", "tiktoken"] = "huggingface"

ARCHITECTURE_REGISTRY: dict[str, ArchitectureSpec] = {}

def register(huggingface_name: str, spec: ArchitectureSpec) -> None:
    ARCHITECTURE_REGISTRY[huggingface_name] = spec

def detect_architecture(raw_config: dict[str, Any]) -> ArchitectureSpec:
    architectures: list[str] = raw_config.get("architectures", [])  # pyright: ignore[reportAny]
    for arch_name in architectures:
        if arch_name in ARCHITECTURE_REGISTRY:
            return ARCHITECTURE_REGISTRY[arch_name]

    model_type: str = raw_config.get("model_type", "")  # pyright: ignore[reportAny]
    if model_type in ARCHITECTURE_REGISTRY:
        return ARCHITECTURE_REGISTRY[model_type]
    raise ValueError(
        f"Unsupported Architecture: {raw_config.get('architectures')}"
    )

from . import llama as _llama  # noqa: F401, E402  # pyright: ignore[reportUnusedImport]
from . import qwen as _qwen  # noqa: F401, E402  # pyright: ignore[reportUnusedImport]
