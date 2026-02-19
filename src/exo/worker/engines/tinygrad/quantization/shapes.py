from typing import Literal

from exo.shared.model_config import ModelConfig

LayerType = Literal[
    "q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj",
    "gate_proj", "up_proj", "down_proj",
    "embed_tokens", "lm_head",
    "unknown",
]

_LAYER_PATTERNS: dict[LayerType, list[str]] = {
    "q_proj": [".q_proj.", ".q.", ".attn.q"],
    "k_proj": [".k_proj.", ".k.", ".attn.k"],
    "v_proj": [".v_proj.", ".v.", ".attn.v"],
    "o_proj": [".o_proj.", ".o.", ".attn.o"],
    "qkv_proj": [".c_attn.", ".qkv.", ".attn.c_attn"],
    "gate_proj": [".gate_proj.", ".gate.", ".mlp.gate."],
    "up_proj": [".up_proj.", ".up.", ".mlp.up."],
    "down_proj": [".down_proj.", ".down.", ".mlp.down."],
    "embed_tokens": ["embed_tokens", "wte"],
    "lm_head": ["lm_head", "output_layer"],
}

def detect_layer_type(weight_key: str) -> LayerType:
    """
        Detect canonical layer type from a safetensors key.
    """

    key = weight_key.lower()

    for layer_type, patterns in _LAYER_PATTERNS.items():
        if any(pattern in key for pattern in patterns):
            return layer_type

    return "unknown"

def infer_weight_shape(weight_key: str, config: ModelConfig) -> tuple[int, ...]:
    """
        Infer original weight shape form layer name and ModelConfig.

        Handles GQA correctly: k_proj and v_proj use kv_dim instead of
        hidden_size.
    """

    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    vocab_size = config.vocab_size
    kv_dim = config.num_key_value_heads * config.head_dim

    shape_map: dict[LayerType, tuple[int, ...]] = {
        "q_proj": (hidden_size, hidden_size),
        "k_proj": (kv_dim, hidden_size),
        "v_proj": (kv_dim, hidden_size),
        "o_proj": (hidden_size, hidden_size),
        "qkv_proj": (3 * hidden_size, hidden_size),
        "gate_proj": (intermediate_size, hidden_size),
        "up_proj": (intermediate_size, hidden_size),
        "down_proj": (hidden_size, intermediate_size),
        "embed_tokens": (vocab_size, hidden_size),
        "lm_head": (vocab_size, hidden_size),
    }

    layer_type = detect_layer_type(weight_key)
    if layer_type in shape_map:
        return shape_map[layer_type]

    # Use hidden_size^2 as a generic fallback
    return (hidden_size, hidden_size)
