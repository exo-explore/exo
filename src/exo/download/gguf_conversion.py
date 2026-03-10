import json
from pathlib import Path
from typing import Any

import mlx.core as mx
from loguru import logger

# ---------------------------------------------------------------------------
# Architecture mapping: GGUF general.architecture → HuggingFace
# ---------------------------------------------------------------------------

_ARCH_TO_HF: dict[str, list[str]] = {
    "llama": ["LlamaForCausalLM"],
    "qwen2": ["Qwen2ForCausalLM"],
    "qwen3": ["Qwen3ForCausalLM"],
    "mistral": ["MistralForCausalLM"],
    "phi3": ["Phi3ForCausalLM"],
    "gemma": ["GemmaForCausalLM"],
    "gemma2": ["Gemma2ForCausalLM"],
    "starcoder2": ["Starcoder2ForCausalLM"],
}

_ARCH_TO_MODEL_TYPE: dict[str, str] = {
    "llama": "llama",
    "qwen2": "qwen2",
    "qwen3": "qwen3",
    "mistral": "mistral",
    "phi3": "phi3",
    "gemma": "gemma",
    "gemma2": "gemma2",
    "starcoder2": "starcoder2",
}

# GGUF general.file_type → quantization config for mlx_lm
_FILE_TYPE_TO_QUANT: dict[int, dict[str, int] | None] = {
    0: None,  # ALL_F32
    1: None,  # MOSTLY_F16
    2: {"group_size": 32, "bits": 4},  # MOSTLY_Q4_0
    3: {"group_size": 32, "bits": 4},  # MOSTLY_Q4_1
    7: {"group_size": 32, "bits": 8},  # MOSTLY_Q8_0
}

# Maximum safetensors shard size in bytes (5 GB)
_MAX_SHARD_BYTES = 5 * 1024 * 1024 * 1024


# ---------------------------------------------------------------------------
# Weight name translation
# ---------------------------------------------------------------------------


def translate_gguf_weight_name(name: str) -> str:
    """Translate a GGUF tensor name to HuggingFace-style name.

    Vendored from: https://github.com/ml-explore/mlx-examples/blob/main/llms/gguf_llm/models.py
    (translate_weight_names function)

    Covers the standard Llama-family convention which is shared by
    Llama, Qwen, Mistral, Gemma, Phi, and most other transformer models.
    """
    name = name.replace("blk.", "model.layers.")
    name = name.replace("ffn_gate", "mlp.gate_proj")
    name = name.replace("ffn_down", "mlp.down_proj")
    name = name.replace("ffn_up", "mlp.up_proj")
    name = name.replace("attn_q", "self_attn.q_proj")
    name = name.replace("attn_k", "self_attn.k_proj")
    name = name.replace("attn_v", "self_attn.v_proj")
    name = name.replace("attn_output", "self_attn.o_proj")
    name = name.replace("attn_norm", "input_layernorm")
    name = name.replace("ffn_norm", "post_attention_layernorm")
    name = name.replace("token_embd", "model.embed_tokens")
    name = name.replace("output_norm", "model.norm")
    # "output" must come last — it's a substring of "attn_output" / "output_norm"
    if name == "output.weight":
        name = "lm_head.weight"
    return name


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------


def _meta_val(metadata: dict[str, Any], key: str) -> int | float | str:
    """Get a metadata value, converting mx.array scalars to Python types."""
    val = metadata[key]  # pyright: ignore[reportAny]
    if isinstance(val, mx.array):
        return int(val.item())  # pyright: ignore[reportAny]
    if isinstance(val, (int, float)):
        return val
    return str(val)  # pyright: ignore[reportAny]


def extract_config_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Build a HuggingFace-style config.json dict from GGUF metadata."""
    raw_arch: str = str(metadata.get("general.architecture", "llama"))  # pyright: ignore[reportAny]
    arch: str = raw_arch.lower()
    prefix = f"{arch}."

    def get(suffix: str, default: int | float | str | None = None) -> int | float | str | None:
        key = prefix + suffix
        if key not in metadata:
            return default
        return _meta_val(metadata, key)

    # Vocab size: prefer metadata field, fall back to tokenizer token count
    vocab_size: int | float | str | None = get("vocab_size")
    if vocab_size is None:
        tokens: Any = metadata.get("tokenizer.ggml.tokens")
        vocab_size = len(tokens) if tokens is not None else 0  # pyright: ignore[reportAny]

    config: dict[str, Any] = {
        "architectures": _ARCH_TO_HF.get(arch, ["LlamaForCausalLM"]),
        "model_type": _ARCH_TO_MODEL_TYPE.get(arch, arch),
        "hidden_size": get("embedding_length", 0),
        "num_hidden_layers": get("block_count", 0),
        "num_attention_heads": get("attention.head_count", 0),
        "num_key_value_heads": get("attention.head_count_kv"),
        "intermediate_size": get("feed_forward_length", 0),
        "vocab_size": vocab_size,
        "rms_norm_eps": get("attention.layer_norm_rms_epsilon", 1e-5),
        "rope_theta": get("rope.freq_base", 10000.0),
        "max_position_embeddings": get("context_length", 4096),
        "tie_word_embeddings": False,
    }

    # Remove None values
    config = {k: v for k, v in config.items() if v is not None}  # pyright: ignore[reportAny]

    # Quantization config
    file_type_raw: Any = metadata.get("general.file_type")
    if file_type_raw is not None:
        file_type = int(_meta_val(metadata, "general.file_type"))
        quant = _FILE_TYPE_TO_QUANT.get(file_type)
        if quant is not None:
            config["quantization_config"] = quant

    return config


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def _load_gguf(gguf_path: Path) -> tuple[dict[str, mx.array], dict[str, Any]]:
    """Load a GGUF file, returning (weights, metadata)."""
    result: Any = mx.load(str(gguf_path), return_metadata=True)  # pyright: ignore[reportAny]
    weights: dict[str, mx.array] = result[0]  # pyright: ignore[reportAny]
    metadata: dict[str, Any] = result[1]  # pyright: ignore[reportAny]
    return weights, metadata


def convert_gguf_to_safetensors(gguf_path: Path, output_dir: Path) -> None:
    """Convert a GGUF file to safetensors + config.json in output_dir.

    After conversion:
    - output_dir contains model*.safetensors, config.json,
      model.safetensors.index.json, and a .gguf_converted marker
    - The original .gguf file is deleted to save disk space
    """
    logger.info(f"Loading GGUF file: {gguf_path}")
    weights, metadata = _load_gguf(gguf_path)

    # Translate weight names
    translated: dict[str, mx.array] = {}
    for name, tensor in weights.items():
        new_name = translate_gguf_weight_name(name)
        translated[new_name] = tensor
    del weights  # Free memory

    # Extract and write config.json
    config = extract_config_from_metadata(metadata)
    config_path = output_dir / "config.json"
    if not config_path.exists():
        logger.info(f"Writing config.json to {config_path}")
        config_path.write_text(json.dumps(config, indent=2, default=str))

    # Shard weights and save as safetensors
    weight_map: dict[str, str] = {}
    shards: list[tuple[str, dict[str, mx.array]]] = []
    current_shard: dict[str, mx.array] = {}
    current_size = 0
    shard_idx = 0

    for name, tensor in sorted(translated.items()):
        tensor_bytes = tensor.nbytes
        if current_size + tensor_bytes > _MAX_SHARD_BYTES and current_shard:
            shard_idx += 1
            shards.append((f"model-{shard_idx:05d}", current_shard))
            current_shard = {}
            current_size = 0
        current_shard[name] = tensor
        current_size += tensor_bytes

    if current_shard:
        shard_idx += 1
        shards.append((f"model-{shard_idx:05d}", current_shard))

    total_shards = len(shards)
    total_size = 0

    for shard_name, shard_weights in shards:
        if total_shards == 1:
            filename = "model.safetensors"
        else:
            filename = f"{shard_name}-of-{total_shards:05d}.safetensors"

        shard_path = output_dir / filename
        logger.info(f"Writing {filename} ({len(shard_weights)} tensors)")
        mx.save_safetensors(str(shard_path), shard_weights)  # pyright: ignore[reportUnknownMemberType]

        for weight_name in shard_weights:
            weight_map[weight_name] = filename
            total_size += shard_weights[weight_name].nbytes

    del translated  # Free memory

    # Write safetensors index
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    index_path = output_dir / "model.safetensors.index.json"
    logger.info(f"Writing safetensors index to {index_path}")
    index_path.write_text(json.dumps(index, indent=2))

    # Write marker
    marker_path = output_dir / ".gguf_converted"
    marker_path.write_text(f"Converted from {gguf_path.name}\n")

    # Delete original GGUF file
    logger.info(f"Deleting original GGUF file: {gguf_path}")
    gguf_path.unlink()

    logger.info("GGUF → safetensors conversion complete")


# ---------------------------------------------------------------------------
# Helpers for model card creation
# ---------------------------------------------------------------------------


def is_gguf_repo(file_paths: list[str]) -> bool:
    """Check if a file list represents a GGUF-only repo (has .gguf but no .safetensors)."""
    has_gguf = any(p.endswith(".gguf") for p in file_paths)
    has_safetensors = any(p.endswith(".safetensors") for p in file_paths)
    return has_gguf and not has_safetensors


def select_gguf_file(file_paths: list[str], file_sizes: dict[str, int | None] | None = None) -> str | None:
    """Pick the best GGUF file from a file list.

    Prefers the largest file (usually highest quality quant).
    """
    gguf_files = [p for p in file_paths if p.endswith(".gguf")]
    if not gguf_files:
        return None

    if file_sizes:
        # Pick largest
        return max(gguf_files, key=lambda p: file_sizes.get(p, 0) or 0)

    # No size info — just pick first
    return gguf_files[0]
