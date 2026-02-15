import json
from pathlib import Path

import pytest
from pydantic import ValidationError


def test_registered_architecture():
    """
        Tests the following architecture to be present in the
        architecture registry:
            - Llama
            - Qwen
            - Mistral
    """

    from exo.shared.architecture import ARCHITECTURE_REGISTRY

    architectures = [
        "LlamaForCausalLM",
        "Qwen2ForCausalLM",
        "MistralForCausalLM",
    ]

    assert all(arch in ARCHITECTURE_REGISTRY for arch in architectures)

def test_detect_llama():
    """
        detect_architecture() should find Llama from a new config
        dict.
    """

    from exo.shared.architecture import detect_architecture

    raw = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
    }

    spec = detect_architecture(raw)

    assert spec.name == "llama"
    assert spec.attention_type == "grouped_query"
    assert spec.mlp_type == "swiglu"
    assert spec.norm_type == "rms_norm"

def test_detect_unknown_raises():
    """
        Unsupported architectures should raise ValueError.
    """
    from exo.shared.architecture import detect_architecture

    with pytest.raises(ValueError, match="Unsupported Architecture"):
        detect_architecture({"architectures": ["FakeModelForCausalLM"]})

def test_architecture_spec_is_frozen():
    """
        Architectures must be immutable
    """
    from exo.shared.architecture import ARCHITECTURE_REGISTRY

    spec = ARCHITECTURE_REGISTRY["LlamaForCausalLM"]
    with pytest.raises(ValidationError):
        spec.name = "modified"  # type: ignore[misc]


def test_parse_model_config(tmp_path: Path):
    """
        parse_mode_config() must produce a correct ModelConfig
        from JSON.
    """

    from exo.shared.model_config import parse_model_config

    config = {
        "architectures" : ["LlamaForCausalLM"],
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 16,
        "vocab_size": 128256,
        "rope_theta": 500000.0,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-5,
    }

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    parsed = parse_model_config(config_path)
    assert parsed.num_attention_heads == 32
    assert parsed.num_key_value_heads == 8
    assert parsed.head_dim == 64  # 2048 // 32
    assert parsed.hidden_size == 2048
    assert parsed.quantization_config is None


def test_parse_model_config_with_quantization(tmp_path: Path):
    """
        Quantized models should populate quantization_config.
    """
    from exo.shared.model_config import parse_model_config

    config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 16,
        "vocab_size": 128256,
        "quantization_config": {"bits": 4, "group_size": 64},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    parsed = parse_model_config(config_path)
    assert parsed.quantization_config is not None
    assert parsed.quantization_config.bits == 4
    assert parsed.quantization_config.group_size == 64


def test_parse_model_config_defaults(tmp_path: Path):
    """
        Missing optional fields should use sensible defaults.
    """
    from exo.shared.model_config import parse_model_config

    config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "vocab_size": 128256,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    parsed = parse_model_config(config_path)
    assert parsed.num_key_value_heads == 32  # defaults to num_attention_heads
    assert parsed.tie_word_embeddings is False
    assert parsed.rope_scaling is None


def test_model_config_is_frozen(tmp_path: Path):
    """
        ModelConfig must be immutable (frozen=True).
    """
    from exo.shared.model_config import parse_model_config

    config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "vocab_size": 128256,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    parsed = parse_model_config(config_path)
    with pytest.raises(ValidationError):
        parsed.hidden_size = 4096  # type: ignore[misc]
