"""Tests for content-based fingerprinting of model directories."""

import json
from pathlib import Path

from exo.download.fingerprint import fingerprint_config, fingerprint_directory


def test_same_logical_config_yields_same_fingerprint() -> None:
    """Reordering keys and whitespace are cosmetic — fingerprints must match."""
    a = {
        "model_type": "qwen3",
        "hidden_size": 1024,
        "num_hidden_layers": 28,
        "vocab_size": 151936,
    }
    b = {
        "vocab_size": 151936,
        "hidden_size": 1024,
        "num_hidden_layers": 28,
        "model_type": "qwen3",
    }
    assert fingerprint_config(a) == fingerprint_config(b)


def test_different_quantization_yields_different_fingerprint() -> None:
    """A 4-bit quant must not match its bf16 sibling."""
    base = {"model_type": "qwen3", "hidden_size": 1024, "num_hidden_layers": 28}
    bf16 = {**base}
    quant_4bit = {**base, "quantization": {"bits": 4, "group_size": 64}}
    assert fingerprint_config(bf16) != fingerprint_config(quant_4bit)


def test_different_architecture_yields_different_fingerprint() -> None:
    a = {"model_type": "qwen3", "hidden_size": 1024, "num_hidden_layers": 28}
    b = {"model_type": "qwen3", "hidden_size": 2048, "num_hidden_layers": 28}
    assert fingerprint_config(a) != fingerprint_config(b)


def test_cosmetic_only_fields_are_ignored() -> None:
    """Unrelated keys (``_name_or_path``, ``transformers_version``) must not affect the fingerprint."""
    base = {"model_type": "qwen3", "hidden_size": 1024, "num_hidden_layers": 28}
    enriched = {
        **base,
        "_name_or_path": "/some/local/path",
        "transformers_version": "4.45.0",
        "torch_dtype": "bfloat16",
    }
    assert fingerprint_config(base) == fingerprint_config(enriched)


def test_fingerprint_directory_returns_none_for_empty_dir(tmp_path: Path) -> None:
    assert fingerprint_directory(tmp_path) is None


def test_fingerprint_directory_returns_none_for_invalid_json(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("not actually json")
    assert fingerprint_directory(tmp_path) is None


def test_fingerprint_directory_matches_fingerprint_config(tmp_path: Path) -> None:
    """End-to-end: the directory variant must match the in-memory variant for the same config."""
    config = {
        "model_type": "qwen3",
        "hidden_size": 1024,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "intermediate_size": 3072,
        "vocab_size": 151936,
        "quantization": {"bits": 4, "group_size": 64},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert fingerprint_directory(tmp_path) == fingerprint_config(config)


def test_fingerprint_is_deterministic_across_calls() -> None:
    """Whatever opaque hash representation we use must be stable run-to-run."""
    config = {"model_type": "x", "hidden_size": 64, "num_hidden_layers": 1}
    assert fingerprint_config(config) == fingerprint_config(config)
