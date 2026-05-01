"""Content-based resolution: find a model regardless of folder name."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from exo.download.download_utils import resolve_existing_model
from exo.shared.types.common import ModelId

MODEL_ID = ModelId("test-org/test-model")
NORMALIZED = MODEL_ID.normalize()

_BASE_CONFIG: dict[str, object] = {
    "model_type": "qwen3",
    "architectures": ["Qwen3ForCausalLM"],
    "hidden_size": 1024,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "intermediate_size": 3072,
    "vocab_size": 151936,
    "max_position_embeddings": 40960,
}


def _write_config(model_dir: Path, config: dict[str, object] | None = None) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(config or _BASE_CONFIG))


def _write_complete_model(
    model_dir: Path, config: dict[str, object] | None = None
) -> None:
    """Config + safetensors + index — passes ``is_model_directory_complete``."""
    _write_config(model_dir, config)
    weight_map = {"layer.weight": "model.safetensors"}
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 1024}, "weight_map": weight_map})
    )
    (model_dir / "model.safetensors").write_bytes(b"weights" * 100)


@pytest.fixture
def writable(tmp_path: Path) -> Path:
    out = tmp_path / "writable"
    out.mkdir()
    return out


@pytest.fixture
def readonly(tmp_path: Path) -> Path:
    out = tmp_path / "readonly"
    out.mkdir()
    return out


def test_finds_model_under_wrong_folder_name(writable: Path) -> None:
    """The user's reported scenario: model lives under an unconventional folder name."""
    _write_complete_model(writable / "wrong-typo")
    # Canonical dir has the config but no weights — simulates ``/models/add`` having
    # written the card but no download yet.
    _write_config(writable / NORMALIZED)
    with (
        patch("exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", ()),
        patch("exo.download.download_utils.EXO_MODELS_DIRS", (writable,)),
    ):
        resolved = resolve_existing_model(MODEL_ID)
    assert resolved == writable / "wrong-typo"


def test_convention_path_takes_precedence_over_content(writable: Path) -> None:
    """When both folders are complete, the canonical one wins (fast path)."""
    _write_complete_model(writable / NORMALIZED)
    _write_complete_model(writable / "wrong-typo")
    with (
        patch("exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", ()),
        patch("exo.download.download_utils.EXO_MODELS_DIRS", (writable,)),
    ):
        resolved = resolve_existing_model(MODEL_ID)
    assert resolved == writable / NORMALIZED


def test_partial_mispath_does_not_match(writable: Path) -> None:
    """Mispath has matching fingerprint but only a ``.partial`` weight — must not be selected."""
    target = writable / "wrong-typo"
    target.mkdir(parents=True)
    (target / "config.json").write_text(json.dumps(_BASE_CONFIG))
    (target / "model.safetensors.partial").write_bytes(b"half")
    _write_config(writable / NORMALIZED)
    with (
        patch("exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", ()),
        patch("exo.download.download_utils.EXO_MODELS_DIRS", (writable,)),
    ):
        resolved = resolve_existing_model(MODEL_ID)
    assert resolved is None


def test_returns_none_when_canonical_config_missing(writable: Path) -> None:
    """If the canonical config.json hasn't been fetched yet, we have nothing to match
    against, so the content pass cannot resolve."""
    _write_complete_model(writable / "wrong-typo")
    with (
        patch("exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", ()),
        patch("exo.download.download_utils.EXO_MODELS_DIRS", (writable,)),
    ):
        resolved = resolve_existing_model(MODEL_ID)
    assert resolved is None


def test_different_architecture_not_matched(writable: Path) -> None:
    _write_complete_model(
        writable / "wrong-typo", {**_BASE_CONFIG, "hidden_size": 2048}
    )
    _write_config(writable / NORMALIZED)
    with (
        patch("exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", ()),
        patch("exo.download.download_utils.EXO_MODELS_DIRS", (writable,)),
    ):
        resolved = resolve_existing_model(MODEL_ID)
    assert resolved is None


def test_read_only_dir_wins_in_content_pass(writable: Path, readonly: Path) -> None:
    """Content pass preserves the existing read-only-first precedence."""
    _write_complete_model(readonly / "ro-mispath")
    _write_complete_model(writable / "rw-mispath")
    _write_config(writable / NORMALIZED)
    with (
        patch("exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", (readonly,)),
        patch("exo.download.download_utils.EXO_MODELS_DIRS", (writable,)),
    ):
        resolved = resolve_existing_model(MODEL_ID)
    assert resolved == readonly / "ro-mispath"


def test_quantized_variant_not_matched_by_unquantized_canonical(writable: Path) -> None:
    """A 4-bit quant on disk must not be resolved when the user asked for the bf16 model."""
    _write_complete_model(
        writable / "wrong-typo",
        {**_BASE_CONFIG, "quantization": {"bits": 4, "group_size": 64}},
    )
    _write_config(writable / NORMALIZED)  # unquantized canonical config
    with (
        patch("exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", ()),
        patch("exo.download.download_utils.EXO_MODELS_DIRS", (writable,)),
    ):
        resolved = resolve_existing_model(MODEL_ID)
    assert resolved is None
