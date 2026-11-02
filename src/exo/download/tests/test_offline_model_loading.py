import json
from pathlib import Path

import pytest

from exo.download.download_utils import (
    _scan_model_directory,
    fetch_file_list_with_cache,
    has_partial_files,
    is_local_model_folder_ready,
    is_model_directory_complete,
)
from exo.shared.models.model_cards import ModelId


def test_has_partial_files(tmp_path: Path):
    """Verify detection of in-progress .partial download fragments."""
    model_dir = tmp_path / "test_model"
    model_dir.mkdir()
    assert not has_partial_files(model_dir)

    # Add standard files
    (model_dir / "config.json").write_text("{}")
    (model_dir / "model.safetensors").write_bytes(b"dummy weight data")
    assert not has_partial_files(model_dir)

    # Add a .partial file in a nested subdirectory
    nested = model_dir / "sub"
    nested.mkdir()
    (nested / "shard_02.safetensors.partial").write_bytes(b"partial data")
    assert has_partial_files(model_dir)


def test_is_local_model_folder_ready(tmp_path: Path):
    """Verify non-empty folder validation and requirement of no .partial files (TODO #8)."""
    model_dir = tmp_path / "llama-offline"
    model_dir.mkdir()

    # Empty folder is not ready
    assert not is_local_model_folder_ready(model_dir)

    # Folder with only random unrelated text is not recognized
    (model_dir / "notes.txt").write_text("hello")
    assert not is_local_model_folder_ready(model_dir)

    # Folder with config.json and weights
    (model_dir / "config.json").write_text('{"architectures": ["LlamaForCausalLM"]}')
    (model_dir / "model.safetensors").write_bytes(b"XYZ" * 100)
    assert is_local_model_folder_ready(model_dir)

    # If a .partial file is present, it must be rejected as incomplete
    (model_dir / "download.partial").write_bytes(b"incomplete")
    assert not is_local_model_folder_ready(model_dir)


def test_scan_model_directory_offline_copy_without_index(tmp_path: Path):
    """Verify scanning offline pre-copied model directories without an index json."""
    model_dir = tmp_path / "qwen-offline"
    model_dir.mkdir()

    config_path = model_dir / "config.json"
    config_path.write_text('{"vocab_size": 32000}')

    weight_path = model_dir / "model.safetensors"
    weight_path.write_bytes(b"\x00\x01\x02\x03" * 256)

    # Must build valid FileListEntry list even without *.safetensors.index.json
    file_list = _scan_model_directory(model_dir, recursive=True)
    assert file_list is not None
    assert len(file_list) == 2

    paths = {f.path for f in file_list}
    assert "config.json" in paths
    assert "model.safetensors" in paths

    for f in file_list:
        assert f.size is not None
        assert f.size > 0

    # Directory must be considered complete
    assert is_model_directory_complete(model_dir)


def test_scan_model_directory_with_partial_file_fails(tmp_path: Path):
    """Verify directory with in-progress .partial file is rejected as incomplete."""
    model_dir = tmp_path / "incomplete-model"
    model_dir.mkdir()

    (model_dir / "config.json").write_text("{}")
    (model_dir / "weights.safetensors.partial").write_bytes(b"incomplete bytes")

    file_list = _scan_model_directory(model_dir, recursive=True)
    assert file_list is None
    assert not is_model_directory_complete(model_dir)


@pytest.mark.asyncio
async def test_fetch_file_list_with_cache_offline_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify fetch_file_list_with_cache succeeds offline with no pre-existing cache file."""
    # Point models dirs and cache dir to temporary path
    models_root = tmp_path / "models"
    models_root.mkdir()
    cache_root = tmp_path / "cache"
    cache_root.mkdir()

    model_id = ModelId("meta-llama/Llama-Offline-Test")
    normalized_name = model_id.normalize()
    local_model_dir = models_root / normalized_name
    local_model_dir.mkdir(parents=True)

    (local_model_dir / "config.json").write_text('{"hidden_size": 2048}')
    (local_model_dir / "model.safetensors").write_bytes(b"test weights" * 50)

    # Monkeypatch search dirs and cache dir
    monkeypatch.setattr("exo.download.download_utils.EXO_MODELS_DIRS", [models_root])
    monkeypatch.setattr("exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", [])

    async def _mock_ensure_cache_dir(_model_id: ModelId) -> Path:
        return cache_root

    monkeypatch.setattr("exo.download.download_utils.ensure_cache_dir", _mock_ensure_cache_dir)

    # Call with skip_internet=True and no pre-existing cache file
    entries = await fetch_file_list_with_cache(
        model_id=model_id,
        revision="main",
        recursive=True,
        skip_internet=True,
    )

    assert entries is not None
    assert len(entries) == 2
    paths = {e.path for e in entries}
    assert "config.json" in paths
    assert "model.safetensors" in paths

    # Verify that a local cache file was automatically constructed
    cache_files = list(cache_root.glob("*.json"))
    assert len(cache_files) == 1
    cached_content = json.loads(cache_files[0].read_text())
    assert len(cached_content) == 2
