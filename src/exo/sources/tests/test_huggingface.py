"""Tests for the HuggingFace cache scanner."""

import json
from pathlib import Path

import pytest

from exo.shared.types.common import NodeId
from exo.sources.huggingface import HuggingFaceSource


def _make_repo(
    cache_root: Path,
    repo_id: str,
    *,
    files: dict[str, bytes],
    revision: str = "abc123",
) -> Path:
    """Build a minimal HF cache layout: ``models--{org}--{name}/snapshots/{rev}/...``."""
    repo_dir = cache_root / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / revision
    blobs = repo_dir / "blobs"
    refs = repo_dir / "refs"
    snapshot.mkdir(parents=True)
    blobs.mkdir(parents=True)
    refs.mkdir(parents=True)
    (refs / "main").write_text(revision)
    for fname, payload in files.items():
        blob_path = blobs / f"blob_{fname}"
        blob_path.write_bytes(payload)
        (snapshot / fname).symlink_to(blob_path)
    return snapshot


@pytest.fixture
def hf_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache = tmp_path / "hf_hub"
    cache.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    # huggingface_hub.constants is read at import time; reload to pick up the env.
    import importlib

    import huggingface_hub.constants as constants

    importlib.reload(constants)
    return cache


def test_scan_finds_safetensors_repo(hf_cache: Path) -> None:
    _make_repo(
        hf_cache,
        "meta-llama/Llama-3.2-1B-Instruct",
        files={
            "config.json": b'{"model_type": "llama"}',
            "model.safetensors": b"x" * 4096,
        },
    )
    src = HuggingFaceSource()
    entries = list(src.scan(NodeId("node-1")))
    assert len(entries) == 1
    e = entries[0]
    assert e.source == "huggingface"
    assert e.external_id == "meta-llama/Llama-3.2-1B-Instruct"
    assert e.format == "safetensors"
    assert e.loadable_with_mlx is True


def test_scan_classifies_mlx_community_as_mlx(hf_cache: Path) -> None:
    _make_repo(
        hf_cache,
        "mlx-community/Llama-3.2-1B-Instruct-4bit",
        files={
            "config.json": json.dumps(
                {"model_type": "llama", "quantization": {"group_size": 64, "bits": 4}}
            ).encode(),
            "model.safetensors": b"x" * 4096,
        },
    )
    src = HuggingFaceSource()
    entries = list(src.scan(NodeId("node-1")))
    assert [e.format for e in entries] == ["mlx"]
    assert entries[0].loadable_with_mlx is True


def test_resolve_path_returns_snapshot(hf_cache: Path) -> None:
    snapshot = _make_repo(
        hf_cache,
        "test-org/test-model",
        files={
            "config.json": b'{"model_type": "test"}',
            "model.safetensors": b"weights",
        },
    )
    src = HuggingFaceSource()
    resolved = src.resolve_path("test-org/test-model")
    assert resolved == snapshot


def test_scan_skips_repos_without_weights(hf_cache: Path) -> None:
    _make_repo(
        hf_cache,
        "empty/repo",
        files={"README.md": b"# nothing here"},
    )
    src = HuggingFaceSource()
    assert list(src.scan(NodeId("node-1"))) == []
