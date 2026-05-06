"""Tests for the LM Studio scanner."""

from pathlib import Path

import pytest

from exo.shared.types.common import NodeId
from exo.sources.lmstudio import LMStudioSource


def _make_mlx_model(parent: Path, publisher: str, name: str) -> Path:
    model_dir = parent / publisher / name
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"model_type": "llama"}')
    (model_dir / "model.safetensors").write_bytes(b"x" * 1024)
    return model_dir


def _make_gguf_model(parent: Path, publisher: str, name: str) -> Path:
    model_dir = parent / publisher / name
    model_dir.mkdir(parents=True)
    (model_dir / f"{name}.gguf").write_bytes(b"GGUF" + b"\0" * 1020)
    return model_dir


@pytest.fixture
def lm_studio_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "lmstudio_models"
    root.mkdir()
    monkeypatch.setenv("EXO_LMSTUDIO_DIR", str(root))
    return root


def test_scan_finds_mlx_and_gguf(lm_studio_root: Path) -> None:
    _make_mlx_model(lm_studio_root, "MLX", "Llama-3.2-1B-Instruct-4bit-MLX")
    _make_gguf_model(lm_studio_root, "Bartowski", "Llama-3.2-1B-Instruct-GGUF")
    src = LMStudioSource()
    entries = sorted(src.scan(NodeId("node-1")), key=lambda e: e.external_id)
    assert len(entries) == 2
    by_id = {e.external_id: e for e in entries}
    mlx = by_id["MLX/Llama-3.2-1B-Instruct-4bit-MLX"]
    assert mlx.format == "safetensors"
    assert mlx.loadable_with_mlx is True
    gguf = by_id["Bartowski/Llama-3.2-1B-Instruct-GGUF"]
    assert gguf.format == "gguf"
    assert gguf.loadable_with_mlx is False


def test_resolve_path(lm_studio_root: Path) -> None:
    expected = _make_mlx_model(lm_studio_root, "Pub", "Model-X")
    src = LMStudioSource()
    assert src.resolve_path("Pub/Model-X") == expected
    assert src.resolve_path("Other/Model-X") is None
    assert src.resolve_path("malformed-id") is None


def test_skips_directories_without_models(lm_studio_root: Path) -> None:
    publisher = lm_studio_root / "Pub" / "EmptyModel"
    publisher.mkdir(parents=True)
    (publisher / "README.md").write_text("nothing")
    src = LMStudioSource()
    assert list(src.scan(NodeId("node-1"))) == []


def test_is_unavailable_when_dir_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EXO_LMSTUDIO_DIR", str(tmp_path / "does-not-exist"))
    src = LMStudioSource()
    assert src.is_available() is False
    assert list(src.scan(NodeId("node-1"))) == []
