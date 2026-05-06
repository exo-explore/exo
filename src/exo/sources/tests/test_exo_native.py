"""Tests for the exo-native scanner over EXO_MODELS_DIRS."""

from pathlib import Path
from unittest.mock import patch

import pytest

from exo.shared.types.common import NodeId
from exo.sources.exo_native import ExoNativeSource


def _make_model(root: Path, model_id: str) -> Path:
    """Create a fake exo-cache model dir matching ``ModelId.normalize`` layout."""
    normalized = model_id.replace("/", "--")
    model_dir = root / normalized
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"model_type": "llama"}')
    (model_dir / "model.safetensors").write_bytes(b"x" * 4096)
    return model_dir


@pytest.fixture
def writable_dir(tmp_path: Path) -> Path:
    out = tmp_path / "writable"
    out.mkdir()
    return out


@pytest.fixture
def readonly_dir(tmp_path: Path) -> Path:
    out = tmp_path / "ro"
    out.mkdir()
    return out


def test_scan_collects_models(writable_dir: Path, readonly_dir: Path) -> None:
    _make_model(writable_dir, "test-org/model-A")
    _make_model(readonly_dir, "test-org/model-B")
    with (
        patch("exo.sources.exo_native.EXO_MODELS_DIRS", (writable_dir,)),
        patch("exo.sources.exo_native.EXO_MODELS_READ_ONLY_DIRS", (readonly_dir,)),
    ):
        src = ExoNativeSource()
        entries = sorted(src.scan(NodeId("node-1")), key=lambda e: e.external_id)
    assert [e.external_id for e in entries] == ["test-org/model-A", "test-org/model-B"]
    assert all(e.source == "exo" for e in entries)
    assert all(e.format == "safetensors" for e in entries)
    assert all(e.matched_model_id is not None for e in entries)


def test_dedupes_when_same_dir_in_both_lists(writable_dir: Path) -> None:
    _make_model(writable_dir, "x/y")
    with (
        patch("exo.sources.exo_native.EXO_MODELS_DIRS", (writable_dir,)),
        patch("exo.sources.exo_native.EXO_MODELS_READ_ONLY_DIRS", (writable_dir,)),
    ):
        src = ExoNativeSource()
        entries = list(src.scan(NodeId("node-1")))
    assert len(entries) == 1


def test_resolve_path(writable_dir: Path, readonly_dir: Path) -> None:
    expected = _make_model(writable_dir, "x/y")
    with (
        patch("exo.sources.exo_native.EXO_MODELS_DIRS", (writable_dir,)),
        patch("exo.sources.exo_native.EXO_MODELS_READ_ONLY_DIRS", (readonly_dir,)),
    ):
        src = ExoNativeSource()
        assert src.resolve_path("x/y") == expected
        assert src.resolve_path("absent/model") is None
