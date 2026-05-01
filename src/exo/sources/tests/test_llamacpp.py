"""Tests for the llama.cpp standalone GGUF cache scanner."""

from pathlib import Path

import pytest

from exo.shared.types.common import NodeId
from exo.sources.llamacpp import LlamaCppSource


@pytest.fixture
def llamacpp_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "llamacpp_cache"
    root.mkdir()
    monkeypatch.setenv("EXO_LLAMACPP_DIR", str(root))
    return root


def test_scan_finds_ggufs(llamacpp_root: Path) -> None:
    payload = b"GGUF" + b"\0" * 1024
    (llamacpp_root / "Llama-3.2-1B-Q4_K_M.gguf").write_bytes(payload)
    (llamacpp_root / "subdir").mkdir()
    (llamacpp_root / "subdir" / "Phi-3-mini-Q5_K_M.gguf").write_bytes(payload)
    src = LlamaCppSource()
    entries = sorted(src.scan(NodeId("n-1")), key=lambda e: e.external_id)
    assert [e.external_id for e in entries] == [
        "Llama-3.2-1B-Q4_K_M",
        "Phi-3-mini-Q5_K_M",
    ]
    assert all(e.format == "gguf" for e in entries)
    assert all(not e.loadable_with_mlx for e in entries)
    assert all(e.size_bytes.in_bytes == len(payload) for e in entries)


def test_resolve_path(llamacpp_root: Path) -> None:
    target = llamacpp_root / "model-A.gguf"
    target.write_bytes(b"GGUF")
    src = LlamaCppSource()
    assert src.resolve_path("model-A") == target
    assert src.resolve_path("absent") is None


def test_unavailable_when_root_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EXO_LLAMACPP_DIR", str(tmp_path / "nope"))
    src = LlamaCppSource()
    assert src.is_available() is False
    assert list(src.scan(NodeId("n-1"))) == []
