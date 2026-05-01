"""Tests for the Ollama scanner — manifest+blob walk without invoking the daemon."""

import hashlib
import json
from pathlib import Path

import pytest

from exo.shared.types.common import NodeId
from exo.sources.ollama import OllamaSource


def _write_blob(blobs_dir: Path, content: bytes) -> str:
    digest = hashlib.sha256(content).hexdigest()
    blob_path = blobs_dir / f"sha256-{digest}"
    blob_path.write_bytes(content)
    return f"sha256:{digest}"


def _write_manifest(
    manifests_dir: Path,
    *,
    host: str,
    namespace: str,
    name: str,
    tag: str,
    model_digest: str,
    model_size: int,
) -> Path:
    manifest_dir = manifests_dir / host / namespace / name
    manifest_dir.mkdir(parents=True)
    manifest = {
        "schemaVersion": 2,
        "config": {
            "mediaType": "application/vnd.docker.container.image.v1+json",
            "digest": "sha256:0",
            "size": 0,
        },
        "layers": [
            {
                "mediaType": "application/vnd.ollama.image.model",
                "digest": model_digest,
                "size": model_size,
            },
            {
                "mediaType": "application/vnd.ollama.image.template",
                "digest": "sha256:1",
                "size": 1,
            },
        ],
    }
    manifest_path = manifest_dir / tag
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


@pytest.fixture
def ollama_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "ollama"
    (root / "manifests").mkdir(parents=True)
    (root / "blobs").mkdir(parents=True)
    monkeypatch.setenv("EXO_OLLAMA_DIR", str(root))
    return root


def test_scan_resolves_default_registry_models(ollama_root: Path) -> None:
    blob_payload = b"GGUF" + b"\0" * 8000
    digest = _write_blob(ollama_root / "blobs", blob_payload)
    _write_manifest(
        ollama_root / "manifests",
        host="registry.ollama.ai",
        namespace="library",
        name="llama3",
        tag="8b",
        model_digest=digest,
        model_size=len(blob_payload),
    )
    src = OllamaSource()
    entries = list(src.scan(NodeId("node-1")))
    assert len(entries) == 1
    e = entries[0]
    assert e.external_id == "llama3:8b"
    assert e.format == "gguf"
    assert e.loadable_with_mlx is False
    assert e.size_bytes.in_bytes == len(blob_payload)
    assert Path(e.path).read_bytes() == blob_payload


def test_scan_keeps_namespace_for_non_default_registry(ollama_root: Path) -> None:
    digest = _write_blob(ollama_root / "blobs", b"GGUF" + b"\0" * 200)
    _write_manifest(
        ollama_root / "manifests",
        host="hf.co",
        namespace="bartowski",
        name="model",
        tag="latest",
        model_digest=digest,
        model_size=204,
    )
    src = OllamaSource()
    entries = list(src.scan(NodeId("node-1")))
    assert [e.external_id for e in entries] == ["hf.co/bartowski/model:latest"]


def test_resolve_path_finds_blob(ollama_root: Path) -> None:
    payload = b"GGUF" + b"\0" * 100
    digest = _write_blob(ollama_root / "blobs", payload)
    _write_manifest(
        ollama_root / "manifests",
        host="registry.ollama.ai",
        namespace="library",
        name="llama3",
        tag="8b",
        model_digest=digest,
        model_size=len(payload),
    )
    src = OllamaSource()
    resolved = src.resolve_path("llama3:8b")
    assert resolved is not None
    assert resolved.read_bytes() == payload


def test_skips_when_blob_missing(ollama_root: Path) -> None:
    _write_manifest(
        ollama_root / "manifests",
        host="registry.ollama.ai",
        namespace="library",
        name="llama3",
        tag="8b",
        model_digest="sha256:" + "ab" * 32,
        model_size=10,
    )
    # The referenced blob does not exist on disk.
    src = OllamaSource()
    assert list(src.scan(NodeId("node-1"))) == []


def test_is_unavailable_when_root_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EXO_OLLAMA_DIR", str(tmp_path / "missing"))
    src = OllamaSource()
    assert src.is_available() is False
