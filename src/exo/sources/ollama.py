"""Ollama model store scanner.

Ollama uses an OCI-style content-addressed layout under ``~/.ollama/models/``:

    manifests/{registry_host}/{namespace}/{model}/{tag}   # JSON manifest
    blobs/sha256-{hex}                                    # content-addressed blobs

To enumerate offline (without invoking the daemon) we walk every JSON file under
``manifests/``, parse its ``layers[]``, and follow any layer with media type
``application/vnd.ollama.image.model`` to its GGUF blob.

Override the root with ``OLLAMA_MODELS`` (the same env Ollama itself honors) or
``EXO_OLLAMA_DIR`` (used in tests).
"""

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast, final

from loguru import logger

from exo.shared.types.common import ModelSourceKind, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.local_models import LocalModelEntry

_MODEL_LAYER_MEDIA_TYPE = "application/vnd.ollama.image.model"


def _ollama_root() -> Path:
    override = os.environ.get("EXO_OLLAMA_DIR") or os.environ.get("OLLAMA_MODELS")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".ollama" / "models"


def _digest_to_path(root: Path, digest: str) -> Path | None:
    if not digest.startswith("sha256:"):
        return None
    blob = root / "blobs" / f"sha256-{digest.removeprefix('sha256:')}"
    return blob if blob.is_file() else None


def _parse_manifest(manifest_path: Path) -> tuple[str, int] | None:
    """Return ``(blob_digest, total_bytes)`` for the model layer, or ``None``."""
    try:
        with manifest_path.open() as f:
            manifest_raw: object = json.load(f)  # pyright: ignore[reportAny]
    except (OSError, ValueError):
        return None
    if not isinstance(manifest_raw, dict):
        return None
    manifest = cast(dict[str, Any], manifest_raw)
    layers_obj: object = manifest.get("layers")
    if not isinstance(layers_obj, list):
        return None
    layers: list[Any] = cast(list[Any], layers_obj)
    for layer_obj in layers:  # pyright: ignore[reportAny]
        if not isinstance(layer_obj, dict):
            continue
        layer = cast(dict[str, Any], layer_obj)
        media_type_obj: object = layer.get("mediaType")
        digest_obj: object = layer.get("digest")
        size_obj: object = layer.get("size")
        if (
            media_type_obj == _MODEL_LAYER_MEDIA_TYPE
            and isinstance(digest_obj, str)
            and isinstance(size_obj, int)
        ):
            return digest_obj, size_obj
    return None


@final
class OllamaSource:
    kind: ModelSourceKind = "ollama"
    display_name: str = "Ollama"

    def is_available(self) -> bool:
        return (_ollama_root() / "manifests").exists()

    def scan(self, node_id: NodeId) -> Iterable[LocalModelEntry]:
        root = _ollama_root()
        manifests_dir = root / "manifests"
        if not manifests_dir.is_dir():
            return ()

        entries: list[LocalModelEntry] = []
        try:
            manifest_files = [
                p
                for p in manifests_dir.rglob("*")
                if p.is_file() and not p.name.startswith(".")
            ]
        except OSError as exc:
            logger.warning(
                f"Failed to walk Ollama manifests at {manifests_dir}: {exc!r}"
            )
            return ()

        for manifest_path in manifest_files:
            parsed = _parse_manifest(manifest_path)
            if parsed is None:
                continue
            digest, size = parsed
            blob_path = _digest_to_path(root, digest)
            if blob_path is None:
                continue
            external_id = _manifest_to_id(manifest_path, manifests_dir)
            entries.append(
                LocalModelEntry(
                    node_id=node_id,
                    source=self.kind,
                    external_id=external_id,
                    display_name=external_id,
                    path=str(blob_path),
                    format="gguf",
                    size_bytes=Memory(in_bytes=int(size)),
                    loadable_with_mlx=False,
                )
            )
        return entries

    def resolve_path(self, external_id: str) -> Path | None:
        # Reverse _manifest_to_id and locate the GGUF blob.
        root = _ollama_root()
        manifests_dir = root / "manifests"
        if not manifests_dir.is_dir():
            return None
        for manifest_path in manifests_dir.rglob("*"):
            if not manifest_path.is_file():
                continue
            if _manifest_to_id(manifest_path, manifests_dir) != external_id:
                continue
            parsed = _parse_manifest(manifest_path)
            if parsed is None:
                continue
            return _digest_to_path(root, parsed[0])
        return None


def _manifest_to_id(manifest_path: Path, manifests_dir: Path) -> str:
    """Convert ``manifests/registry.ollama.ai/library/llama3/8b`` → ``llama3:8b``.

    For non-default registries we keep the host prefix. The tag (last path part)
    becomes the ``:tag`` suffix; the rest becomes the model name.
    """
    rel_parts = manifest_path.relative_to(manifests_dir).parts
    if len(rel_parts) < 3:
        return manifest_path.name
    *prefix_parts, name, tag = rel_parts
    host = prefix_parts[0] if prefix_parts else ""
    namespace_parts = prefix_parts[1:] if len(prefix_parts) > 1 else []
    if host in ("", "registry.ollama.ai") and namespace_parts == ["library"]:
        return f"{name}:{tag}"
    namespace = "/".join([*([host] if host else []), *namespace_parts])
    return f"{namespace}/{name}:{tag}" if namespace else f"{name}:{tag}"
