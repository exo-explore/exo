"""HuggingFace cache scanner.

Covers the standard HF hub cache (``~/.cache/huggingface/hub``) which is also where
``mlx-lm`` puts MLX checkpoints and where modern ``llama.cpp -hf`` lands GGUFs. We
classify each entry's format from the snapshot directory contents, and tag MLX-format
models loadable by exo's MLX engine.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import final

from huggingface_hub import scan_cache_dir
from huggingface_hub.errors import CacheNotFound
from loguru import logger

from exo.shared.types.common import ModelSourceKind, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.local_models import LocalModelEntry
from exo.sources.base import classify_directory_format


def _hf_cache_dir() -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


@final
class HuggingFaceSource:
    kind: ModelSourceKind = "huggingface"
    display_name: str = "HuggingFace"

    def is_available(self) -> bool:
        return _hf_cache_dir().exists()

    def scan(self, node_id: NodeId) -> Iterable[LocalModelEntry]:
        if not self.is_available():
            return ()
        try:
            info = scan_cache_dir(_hf_cache_dir())
        except CacheNotFound:
            return ()
        except Exception as exc:
            logger.warning(f"HF cache scan failed: {exc!r}")
            return ()

        entries: list[LocalModelEntry] = []
        for repo in info.repos:
            if repo.repo_type != "model" or not repo.revisions:
                continue
            # Pick the largest revision — usually "main"; only one in practice for users.
            latest = max(repo.revisions, key=lambda rev: rev.size_on_disk)
            snapshot_path = Path(latest.snapshot_path)
            fmt = classify_directory_format(snapshot_path)
            # Fallback: mlx-community org override even when config.json lacks quantization.
            if fmt == "safetensors" and repo.repo_id.startswith("mlx-community/"):
                fmt = "mlx"
            if fmt is None:
                continue
            entries.append(
                LocalModelEntry(
                    node_id=node_id,
                    source=self.kind,
                    external_id=repo.repo_id,
                    display_name=repo.repo_id,
                    path=str(snapshot_path),
                    format=fmt,
                    size_bytes=Memory(in_bytes=int(latest.size_on_disk)),
                    loadable_with_mlx=fmt in ("mlx", "safetensors"),
                )
            )
        return entries

    def resolve_path(self, external_id: str) -> Path | None:
        try:
            info = scan_cache_dir(_hf_cache_dir())
        except (CacheNotFound, OSError):
            return None
        for repo in info.repos:
            if repo.repo_id != external_id or not repo.revisions:
                continue
            latest = max(repo.revisions, key=lambda rev: rev.size_on_disk)
            return Path(latest.snapshot_path)
        return None
