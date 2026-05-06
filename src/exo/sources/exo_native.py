"""Scanner over exo's own writable cache (``EXO_MODELS_DIRS``).

Each ``models--{org}--{name}`` directory becomes one entry. Models are exposed even
when in-flight downloads are tracked in ``state.downloads``; the dashboard merges both
views so users see the same model in both places only when it really is mid-download.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import final

from loguru import logger

from exo.shared.constants import EXO_MODELS_DIRS, EXO_MODELS_READ_ONLY_DIRS
from exo.shared.types.common import ModelId, ModelSourceKind, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.local_models import LocalModelEntry
from exo.sources.base import classify_directory_format, directory_size_bytes


def _denormalize(dir_name: str) -> str:
    """Reverse ``ModelId.normalize`` (``foo--bar`` → ``foo/bar``).

    exo stores models at ``{models_dir}/{org--name}``; we restore the canonical form
    so external IDs match what the user typed.
    """
    return dir_name.replace("--", "/", 1)


@final
class ExoNativeSource:
    kind: ModelSourceKind = "exo"
    display_name: str = "exo"

    def is_available(self) -> bool:
        return any(d.exists() for d in (*EXO_MODELS_DIRS, *EXO_MODELS_READ_ONLY_DIRS))

    def scan(self, node_id: NodeId) -> Iterable[LocalModelEntry]:
        entries: list[LocalModelEntry] = []
        seen: set[str] = set()
        for root in (*EXO_MODELS_DIRS, *EXO_MODELS_READ_ONLY_DIRS):
            if not root.exists():
                continue
            try:
                for child in root.iterdir():
                    if not child.is_dir() or "--" not in child.name:
                        continue
                    if child.name in seen:
                        continue
                    seen.add(child.name)
                    fmt = classify_directory_format(child)
                    if fmt is None:
                        continue
                    external_id = _denormalize(child.name)
                    entries.append(
                        LocalModelEntry(
                            node_id=node_id,
                            source=self.kind,
                            external_id=external_id,
                            display_name=external_id,
                            path=str(child),
                            format=fmt,
                            size_bytes=Memory(in_bytes=directory_size_bytes(child)),
                            loadable_with_mlx=fmt in ("mlx", "safetensors"),
                            matched_model_id=ModelId(external_id),
                        )
                    )
            except OSError as exc:
                logger.warning(f"Failed to scan exo models dir {root}: {exc!r}")
        return entries

    def resolve_path(self, external_id: str) -> Path | None:
        normalized = ModelId(external_id).normalize()
        for root in (*EXO_MODELS_DIRS, *EXO_MODELS_READ_ONLY_DIRS):
            candidate = root / normalized
            if candidate.is_dir():
                return candidate
        return None
