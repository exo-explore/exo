"""LM Studio model library scanner.

LM Studio stores models under ``~/.lmstudio/models/{publisher}/{model}/``. The model
slot is either an MLX-style directory (``config.json`` + ``*.safetensors``) or a folder
containing one or more GGUF files. We expose either as a :class:`LocalModelEntry`; only
MLX/safetensors layouts are flagged ``loadable_with_mlx``.

Override the root with ``EXO_LMSTUDIO_DIR`` (used in tests).
"""

import os
from collections.abc import Iterable
from pathlib import Path
from typing import final

from loguru import logger

from exo.shared.types.common import ModelSourceKind, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.local_models import LocalModelEntry
from exo.sources.base import classify_directory_format, directory_size_bytes


def _lmstudio_root() -> Path:
    override = os.environ.get("EXO_LMSTUDIO_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".lmstudio" / "models"


@final
class LMStudioSource:
    kind: ModelSourceKind = "lmstudio"
    display_name: str = "LM Studio"

    def is_available(self) -> bool:
        return _lmstudio_root().exists()

    def scan(self, node_id: NodeId) -> Iterable[LocalModelEntry]:
        root = _lmstudio_root()
        if not root.exists():
            return ()
        entries: list[LocalModelEntry] = []
        try:
            publishers = [p for p in root.iterdir() if p.is_dir()]
        except OSError as exc:
            logger.warning(f"Failed to read LM Studio root {root}: {exc!r}")
            return ()

        for publisher_dir in publishers:
            try:
                model_dirs = [m for m in publisher_dir.iterdir() if m.is_dir()]
            except OSError:
                continue
            for model_dir in model_dirs:
                fmt = classify_directory_format(model_dir)
                if fmt is None:
                    continue
                external_id = f"{publisher_dir.name}/{model_dir.name}"
                entries.append(
                    LocalModelEntry(
                        node_id=node_id,
                        source=self.kind,
                        external_id=external_id,
                        display_name=external_id,
                        path=str(model_dir),
                        format=fmt,
                        size_bytes=Memory(in_bytes=directory_size_bytes(model_dir)),
                        loadable_with_mlx=fmt in ("mlx", "safetensors"),
                    )
                )
        return entries

    def resolve_path(self, external_id: str) -> Path | None:
        if "/" not in external_id:
            return None
        publisher, model_name = external_id.split("/", 1)
        candidate = _lmstudio_root() / publisher / model_name
        return candidate if candidate.is_dir() else None
