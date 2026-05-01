"""llama.cpp standalone GGUF cache scanner.

Modern llama.cpp ``-hf`` lands GGUFs in the HF cache (covered by ``HuggingFaceSource``),
but its standalone ``LLAMA_CACHE`` and per-OS cache dirs hold a flat directory of GGUF
files. We enumerate every ``*.gguf`` under that root.

Override the root with ``EXO_LLAMACPP_DIR`` (used in tests) or ``LLAMA_CACHE``.
"""

import os
import platform
from collections.abc import Iterable
from pathlib import Path
from typing import final

from loguru import logger

from exo.shared.types.common import ModelSourceKind, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.local_models import LocalModelEntry


def _llamacpp_root() -> Path:
    override = os.environ.get("EXO_LLAMACPP_DIR") or os.environ.get("LLAMA_CACHE")
    if override:
        return Path(override).expanduser()
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Caches" / "llama.cpp"
    if system == "Windows":
        local = os.environ.get("LOCALAPPDATA")
        base = Path(local) if local else Path.home() / "AppData" / "Local"
        return base / "llama.cpp" / "cache"
    return Path.home() / ".cache" / "llama.cpp"


@final
class LlamaCppSource:
    kind: ModelSourceKind = "llamacpp"
    display_name: str = "llama.cpp"

    def is_available(self) -> bool:
        return _llamacpp_root().exists()

    def scan(self, node_id: NodeId) -> Iterable[LocalModelEntry]:
        root = _llamacpp_root()
        if not root.exists():
            return ()
        entries: list[LocalModelEntry] = []
        try:
            ggufs = list(root.rglob("*.gguf"))
        except OSError as exc:
            logger.warning(f"Failed to walk llama.cpp cache {root}: {exc!r}")
            return ()
        for gguf in ggufs:
            if not gguf.is_file():
                continue
            try:
                size = gguf.stat().st_size
            except OSError:
                continue
            external_id = gguf.stem
            entries.append(
                LocalModelEntry(
                    node_id=node_id,
                    source=self.kind,
                    external_id=external_id,
                    display_name=external_id,
                    path=str(gguf),
                    format="gguf",
                    size_bytes=Memory(in_bytes=size),
                    loadable_with_mlx=False,
                )
            )
        return entries

    def resolve_path(self, external_id: str) -> Path | None:
        root = _llamacpp_root()
        if not root.exists():
            return None
        for gguf in root.rglob("*.gguf"):
            if gguf.is_file() and gguf.stem == external_id:
                return gguf
        return None
