import json
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable

from exo.shared.types.common import ModelFileFormat, ModelSourceKind, NodeId
from exo.shared.types.worker.local_models import LocalModelEntry


def classify_directory_format(model_dir: Path) -> ModelFileFormat | None:
    """Identify the weight format of an HF-style model directory.

    Returns ``"mlx"`` if the config carries a ``quantization`` block (the canonical
    marker for an MLX checkpoint), ``"safetensors"`` for plain HF weights,
    ``"gguf"`` for a directory of GGUFs, ``None`` if the directory isn't a usable model.

    This is shared between the exo_native, HuggingFace, and LMStudio scanners so all
    three apply identical heuristics.
    """
    if not model_dir.is_dir():
        return None
    has_config = (model_dir / "config.json").exists()
    safetensors_present = any(model_dir.glob("*.safetensors"))
    ggufs_present = any(model_dir.glob("*.gguf"))
    if has_config and safetensors_present:
        return "mlx" if _is_mlx_config(model_dir / "config.json") else "safetensors"
    if ggufs_present:
        return "gguf"
    return None


def _is_mlx_config(config_path: Path) -> bool:
    try:
        with config_path.open() as f:
            config_raw: object = json.load(f)  # pyright: ignore[reportAny]
    except (OSError, ValueError):
        return False
    return isinstance(config_raw, dict) and "quantization" in config_raw


def directory_size_bytes(path: Path) -> int:
    """Best-effort recursive byte count; tolerates missing files mid-scan."""
    total = 0
    try:
        for child in path.rglob("*"):
            if child.is_file():
                try:
                    total += child.stat().st_size
                except OSError:
                    continue
    except OSError:
        return 0
    return total


@runtime_checkable
class ModelSource(Protocol):
    """A scanner for one place where models live on disk.

    Implementations should be cheap to construct and tolerant of a missing cache
    directory (``is_available`` returns ``False`` rather than raising). ``scan`` is
    called repeatedly by the per-worker scanner service; it must not raise — return
    an empty iterable if the layout is malformed.
    """

    kind: ModelSourceKind
    display_name: str

    def is_available(self) -> bool:
        """Return ``True`` if this source's cache directory is configured and exists."""
        ...

    def scan(self, node_id: NodeId) -> Iterable[LocalModelEntry]:
        """Enumerate every local model this source can see, tagged with ``node_id``."""
        ...

    def resolve_path(self, external_id: str) -> Path | None:
        """Return the on-disk path for an entry's natural identifier, or ``None`` if absent.

        ``external_id`` matches the value the source set on :class:`LocalModelEntry`.
        Used by ``build_model_path`` to fall back to external sources when exo's own
        cache doesn't have the model.
        """
        ...
