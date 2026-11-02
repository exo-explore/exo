"""Models on disk: where they are, whether they are complete, where a new one goes.

A model lives in ``<models dir>/<org>--<name>/``. A download writes each file as
``<path>.partial`` and renames it once its hash matched. The commit it pins and the
files it fetches are kept in the model dir's manifest, ``.exo-download.json``,
written before the first byte and marked complete after the last rename.

No network here.
"""

from __future__ import annotations

import fnmatch
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path

from loguru import logger
from pydantic import ValidationError

from exo.download.hub import RepoFile
from exo.shared.constants import (
    EXO_DEFAULT_MODELS_DIR,
    EXO_MODELS_DIRS,
    EXO_MODELS_READ_ONLY_DIRS,
)
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.common import ModelId
from exo.utils.pydantic_ext import FrozenModel

MANIFEST = ".exo-download.json"
PARTIAL_SUFFIX = ".partial"
SKIPPED = ("original/*", "metal/*")


class Manifest(FrozenModel):
    repo: str
    commit: str
    files: list[RepoFile]
    complete: bool = False

    @property
    def total(self) -> int:
        return sum(f.size for f in self.files)


# --- directories ---------------------------------------------------------------------


def default_dir() -> Path:
    return EXO_DEFAULT_MODELS_DIR


def writable_dirs() -> tuple[Path, ...]:
    return EXO_MODELS_DIRS


def read_only_dirs() -> tuple[Path, ...]:
    return EXO_MODELS_READ_ONLY_DIRS


def is_read_only(model_dir: Path) -> bool:
    return any(model_dir.is_relative_to(d) for d in read_only_dirs())


def candidates(model_id: ModelId) -> list[Path]:
    """Every existing dir for the model: read-only dirs first, then writable ones."""
    name = model_id.normalize()
    return [
        d / name for d in (*read_only_dirs(), *writable_dirs()) if (d / name).is_dir()
    ]


# --- the manifest --------------------------------------------------------------------


def read_manifest(model_dir: Path) -> Manifest | None:
    try:
        return Manifest.model_validate_json((model_dir / MANIFEST).read_bytes())
    except FileNotFoundError:
        return None
    except (OSError, ValidationError) as error:
        logger.warning(f"Ignoring unreadable {model_dir / MANIFEST}: {error}")
        return None


def write_manifest(model_dir: Path, manifest: Manifest) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=model_dir, prefix=MANIFEST + ".", delete=False
    ) as temporary:
        temporary.write(manifest.model_dump_json(by_alias=True))
    os.replace(temporary.name, model_dir / MANIFEST)


# --- which files, and are they all there ---------------------------------------------


def is_image_model(card: ModelCard) -> bool:
    return ModelTask.TextToImage in card.tasks or ModelTask.ImageToImage in card.tasks


def wanted_files(
    files: Iterable[RepoFile],
    *,
    image_model: bool = False,
    only: tuple[str, ...] | None = None,
) -> list[RepoFile]:
    """The files exo downloads: all but ``original/`` and ``metal/``; for image models
    not the root ``*.safetensors`` (their weights are in component folders)."""
    return [
        f
        for f in files
        if not any(fnmatch.fnmatch(f.path, p) for p in SKIPPED)
        and not (image_model and "/" not in f.path and f.path.endswith(".safetensors"))
        and (only is None or any(fnmatch.fnmatch(f.path, p) for p in only))
    ]


def file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def is_complete(model_dir: Path, *, vision_weights: bool = False) -> bool:
    """Whether every file exo downloads for this model is here and was verified.

    With a manifest: it says so, and every listed file has its listed size (files
    are only renamed into place after their hash matched). Without one (older exo,
    copied by hand, read-only dirs) older exo's local rule applies: a safetensors
    index and every weight file it names; for a separate vision weights repo, which
    older exo fetched without an index, ``config.json`` and its ``*.safetensors``.
    """
    manifest = read_manifest(model_dir)
    if manifest is not None:
        return manifest.complete and all(
            file_size(model_dir / f.path) == f.size for f in manifest.files
        )
    try:
        if _legacy_complete(model_dir):
            return True
        return vision_weights and _legacy_vision_complete(model_dir)
    except OSError as error:
        logger.warning(f"Can't read {model_dir}: {error}")
        return False


def _legacy_complete(model_dir: Path) -> bool:
    index_files = list(model_dir.glob("**/*.safetensors.index.json"))
    if not index_files:
        return False
    for index_file in index_files:
        try:
            weight_map = json.loads(index_file.read_text()).get("weight_map", {})  # pyright: ignore[reportAny]
        except (ValueError, AttributeError):
            return False
        if not isinstance(weight_map, dict):
            return False
        for name in set(weight_map.values()):  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
            if not (index_file.parent / str(name)).is_file():  # pyright: ignore[reportUnknownArgumentType]
                return False
    return True


def _legacy_vision_complete(model_dir: Path) -> bool:
    return (model_dir / "config.json").is_file() and any(
        model_dir.rglob("*.safetensors")
    )


def find_complete(model_id: ModelId, card: ModelCard | None = None) -> Path | None:
    """The dir holding a complete copy of the model (and of its vision weights repo)."""
    found = next((d for d in candidates(model_id) if is_complete(d)), None)
    if found is None:
        return None
    sibling = vision_repo(card)
    if sibling is not None and not any(
        is_complete(d, vision_weights=True) for d in candidates(sibling)
    ):
        return None
    return found


def vision_repo(card: ModelCard | None) -> ModelId | None:
    """A separate repo holding the card's vision weights, if any."""
    if (
        card is None
        or card.vision is None
        or card.vision.weights_repo == str(card.model_id)
    ):
        return None
    return ModelId(card.vision.weights_repo)


# --- bytes on disk -------------------------------------------------------------------


def present_bytes(model_dir: Path, file: RepoFile) -> int:
    """Bytes of ``file`` already downloaded into ``model_dir``."""
    final = file_size(model_dir / file.path)
    if final == file.size:
        return file.size
    partial = file_size(model_dir / (file.path + PARTIAL_SUFFIX)) or 0
    return partial if partial <= file.size else 0


def legacy_present_bytes(model_dir: Path) -> int:
    """Bytes of a model dir with no manifest (what can be read of it)."""
    total = 0
    for folder, _, names in os.walk(model_dir):
        for name in names:
            total += file_size(Path(folder) / name) or 0
    return total


def free_bytes(directory: Path) -> int:
    try:
        return shutil.disk_usage(directory).free
    except OSError:
        return 0


def choose_root(
    model_id: ModelId, files: list[RepoFile], reserved: Mapping[Path, int]
) -> tuple[Path, int]:
    """The writable models dir to download into, and the bytes still needed there.

    Prefers a dir already holding part of the model, if the rest fits there;
    otherwise the first with room for what it lacks. ``reserved`` is what other
    running downloads still need in each dir. Raises :class:`NotEnoughSpaceError`
    when nothing fits.
    """
    name = model_id.normalize()
    total = sum(f.size for f in files)
    roots = [d for d in writable_dirs() if d.is_dir() or d == default_dir()]
    started = [d for d in roots if (d / name).is_dir()]
    closest: tuple[int, int] | None = None  # (free, needed) where most is free
    for root in [*started, *[d for d in roots if d not in started]]:
        needed = total - sum(present_bytes(root / name, f) for f in files)
        existing = root if root.is_dir() else root.parent
        free = free_bytes(existing) - reserved.get(root, 0)
        if free >= needed:
            return root, needed
        if closest is None or free > closest[0]:
            closest = (free, needed)
    free, needed = closest or (0, total)
    raise NotEnoughSpaceError(
        f"Not enough disk space for {model_id}: needs {_gb(needed)} more, "
        f"{_gb(max(free, 0))} free."
    )


class NotEnoughSpaceError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def _gb(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.1f} GB"
    return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{n / 1e3:.0f} KB"


# --- deleting ------------------------------------------------------------------------


def forget_manifest(repo: ModelId) -> None:
    """Drop the repo's pinned commit, keeping its files."""
    for model_dir in candidates(repo):
        if not is_read_only(model_dir):
            (model_dir / MANIFEST).unlink(missing_ok=True)


def delete(model_id: ModelId) -> bool:
    """Remove the model from every writable dir (never from read-only ones)."""
    name = model_id.normalize()
    deleted = False
    for root in writable_dirs():
        model_dir = root / name
        # only ever a folder directly inside a models dir
        if Path(os.path.normpath(model_dir)).parent != Path(os.path.normpath(root)):
            logger.warning(f"Refusing to delete {model_dir}: not inside {root}")
            continue
        if model_dir.is_symlink():
            # linked in from elsewhere: remove the link, not what it points to
            model_dir.unlink()
            deleted = True
        elif model_dir.exists():
            shutil.rmtree(model_dir)
            deleted = True
    # file-list cache written by older exo
    shutil.rmtree(default_dir() / "caches" / name, ignore_errors=True)
    return deleted
