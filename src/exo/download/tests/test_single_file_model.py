"""Single-file safetensors model support.

Models that fit in one ``model.safetensors`` (e.g. ``Qwen/Qwen3-0.6B``, smaller MLX
quants) ship without ``model.safetensors.index.json``. exo's downloader and
"is this complete?" check used to require the index unconditionally — these tests
lock in the relaxation.
"""

from collections.abc import Awaitable, Callable
from pathlib import Path
from unittest.mock import patch

import pytest

from exo.download.download_utils import (
    _scan_model_directory,  # pyright: ignore[reportPrivateUsage] — needed for direct unit tests
    is_model_directory_complete,
)
from exo.shared.models.model_cards import fetch_safetensors_size
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory


def _make_single_file_model(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text('{"model_type": "qwen3"}')
    (model_dir / "model.safetensors").write_bytes(b"weights" * 100)
    (model_dir / "tokenizer.json").write_text("{}")


def test_scan_returns_entries_for_single_file_layout(tmp_path: Path) -> None:
    _make_single_file_model(tmp_path)
    entries = _scan_model_directory(tmp_path, recursive=True)
    assert entries is not None
    names = {e.path for e in entries}
    assert "model.safetensors" in names
    assert "config.json" in names
    # All on-disk entries should have a real size — completeness check needs this.
    assert all(e.size is not None for e in entries)


def test_is_complete_true_for_single_file(tmp_path: Path) -> None:
    _make_single_file_model(tmp_path)
    assert is_model_directory_complete(tmp_path) is True


def test_is_complete_false_for_partial_only(tmp_path: Path) -> None:
    """A partial download lives under ``model.safetensors.partial`` until rename — must not be marked complete."""
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "config.json").write_text('{"model_type": "qwen3"}')
    (tmp_path / "model.safetensors.partial").write_bytes(b"half")
    assert is_model_directory_complete(tmp_path) is False


def test_scan_returns_none_for_truly_empty_dir(tmp_path: Path) -> None:
    """Regression: an empty dir (no index, no weights) must still return None so callers
    know there's nothing to load."""
    assert _scan_model_directory(tmp_path, recursive=True) is None


@pytest.mark.asyncio
async def test_fetch_safetensors_size_falls_back_on_missing_index(
    tmp_path: Path,
) -> None:
    """When the remote has no index, ``fetch_safetensors_size`` consults
    ``huggingface_hub.model_info`` for the canonical size."""

    async def fake_download_raises_not_found(
        model_id: ModelId,
        revision: str,
        path: str,
        target_dir: Path,
        on_progress: Callable[[int, int, bool], None] = lambda _, __, ___: None,
    ) -> Path:
        raise FileNotFoundError(f"File not found: fake://{model_id}/{path}")

    async def fake_resolve_model_dir(model_id: ModelId) -> Path:
        return tmp_path

    class _FakeSafetensorsInfo:
        total: int = 1234

    class _FakeModelInfo:
        safetensors: _FakeSafetensorsInfo = _FakeSafetensorsInfo()

    def fake_model_info(_id: object, *_args: object, **_kw: object) -> _FakeModelInfo:
        return _FakeModelInfo()

    with (
        patch(
            "exo.download.download_utils.download_file_with_retry",
            fake_download_raises_not_found,
        ),
        patch(
            "exo.download.download_utils.resolve_model_dir",
            fake_resolve_model_dir,
        ),
        patch("exo.shared.models.model_cards.model_info", fake_model_info),
    ):
        size = await fetch_safetensors_size(ModelId("Qwen/Qwen3-0.6B"))

    assert size == Memory.from_bytes(1234)


@pytest.mark.asyncio
async def test_fetch_safetensors_size_raises_when_no_safetensors_info(
    tmp_path: Path,
) -> None:
    """If the index is absent AND HF reports no safetensors metadata, surface a real error."""

    async def fake_download_raises_not_found(
        model_id: ModelId,
        revision: str,
        path: str,
        target_dir: Path,
        on_progress: Callable[[int, int, bool], None] = lambda _, __, ___: None,
    ) -> Path:
        raise FileNotFoundError("missing")

    async def fake_resolve_model_dir(model_id: ModelId) -> Path:
        return tmp_path

    class _FakeNoSafetensors:
        safetensors = None

    def fake_model_info(
        _id: object, *_args: object, **_kw: object
    ) -> _FakeNoSafetensors:
        return _FakeNoSafetensors()

    with (
        patch(
            "exo.download.download_utils.download_file_with_retry",
            fake_download_raises_not_found,
        ),
        patch(
            "exo.download.download_utils.resolve_model_dir",
            fake_resolve_model_dir,
        ),
        patch("exo.shared.models.model_cards.model_info", fake_model_info),
        pytest.raises(ValueError, match="No safetensors info"),
    ):
        await fetch_safetensors_size(ModelId("nonexistent/repo"))


# Sanity: existing multi-file path keeps working.
def test_scan_still_handles_index_layout(tmp_path: Path) -> None:
    import json

    (tmp_path / "config.json").write_text('{"model_type": "x"}')
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"a")
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"b")
    weight_map = {
        "layer.0.weight": "model-00001-of-00002.safetensors",
        "layer.1.weight": "model-00002-of-00002.safetensors",
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 2}, "weight_map": weight_map})
    )
    entries = _scan_model_directory(tmp_path, recursive=True)
    assert entries is not None
    paths = {e.path for e in entries}
    assert "model-00001-of-00002.safetensors" in paths
    assert "model-00002-of-00002.safetensors" in paths
    assert is_model_directory_complete(tmp_path) is True


# unused, but keeps the type-checker honest about the awaitable signature shape.
_AwaitableFn = Callable[..., Awaitable[Path]]
