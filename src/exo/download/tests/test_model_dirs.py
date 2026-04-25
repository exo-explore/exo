"""Tests for multi-directory model resolution, download target selection, and deletion."""

import json
import shutil
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import patch

import aiofiles
import aiofiles.os as aios
import pytest

from exo.download.download_utils import (
    InsufficientDiskSpaceError,
    delete_model,
    is_read_only_model_dir,
    resolve_existing_model,
    select_download_dir,
)
from exo.shared.types.common import ModelId

MODEL_ID = ModelId("test-org/test-model")
NORMALIZED = MODEL_ID.normalize()


def _create_complete_model(model_dir: Path) -> None:
    """Create a minimal complete model directory on disk."""
    model_dir.mkdir(parents=True, exist_ok=True)
    weight_map = {"layer.weight": "model.safetensors"}
    index = {"metadata": {"total_size": 1024}, "weight_map": weight_map}
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index))
    (model_dir / "model.safetensors").write_bytes(b"weights")
    (model_dir / "config.json").write_text('{"model_type": "test"}')


def _create_incomplete_model(model_dir: Path) -> None:
    """Create a model directory missing weight files."""
    model_dir.mkdir(parents=True, exist_ok=True)
    weight_map = {"layer.weight": "model.safetensors"}
    index = {"metadata": {"total_size": 1024}, "weight_map": weight_map}
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index))
    # model.safetensors is missing


# ---------------------------------------------------------------------------
# resolve_existing_model
# ---------------------------------------------------------------------------


class TestResolveExistingModel:
    def test_returns_none_when_no_dirs_have_model(self, tmp_path: Path) -> None:
        writable = tmp_path / "writable"
        writable.mkdir()
        with (
            patch("exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", ()),
            patch("exo.download.download_utils.EXO_MODELS_DIRS", (writable,)),
        ):
            assert resolve_existing_model(MODEL_ID) is None

    def test_finds_model_in_writable_dir(self, tmp_path: Path) -> None:
        writable = tmp_path / "writable"
        _create_complete_model(writable / NORMALIZED)
        with (
            patch("exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", ()),
            patch("exo.download.download_utils.EXO_MODELS_DIRS", (writable,)),
        ):
            assert resolve_existing_model(MODEL_ID) == writable / NORMALIZED

    def test_finds_model_in_read_only_dir(self, tmp_path: Path) -> None:
        read_only = tmp_path / "readonly"
        _create_complete_model(read_only / NORMALIZED)
        writable = tmp_path / "writable"
        writable.mkdir()
        with (
            patch(
                "exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", (read_only,)
            ),
            patch("exo.download.download_utils.EXO_MODELS_DIRS", (writable,)),
        ):
            assert resolve_existing_model(MODEL_ID) == read_only / NORMALIZED

    def test_read_only_takes_priority_over_writable(self, tmp_path: Path) -> None:
        read_only = tmp_path / "readonly"
        _create_complete_model(read_only / NORMALIZED)
        writable = tmp_path / "writable"
        _create_complete_model(writable / NORMALIZED)
        with (
            patch(
                "exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", (read_only,)
            ),
            patch("exo.download.download_utils.EXO_MODELS_DIRS", (writable,)),
        ):
            result = resolve_existing_model(MODEL_ID)
            assert result == read_only / NORMALIZED

    def test_skips_incomplete_model(self, tmp_path: Path) -> None:
        incomplete = tmp_path / "incomplete"
        _create_incomplete_model(incomplete / NORMALIZED)
        complete = tmp_path / "complete"
        _create_complete_model(complete / NORMALIZED)
        with (
            patch(
                "exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", (incomplete,)
            ),
            patch("exo.download.download_utils.EXO_MODELS_DIRS", (complete,)),
        ):
            result = resolve_existing_model(MODEL_ID)
            assert result == complete / NORMALIZED

    def test_searches_multiple_read_only_dirs_in_order(self, tmp_path: Path) -> None:
        ro1 = tmp_path / "ro1"
        ro1.mkdir()
        ro2 = tmp_path / "ro2"
        _create_complete_model(ro2 / NORMALIZED)
        writable = tmp_path / "writable"
        writable.mkdir()
        with (
            patch("exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", (ro1, ro2)),
            patch("exo.download.download_utils.EXO_MODELS_DIRS", (writable,)),
        ):
            assert resolve_existing_model(MODEL_ID) == ro2 / NORMALIZED


# ---------------------------------------------------------------------------
# is_read_only_model_dir
# ---------------------------------------------------------------------------


class TestIsReadOnlyModelDir:
    def test_path_under_read_only_dir(self, tmp_path: Path) -> None:
        ro = tmp_path / "readonly"
        with patch("exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", (ro,)):
            assert is_read_only_model_dir(ro / NORMALIZED) is True

    def test_path_under_writable_dir(self, tmp_path: Path) -> None:
        writable = tmp_path / "writable"
        with patch("exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", ()):
            assert is_read_only_model_dir(writable / NORMALIZED) is False

    def test_path_not_under_any_read_only_dir(self, tmp_path: Path) -> None:
        ro = tmp_path / "readonly"
        other = tmp_path / "other"
        with patch("exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", (ro,)):
            assert is_read_only_model_dir(other / NORMALIZED) is False


# ---------------------------------------------------------------------------
# select_download_dir
# ---------------------------------------------------------------------------


class TestSelectDownloadDir:
    def test_picks_first_dir_with_enough_space(self, tmp_path: Path) -> None:
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()
        # Both exist on same filesystem so both have space; first wins
        with patch("exo.download.download_utils.EXO_MODELS_DIRS", (dir1, dir2)):
            assert select_download_dir(1) == dir1

    def test_skips_dir_without_enough_space(self, tmp_path: Path) -> None:
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        real_disk_usage = shutil.disk_usage

        def mock_disk_usage(path: str | Path) -> object:
            if Path(path).is_relative_to(dir1):
                real = real_disk_usage(path)
                return shutil._ntuple_diskusage(real.total, real.total, 0)  # pyright: ignore[reportPrivateUsage]
            return real_disk_usage(path)

        with (
            patch("exo.download.download_utils.EXO_MODELS_DIRS", (dir1, dir2)),
            patch("shutil.disk_usage", side_effect=mock_disk_usage),
        ):
            assert select_download_dir(1024) == dir2

    def test_raises_when_no_dir_has_space(self, tmp_path: Path) -> None:
        dir1 = tmp_path / "dir1"
        dir1.mkdir()

        real_disk_usage = shutil.disk_usage

        def mock_disk_usage(path: str | Path) -> object:
            real = real_disk_usage(path)
            return shutil._ntuple_diskusage(real.total, real.total, 0)  # pyright: ignore[reportPrivateUsage]

        with (
            patch("exo.download.download_utils.EXO_MODELS_DIRS", (dir1,)),
            patch("shutil.disk_usage", side_effect=mock_disk_usage),
            pytest.raises(InsufficientDiskSpaceError),
        ):
            select_download_dir(1024)

    def test_skips_nonexistent_dir(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "does-not-exist"
        with (
            patch("exo.download.download_utils.EXO_MODELS_DIRS", (nonexistent,)),
            pytest.raises(InsufficientDiskSpaceError),
        ):
            select_download_dir(1)

    def test_skips_dir_raising_oserror(self, tmp_path: Path) -> None:
        dir1 = tmp_path / "unmounted"
        dir2 = tmp_path / "ok"
        dir1.mkdir()
        dir2.mkdir()

        real_disk_usage = shutil.disk_usage

        def mock_disk_usage(path: str | Path) -> object:
            if Path(path).is_relative_to(dir1):
                raise OSError("device not mounted")
            return real_disk_usage(path)

        with (
            patch("exo.download.download_utils.EXO_MODELS_DIRS", (dir1, dir2)),
            patch("shutil.disk_usage", side_effect=mock_disk_usage),
        ):
            assert select_download_dir(1) == dir2


# ---------------------------------------------------------------------------
# delete_model
# ---------------------------------------------------------------------------


class TestDeleteModel:
    @pytest.fixture
    async def dirs(self, tmp_path: Path) -> AsyncIterator[tuple[Path, Path, Path]]:
        writable1 = tmp_path / "w1"
        writable2 = tmp_path / "w2"
        default = tmp_path / "default"
        await aios.makedirs(writable1, exist_ok=True)
        await aios.makedirs(writable2, exist_ok=True)
        await aios.makedirs(default, exist_ok=True)
        with (
            patch(
                "exo.download.download_utils.EXO_MODELS_DIRS",
                (writable1, writable2, default),
            ),
            patch("exo.download.download_utils.EXO_DEFAULT_MODELS_DIR", default),
        ):
            yield writable1, writable2, default

    async def test_deletes_from_writable_dir(
        self, dirs: tuple[Path, Path, Path]
    ) -> None:
        w1, _, _ = dirs
        model_dir = w1 / NORMALIZED
        await aios.makedirs(model_dir, exist_ok=True)
        async with aiofiles.open(model_dir / "weights.safetensors", "w") as f:
            await f.write("data")

        result = await delete_model(MODEL_ID)
        assert result is True
        assert not await aios.path.exists(model_dir)

    async def test_deletes_symlinked_model_target(
        self, dirs: tuple[Path, Path, Path], tmp_path: Path
    ) -> None:
        w1, _, _ = dirs
        target_dir = tmp_path / "external-model-store" / "test-model"
        _create_complete_model(target_dir)
        model_link = w1 / NORMALIZED
        model_link.symlink_to(target_dir, target_is_directory=True)

        result = await delete_model(MODEL_ID)

        assert result is True
        assert not model_link.exists()
        assert not target_dir.exists()

    async def test_rejects_symlink_target_that_is_not_a_model_dir(
        self, dirs: tuple[Path, Path, Path], tmp_path: Path
    ) -> None:
        w1, _, _ = dirs
        target_dir = tmp_path / "not-a-model"
        target_dir.mkdir()
        (target_dir / "notes.txt").write_text("not model data")
        model_link = w1 / NORMALIZED
        model_link.symlink_to(target_dir, target_is_directory=True)

        with pytest.raises(OSError, match="does not look like a model directory"):
            await delete_model(MODEL_ID)

        assert not model_link.exists()
        assert target_dir.exists()

    async def test_deletes_from_multiple_writable_dirs(
        self, dirs: tuple[Path, Path, Path]
    ) -> None:
        w1, w2, _ = dirs
        model_dir1 = w1 / NORMALIZED
        model_dir2 = w2 / NORMALIZED
        await aios.makedirs(model_dir1, exist_ok=True)
        await aios.makedirs(model_dir2, exist_ok=True)
        async with aiofiles.open(model_dir1 / "w.safetensors", "w") as f:
            await f.write("data")
        async with aiofiles.open(model_dir2 / "w.safetensors", "w") as f:
            await f.write("data")

        result = await delete_model(MODEL_ID)
        assert result is True
        assert not await aios.path.exists(model_dir1)
        assert not await aios.path.exists(model_dir2)

    async def test_cleans_cache_from_default_dir(
        self, dirs: tuple[Path, Path, Path]
    ) -> None:
        _, _, default = dirs
        cache_dir = default / "caches" / NORMALIZED
        await aios.makedirs(cache_dir, exist_ok=True)
        async with aiofiles.open(cache_dir / "file_list.json", "w") as f:
            await f.write("[]")

        await delete_model(MODEL_ID)
        assert not await aios.path.exists(cache_dir)

    async def test_returns_false_when_model_not_found(
        self, dirs: tuple[Path, Path, Path]
    ) -> None:
        result = await delete_model(MODEL_ID)
        assert result is False
