"""Tests for offline/air-gapped mode."""

from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, patch

import aiofiles
import aiofiles.os as aios
import pytest

from exo.download.download_utils import (
    _download_file,  # pyright: ignore[reportPrivateUsage]
    download_file_with_retry,
    fetch_file_list_with_cache,
)
from exo.shared.types.common import ModelId
from exo.shared.types.worker.downloads import FileListEntry


@pytest.fixture
def model_id() -> ModelId:
    return ModelId("test-org/test-model")


@pytest.fixture
async def temp_models_dir(tmp_path: Path) -> AsyncIterator[Path]:
    models_dir = tmp_path / "models"
    await aios.makedirs(models_dir, exist_ok=True)
    with patch("exo.download.download_utils.EXO_MODELS_DIR", models_dir):
        yield models_dir


class TestDownloadFileOffline:
    """Tests for _download_file with skip_internet=True."""

    async def test_returns_local_file_without_http_verification(
        self, model_id: ModelId, tmp_path: Path
    ) -> None:
        """When skip_internet=True and file exists locally, return it immediately
        without making any HTTP calls (no file_meta verification)."""
        target_dir = tmp_path / "downloads"
        await aios.makedirs(target_dir, exist_ok=True)

        local_file = target_dir / "model.safetensors"
        async with aiofiles.open(local_file, "wb") as f:
            await f.write(b"model weights data")

        with patch(
            "exo.download.download_utils.file_meta",
            new_callable=AsyncMock,
        ) as mock_file_meta:
            result = await _download_file(
                model_id,
                "main",
                "model.safetensors",
                target_dir,
                skip_internet=True,
            )

            assert result == local_file
            mock_file_meta.assert_not_called()

    async def test_raises_file_not_found_for_missing_file(
        self, model_id: ModelId, tmp_path: Path
    ) -> None:
        """When skip_internet=True and file does NOT exist locally,
        raise FileNotFoundError instead of attempting download."""
        target_dir = tmp_path / "downloads"
        await aios.makedirs(target_dir, exist_ok=True)

        with pytest.raises(FileNotFoundError, match="offline mode"):
            await _download_file(
                model_id,
                "main",
                "missing_model.safetensors",
                target_dir,
                skip_internet=True,
            )

    async def test_returns_local_file_in_subdirectory(
        self, model_id: ModelId, tmp_path: Path
    ) -> None:
        """When skip_internet=True and file exists in a subdirectory,
        return it without HTTP calls."""
        target_dir = tmp_path / "downloads"
        subdir = target_dir / "transformer"
        await aios.makedirs(subdir, exist_ok=True)

        local_file = subdir / "diffusion_pytorch_model.safetensors"
        async with aiofiles.open(local_file, "wb") as f:
            await f.write(b"weights")

        with patch(
            "exo.download.download_utils.file_meta",
            new_callable=AsyncMock,
        ) as mock_file_meta:
            result = await _download_file(
                model_id,
                "main",
                "transformer/diffusion_pytorch_model.safetensors",
                target_dir,
                skip_internet=True,
            )

            assert result == local_file
            mock_file_meta.assert_not_called()


class TestDownloadFileWithRetryOffline:
    """Tests for download_file_with_retry with skip_internet=True."""

    async def test_propagates_skip_internet_to_download_file(
        self, model_id: ModelId, tmp_path: Path
    ) -> None:
        """Verify skip_internet is passed through to _download_file."""
        target_dir = tmp_path / "downloads"
        await aios.makedirs(target_dir, exist_ok=True)

        local_file = target_dir / "config.json"
        async with aiofiles.open(local_file, "wb") as f:
            await f.write(b'{"model_type": "qwen2"}')

        with patch(
            "exo.download.download_utils.file_meta",
            new_callable=AsyncMock,
        ) as mock_file_meta:
            result = await download_file_with_retry(
                model_id,
                "main",
                "config.json",
                target_dir,
                skip_internet=True,
            )

            assert result == local_file
            mock_file_meta.assert_not_called()

    async def test_file_not_found_does_not_retry(
        self, model_id: ModelId, tmp_path: Path
    ) -> None:
        """FileNotFoundError from offline mode should not trigger retries."""
        target_dir = tmp_path / "downloads"
        await aios.makedirs(target_dir, exist_ok=True)

        with pytest.raises(FileNotFoundError):
            await download_file_with_retry(
                model_id,
                "main",
                "nonexistent.safetensors",
                target_dir,
                skip_internet=True,
            )


class TestFetchFileListOffline:
    """Tests for fetch_file_list_with_cache with skip_internet=True."""

    async def test_uses_cached_file_list(
        self, model_id: ModelId, temp_models_dir: Path
    ) -> None:
        """When skip_internet=True and cache file exists, use it without network."""
        from pydantic import TypeAdapter

        cache_dir = temp_models_dir / "caches" / model_id.normalize()
        await aios.makedirs(cache_dir, exist_ok=True)

        cached_list = [
            FileListEntry(type="file", path="model.safetensors", size=1000),
            FileListEntry(type="file", path="config.json", size=200),
        ]
        cache_file = cache_dir / f"{model_id.normalize()}--main--file_list.json"
        async with aiofiles.open(cache_file, "w") as f:
            await f.write(
                TypeAdapter(list[FileListEntry]).dump_json(cached_list).decode()
            )

        with patch(
            "exo.download.download_utils.fetch_file_list_with_retry",
            new_callable=AsyncMock,
        ) as mock_fetch:
            result = await fetch_file_list_with_cache(
                model_id, "main", skip_internet=True
            )

            assert result == cached_list
            mock_fetch.assert_not_called()

    async def test_falls_back_to_local_directory_scan(
        self, model_id: ModelId, temp_models_dir: Path
    ) -> None:
        """When skip_internet=True and no cache but local files exist,
        build file list from local directory."""
        import json

        model_dir = temp_models_dir / model_id.normalize()
        await aios.makedirs(model_dir, exist_ok=True)

        async with aiofiles.open(model_dir / "config.json", "w") as f:
            await f.write('{"model_type": "qwen2"}')

        index_data = {
            "metadata": {},
            "weight_map": {"model.layers.0.weight": "model.safetensors"},
        }
        async with aiofiles.open(model_dir / "model.safetensors.index.json", "w") as f:
            await f.write(json.dumps(index_data))

        async with aiofiles.open(model_dir / "model.safetensors", "wb") as f:
            await f.write(b"x" * 500)

        with patch(
            "exo.download.download_utils.fetch_file_list_with_retry",
            new_callable=AsyncMock,
        ) as mock_fetch:
            result = await fetch_file_list_with_cache(
                model_id, "main", skip_internet=True
            )

            mock_fetch.assert_not_called()
            paths = {entry.path for entry in result}
            assert "config.json" in paths
            assert "model.safetensors" in paths

    async def test_raises_when_no_cache_and_no_local_files(
        self, model_id: ModelId, temp_models_dir: Path
    ) -> None:
        """When skip_internet=True and neither cache nor local files exist,
        raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="No internet"):
            await fetch_file_list_with_cache(model_id, "main", skip_internet=True)
