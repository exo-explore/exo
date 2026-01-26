"""Tests for download verification and cache behavior."""

from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiofiles
import aiofiles.os as aios
import pytest
from pydantic import TypeAdapter

from exo.download.download_utils import (
    delete_model,
    fetch_file_list_with_cache,
)
from exo.shared.types.common import ModelId
from exo.shared.types.worker.downloads import FileListEntry


@pytest.fixture
def model_id() -> ModelId:
    return ModelId("test-org/test-model")


@pytest.fixture
async def temp_models_dir(tmp_path: Path) -> AsyncIterator[Path]:
    """Set up a temporary models directory for testing."""
    models_dir = tmp_path / "models"
    await aios.makedirs(models_dir, exist_ok=True)
    with patch("exo.download.download_utils.EXO_MODELS_DIR", models_dir):
        yield models_dir


class TestFileVerification:
    """Tests for file size verification in _download_file."""

    async def test_redownload_when_file_size_changes_upstream(
        self, model_id: ModelId, tmp_path: Path
    ) -> None:
        """Test that files with mismatched sizes are re-downloaded."""
        # Import inside test to allow patching
        from exo.download.download_utils import (
            _download_file,  # pyright: ignore[reportPrivateUsage]
        )

        target_dir = tmp_path / "downloads"
        await aios.makedirs(target_dir, exist_ok=True)

        # Create a local file with wrong size
        local_file = target_dir / "test.safetensors"
        async with aiofiles.open(local_file, "wb") as f:
            await f.write(b"local content")  # 13 bytes

        remote_size = 1000  # Different from local
        remote_hash = "abc123"

        with (
            patch(
                "exo.download.download_utils.file_meta",
                new_callable=AsyncMock,
                return_value=(remote_size, remote_hash),
            ) as mock_file_meta,
            patch(
                "exo.download.download_utils.create_http_session"
            ) as mock_session_factory,
        ):
            # Set up mock HTTP response for re-download
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.content.read = AsyncMock(  # pyright: ignore[reportAny]
                side_effect=[b"x" * remote_size, b""]
            )

            mock_session = MagicMock()
            mock_session.get.return_value.__aenter__ = AsyncMock(  # pyright: ignore[reportAny]
                return_value=mock_response
            )
            mock_session.get.return_value.__aexit__ = AsyncMock(  # pyright: ignore[reportAny]
                return_value=None
            )
            mock_session_factory.return_value.__aenter__ = AsyncMock(  # pyright: ignore[reportAny]
                return_value=mock_session
            )
            mock_session_factory.return_value.__aexit__ = AsyncMock(  # pyright: ignore[reportAny]
                return_value=None
            )

            # Mock calc_hash to return the expected hash
            with patch(
                "exo.download.download_utils.calc_hash",
                new_callable=AsyncMock,
                return_value=remote_hash,
            ):
                await _download_file(model_id, "main", "test.safetensors", target_dir)

            # file_meta should be called twice: once for verification, once for download
            assert mock_file_meta.call_count == 2

    async def test_skip_download_when_file_size_matches(
        self, model_id: ModelId, tmp_path: Path
    ) -> None:
        """Test that files with matching sizes are not re-downloaded."""
        from exo.download.download_utils import (
            _download_file,  # pyright: ignore[reportPrivateUsage]
        )

        target_dir = tmp_path / "downloads"
        await aios.makedirs(target_dir, exist_ok=True)

        # Create a local file
        local_file = target_dir / "test.safetensors"
        local_content = b"local content"
        async with aiofiles.open(local_file, "wb") as f:
            await f.write(local_content)

        remote_size = len(local_content)  # Same as local
        remote_hash = "abc123"

        with (
            patch(
                "exo.download.download_utils.file_meta",
                new_callable=AsyncMock,
                return_value=(remote_size, remote_hash),
            ) as mock_file_meta,
            patch(
                "exo.download.download_utils.create_http_session"
            ) as mock_session_factory,
        ):
            result = await _download_file(
                model_id, "main", "test.safetensors", target_dir
            )

            # Should return immediately without downloading
            assert result == local_file
            mock_file_meta.assert_called_once()
            mock_session_factory.assert_not_called()

    async def test_offline_fallback_uses_local_file(
        self, model_id: ModelId, tmp_path: Path
    ) -> None:
        """Test that local files are used when network is unavailable."""
        from exo.download.download_utils import (
            _download_file,  # pyright: ignore[reportPrivateUsage]
        )

        target_dir = tmp_path / "downloads"
        await aios.makedirs(target_dir, exist_ok=True)

        # Create a local file
        local_file = target_dir / "test.safetensors"
        async with aiofiles.open(local_file, "wb") as f:
            await f.write(b"local content")

        with (
            patch(
                "exo.download.download_utils.file_meta",
                new_callable=AsyncMock,
                side_effect=Exception("Network error"),
            ),
            patch(
                "exo.download.download_utils.create_http_session"
            ) as mock_session_factory,
        ):
            result = await _download_file(
                model_id, "main", "test.safetensors", target_dir
            )

            # Should return local file without attempting download
            assert result == local_file
            mock_session_factory.assert_not_called()


class TestFileListCache:
    """Tests for file list caching behavior."""

    async def test_fetch_fresh_and_update_cache(
        self, model_id: ModelId, tmp_path: Path
    ) -> None:
        """Test that fresh data is fetched and cache is updated."""
        models_dir = tmp_path / "models"

        file_list = [
            FileListEntry(type="file", path="model.safetensors", size=1000),
            FileListEntry(type="file", path="config.json", size=100),
        ]

        with (
            patch("exo.download.download_utils.EXO_MODELS_DIR", models_dir),
            patch(
                "exo.download.download_utils.fetch_file_list_with_retry",
                new_callable=AsyncMock,
                return_value=file_list,
            ) as mock_fetch,
        ):
            result = await fetch_file_list_with_cache(model_id, "main")

            assert result == file_list
            mock_fetch.assert_called_once()

            # Verify cache was written
            cache_file = (
                models_dir
                / "caches"
                / model_id.normalize()
                / f"{model_id.normalize()}--main--file_list.json"
            )
            assert await aios.path.exists(cache_file)

            async with aiofiles.open(cache_file, "r") as f:
                cached_data = TypeAdapter(list[FileListEntry]).validate_json(
                    await f.read()
                )
            assert cached_data == file_list

    async def test_fallback_to_cache_when_fetch_fails(
        self, model_id: ModelId, tmp_path: Path
    ) -> None:
        """Test that cached data is used when fetch fails."""
        models_dir = tmp_path / "models"
        cache_dir = models_dir / "caches" / model_id.normalize()
        await aios.makedirs(cache_dir, exist_ok=True)

        # Create cache file
        cached_file_list = [
            FileListEntry(type="file", path="model.safetensors", size=1000),
        ]
        cache_file = cache_dir / f"{model_id.normalize()}--main--file_list.json"
        async with aiofiles.open(cache_file, "w") as f:
            await f.write(
                TypeAdapter(list[FileListEntry]).dump_json(cached_file_list).decode()
            )

        with (
            patch("exo.download.download_utils.EXO_MODELS_DIR", models_dir),
            patch(
                "exo.download.download_utils.fetch_file_list_with_retry",
                new_callable=AsyncMock,
                side_effect=Exception("Network error"),
            ),
        ):
            result = await fetch_file_list_with_cache(model_id, "main")

            assert result == cached_file_list

    async def test_error_propagates_when_no_cache(
        self, model_id: ModelId, tmp_path: Path
    ) -> None:
        """Test that errors propagate when fetch fails and no cache exists."""
        models_dir = tmp_path / "models"

        with (
            patch("exo.download.download_utils.EXO_MODELS_DIR", models_dir),
            patch(
                "exo.download.download_utils.fetch_file_list_with_retry",
                new_callable=AsyncMock,
                side_effect=Exception("Network error"),
            ),
            pytest.raises(Exception, match="Network error"),
        ):
            await fetch_file_list_with_cache(model_id, "main")


class TestModelDeletion:
    """Tests for model deletion including cache cleanup."""

    async def test_delete_model_clears_cache(
        self, model_id: ModelId, tmp_path: Path
    ) -> None:
        """Test that deleting a model also deletes its cache."""
        models_dir = tmp_path / "models"
        model_dir = models_dir / model_id.normalize()
        cache_dir = models_dir / "caches" / model_id.normalize()

        # Create model and cache directories
        await aios.makedirs(model_dir, exist_ok=True)
        await aios.makedirs(cache_dir, exist_ok=True)

        # Add some files
        async with aiofiles.open(model_dir / "model.safetensors", "w") as f:
            await f.write("model data")
        async with aiofiles.open(cache_dir / "file_list.json", "w") as f:
            await f.write("[]")

        with patch("exo.download.download_utils.EXO_MODELS_DIR", models_dir):
            result = await delete_model(model_id)

            assert result is True
            assert not await aios.path.exists(model_dir)
            assert not await aios.path.exists(cache_dir)

    async def test_delete_model_only_cache_exists(
        self, model_id: ModelId, tmp_path: Path
    ) -> None:
        """Test deleting when only cache exists (model already deleted)."""
        models_dir = tmp_path / "models"
        cache_dir = models_dir / "caches" / model_id.normalize()

        # Only create cache directory
        await aios.makedirs(cache_dir, exist_ok=True)
        async with aiofiles.open(cache_dir / "file_list.json", "w") as f:
            await f.write("[]")

        with patch("exo.download.download_utils.EXO_MODELS_DIR", models_dir):
            result = await delete_model(model_id)

            # Returns False because model dir didn't exist
            assert result is False
            # But cache should still be cleaned up
            assert not await aios.path.exists(cache_dir)

    async def test_delete_nonexistent_model(
        self, model_id: ModelId, tmp_path: Path
    ) -> None:
        """Test deleting a model that doesn't exist."""
        models_dir = tmp_path / "models"
        await aios.makedirs(models_dir, exist_ok=True)

        with patch("exo.download.download_utils.EXO_MODELS_DIR", models_dir):
            result = await delete_model(model_id)

            assert result is False
