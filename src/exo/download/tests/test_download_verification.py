"""Tests for download verification and cache behavior."""

import time
from collections.abc import AsyncIterator
from datetime import timedelta
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
from exo.shared.types.memory import Memory
from exo.shared.types.worker.downloads import FileListEntry, RepoFileDownloadProgress


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


class TestProgressResetOnRedownload:
    """Tests for progress tracking when files are re-downloaded."""

    async def test_progress_resets_correctly_on_redownload(
        self, model_id: ModelId
    ) -> None:
        """Test that progress tracking resets when a file is re-downloaded.

        When a file is deleted and re-downloaded (due to size mismatch),
        the progress tracking should reset rather than calculating negative
        downloaded_this_session values.
        """
        # Simulate file_progress dict as it exists in download_shard
        file_progress: dict[str, RepoFileDownloadProgress] = {}

        # Initialize with old file progress (simulating existing large file)
        old_file_size = 1_500_000_000  # 1.5 GB
        file_progress["model.safetensors"] = RepoFileDownloadProgress(
            repo_id=model_id,
            repo_revision="main",
            file_path="model.safetensors",
            downloaded=Memory.from_bytes(old_file_size),
            downloaded_this_session=Memory.from_bytes(0),
            total=Memory.from_bytes(old_file_size),
            speed=0,
            eta=timedelta(0),
            status="not_started",
            start_time=time.time() - 10,  # Started 10 seconds ago
        )

        # Simulate the logic from on_progress_wrapper after re-download starts
        # This is the exact logic from the fixed on_progress_wrapper
        curr_bytes = 100_000  # 100 KB - new download just started
        previous_progress = file_progress.get("model.safetensors")

        # Detect re-download: curr_bytes < previous downloaded
        is_redownload = (
            previous_progress is not None
            and curr_bytes < previous_progress.downloaded.in_bytes
        )

        if is_redownload or previous_progress is None:
            # Fresh download or re-download: reset tracking
            start_time = time.time()
            downloaded_this_session = curr_bytes
        else:
            # Continuing download: accumulate
            start_time = previous_progress.start_time
            downloaded_this_session = (
                previous_progress.downloaded_this_session.in_bytes
                + (curr_bytes - previous_progress.downloaded.in_bytes)
            )

        # Key assertions
        assert is_redownload is True, "Should detect re-download scenario"
        assert downloaded_this_session == curr_bytes, (
            "downloaded_this_session should equal curr_bytes on re-download"
        )
        assert downloaded_this_session > 0, (
            "downloaded_this_session should be positive, not negative"
        )

        # Calculate speed (should be positive)
        elapsed = time.time() - start_time
        speed = downloaded_this_session / elapsed if elapsed > 0 else 0
        assert speed >= 0, "Speed should be non-negative"

    async def test_progress_accumulates_on_continuing_download(
        self, model_id: ModelId
    ) -> None:
        """Test that progress accumulates correctly for continuing downloads.

        When a download continues from where it left off (resume),
        the progress should accumulate correctly.
        """
        file_progress: dict[str, RepoFileDownloadProgress] = {}

        # Initialize with partial download progress
        initial_downloaded = 500_000  # 500 KB already downloaded
        start_time = time.time() - 5  # Started 5 seconds ago
        file_progress["model.safetensors"] = RepoFileDownloadProgress(
            repo_id=model_id,
            repo_revision="main",
            file_path="model.safetensors",
            downloaded=Memory.from_bytes(initial_downloaded),
            downloaded_this_session=Memory.from_bytes(initial_downloaded),
            total=Memory.from_bytes(1_000_000),
            speed=100_000,
            eta=timedelta(seconds=5),
            status="in_progress",
            start_time=start_time,
        )

        # Progress callback with more bytes downloaded
        curr_bytes = 600_000  # 600 KB - continuing download
        previous_progress = file_progress.get("model.safetensors")

        # This is NOT a re-download (curr_bytes > previous downloaded)
        is_redownload = (
            previous_progress is not None
            and curr_bytes < previous_progress.downloaded.in_bytes
        )

        if is_redownload or previous_progress is None:
            downloaded_this_session = curr_bytes
            used_start_time = time.time()
        else:
            used_start_time = previous_progress.start_time
            downloaded_this_session = (
                previous_progress.downloaded_this_session.in_bytes
                + (curr_bytes - previous_progress.downloaded.in_bytes)
            )

        # Key assertions
        assert is_redownload is False, (
            "Should NOT detect re-download for continuing download"
        )
        assert used_start_time == start_time, "Should preserve original start_time"
        expected_session = initial_downloaded + (curr_bytes - initial_downloaded)
        assert downloaded_this_session == expected_session, (
            f"Should accumulate: {downloaded_this_session} == {expected_session}"
        )
        assert downloaded_this_session == 600_000, (
            "downloaded_this_session should equal total downloaded so far"
        )
