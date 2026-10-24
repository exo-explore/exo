"""Tests for Hugging Face polling optimization, authentication backoff, and 404 handling."""

from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiofiles
import aiofiles.os as aios
import pytest

from exo.download.download_utils import (
    HuggingFaceAuthenticationError,
    HuggingFaceNotFoundError,
    _failed_fetch_cache,
    download_shard,
    fetch_file_list_with_cache,
    fetch_file_list_with_retry,
    has_local_cache_file,
    has_local_model_files,
)
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.downloads import FileListEntry
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata


def _make_mock_session_returning(
    response_attrs: dict[str, object], method: str = "get"
) -> MagicMock:
    mock_response = MagicMock()
    for k, v in response_attrs.items():
        setattr(mock_response, k, v)

    mock_session = MagicMock()
    method_mock = getattr(mock_session, method)
    method_mock.return_value.__aenter__ = AsyncMock(return_value=mock_response)
    method_mock.return_value.__aexit__ = AsyncMock(return_value=None)

    mock_factory = MagicMock()
    mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_factory.return_value.__aexit__ = AsyncMock(return_value=None)
    return mock_factory


@pytest.fixture(autouse=True)
def clean_failed_fetch_cache() -> None:
    _failed_fetch_cache.clear()


@pytest.fixture
def model_id() -> ModelId:
    return ModelId("test-org/gated-model")


@pytest.fixture
def shard(model_id: ModelId) -> ShardMetadata:
    card = ModelCard(
        model_id=model_id,
        storage_size=Memory.from_mb(100),
        n_layers=28,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )
    return PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=28,
        n_layers=28,
    )


@pytest.fixture
async def temp_models_dir(tmp_path: Path) -> AsyncIterator[Path]:
    models_dir = tmp_path / "models"
    await aios.makedirs(models_dir, exist_ok=True)
    with (
        patch("exo.download.download_utils.EXO_MODELS_DIRS", (models_dir,)),
        patch("exo.download.download_utils.EXO_DEFAULT_MODELS_DIR", models_dir),
    ):
        yield models_dir


class TestDownloadShardStatusShortCircuit:
    async def test_uninstalled_model_no_remote_network_call(
        self, shard: ShardMetadata, temp_models_dir: Path
    ) -> None:
        """When skip_download=True and no local files or cache exist,
        download_shard must return not_started immediately without calling
        fetch_file_list_with_cache (Issue #2376)."""
        with patch(
            "exo.download.download_utils.fetch_file_list_with_cache",
            new_callable=AsyncMock,
        ) as mock_fetch:
            _path, progress = await download_shard(
                shard,
                on_progress=AsyncMock(),
                skip_download=True,
                skip_internet=False,
            )

            assert progress.status == "not_started"
            assert progress.downloaded.in_bytes == 0
            assert progress.completed_files == 0
            mock_fetch.assert_not_called()

    async def test_installed_model_with_cache_calls_fetch_cache(
        self, shard: ShardMetadata, temp_models_dir: Path
    ) -> None:
        """When a cached file list exists, skip_download=True proceeds to
        load file list and check file presence."""
        cache_dir = temp_models_dir / "caches" / shard.model_card.model_id.normalize()
        await aios.makedirs(cache_dir, exist_ok=True)
        cache_file = (
            cache_dir / f"{shard.model_card.model_id.normalize()}--main--file_list.json"
        )
        async with aiofiles.open(cache_file, "w") as f:
            await f.write("[]")

        with patch(
            "exo.download.download_utils.fetch_file_list_with_cache",
            new_callable=AsyncMock,
            return_value=[
                FileListEntry(type="file", path="model.safetensors", size=1000)
            ],
        ) as mock_fetch:
            _path, progress = await download_shard(
                shard,
                on_progress=AsyncMock(),
                skip_download=True,
                skip_internet=False,
            )

            assert progress.status == "not_started"
            mock_fetch.assert_called_once()


class TestLocalFileChecks:
    def test_has_local_model_files(self, model_id: ModelId, tmp_path: Path) -> None:
        model_dir = tmp_path / model_id.normalize()

        with patch("exo.download.download_utils.EXO_MODELS_DIRS", (tmp_path,)):
            # Directory does not exist
            assert not has_local_model_files(model_id)

            # Directory exists but is empty
            model_dir.mkdir(parents=True, exist_ok=True)
            assert not has_local_model_files(model_id)

            # File created inside directory
            (model_dir / "config.json").write_text("{}")
            assert has_local_model_files(model_id)

    async def test_has_local_cache_file(
        self, model_id: ModelId, temp_models_dir: Path
    ) -> None:
        assert not await has_local_cache_file(model_id)

        cache_dir = temp_models_dir / "caches" / model_id.normalize()
        await aios.makedirs(cache_dir, exist_ok=True)
        cache_file = cache_dir / f"{model_id.normalize()}--main--file_list.json"
        async with aiofiles.open(cache_file, "w") as f:
            await f.write("[]")

        assert await has_local_cache_file(model_id)


class TestFailedFetchBackoff:
    async def test_auth_error_recorded_in_backoff_cache(
        self, model_id: ModelId, temp_models_dir: Path
    ) -> None:
        """When fetch_file_list_with_retry raises HuggingFaceAuthenticationError,
        subsequent requests within backoff TTL skip the network call."""
        call_count = 0

        async def fake_fetch(*_args: object, **_kwargs: object) -> list[FileListEntry]:
            nonlocal call_count
            call_count += 1
            raise HuggingFaceAuthenticationError(f"Model {model_id} requires auth")

        with patch(
            "exo.download.download_utils.fetch_file_list_with_retry",
            side_effect=fake_fetch,
        ):
            # First call: hits network and fails
            with pytest.raises(FileNotFoundError, match="requires auth"):
                await fetch_file_list_with_cache(model_id)
            assert call_count == 1

            # Second call: within cooldown, skips network call
            with pytest.raises(FileNotFoundError, match="backing off"):
                await fetch_file_list_with_cache(model_id)
            assert call_count == 1

    async def test_token_change_bypasses_backoff(
        self, model_id: ModelId, temp_models_dir: Path
    ) -> None:
        """When HF_TOKEN changes, the backoff key changes and retries immediately."""
        call_count = 0

        async def fake_fetch(*_args: object, **_kwargs: object) -> list[FileListEntry]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise HuggingFaceAuthenticationError("Auth failed")
            return [FileListEntry(type="file", path="config.json", size=100)]

        with patch(
            "exo.download.download_utils.fetch_file_list_with_retry",
            side_effect=fake_fetch,
        ):
            with patch(
                "exo.download.download_utils.get_hf_token",
                new_callable=AsyncMock,
                return_value=None,
            ):
                with pytest.raises(FileNotFoundError):
                    await fetch_file_list_with_cache(model_id)
                assert call_count == 1

            # Token now provided by user
            with patch(
                "exo.download.download_utils.get_hf_token",
                new_callable=AsyncMock,
                return_value="hf_token_secret",
            ):
                result = await fetch_file_list_with_cache(model_id)
                assert call_count == 2
                assert len(result) == 1


class Test404Handling:
    async def test_404_raises_huggingface_not_found_without_5_retries(
        self, model_id: ModelId
    ) -> None:
        """HTTP 404 should immediately raise HuggingFaceNotFoundError and not retry 5 times."""
        mock_factory = _make_mock_session_returning({"status": 404})
        connection_lost_called = False

        def on_connection_lost() -> None:
            nonlocal connection_lost_called
            connection_lost_called = True

        with (
            patch(
                "exo.download.download_utils.create_http_session",
                mock_factory,
            ),
            patch(
                "exo.download.download_utils.get_download_headers",
                new_callable=AsyncMock,
                return_value={},
            ),
        ):
            with pytest.raises(HuggingFaceNotFoundError, match="HTTP 404"):
                await fetch_file_list_with_retry(
                    model_id, on_connection_lost=on_connection_lost
                )

            assert not connection_lost_called
            assert mock_factory.call_count == 1
