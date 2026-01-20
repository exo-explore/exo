from abc import ABC, abstractmethod
from collections.abc import Awaitable
from copy import copy
from datetime import timedelta
from pathlib import Path
from typing import AsyncIterator, Callable

from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.memory import Memory
from exo.shared.types.worker.shards import (
    PipelineShardMetadata,
    ShardMetadata,
)
from exo.worker.download.download_utils import RepoDownloadProgress


# TODO: the PipelineShardMetadata getting reinstantiated is a bit messy. Should this be a classmethod?
class ShardDownloader(ABC):
    @abstractmethod
    async def ensure_shard(
        self, shard: ShardMetadata, config_only: bool = False
    ) -> Path:
        """
        Ensures that the shard is downloaded.
        Does not allow multiple overlapping downloads at once.
        If you try to download a Shard which overlaps a Shard that is already being downloaded,
        the download will be cancelled and a new download will start.

        Args:
            shard (Shard): The shard to download.
        """

    @abstractmethod
    def on_progress(
        self,
        callback: Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]],
    ) -> None:
        pass

    @abstractmethod
    async def get_shard_download_status(
        self,
    ) -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:
        """Get the download status of shards.

        Yields:
            tuple[Path, RepoDownloadProgress]: The path and progress of a shard download.
        """
        yield (Path("/tmp/noop_shard"), NOOP_DOWNLOAD_PROGRESS)

    @abstractmethod
    async def get_shard_download_status_for_shard(
        self, shard: ShardMetadata
    ) -> RepoDownloadProgress: ...


class NoopShardDownloader(ShardDownloader):
    async def ensure_shard(
        self, shard: ShardMetadata, config_only: bool = False
    ) -> Path:
        return Path("/tmp/noop_shard")

    def on_progress(
        self,
        callback: Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]],
    ) -> None:
        pass

    async def get_shard_download_status(
        self,
    ) -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:
        yield (
            Path("/tmp/noop_shard"),
            NOOP_DOWNLOAD_PROGRESS,
        )

    async def get_shard_download_status_for_shard(
        self, shard: ShardMetadata
    ) -> RepoDownloadProgress:
        dp = copy(NOOP_DOWNLOAD_PROGRESS)
        dp.shard = shard
        return dp


NOOP_DOWNLOAD_PROGRESS = RepoDownloadProgress(
    repo_id="noop",
    repo_revision="noop",
    shard=PipelineShardMetadata(
        model_card=ModelCard(
            model_id=ModelId("noop"),
            storage_size=Memory.from_bytes(0),
            n_layers=1,
            hidden_size=1,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
        ),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    ),
    completed_files=0,
    total_files=0,
    downloaded_bytes=Memory.from_bytes(0),
    downloaded_bytes_this_session=Memory.from_bytes(0),
    total_bytes=Memory.from_bytes(0),
    overall_speed=0,
    overall_eta=timedelta(seconds=0),
    status="complete",
)
