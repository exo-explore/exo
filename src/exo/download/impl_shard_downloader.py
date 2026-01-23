import asyncio
from collections.abc import Awaitable
from pathlib import Path
from typing import AsyncIterator, Callable

from loguru import logger

from exo.download.download_utils import RepoDownloadProgress, download_shard
from exo.download.node_address_book import NodeAddressBook
from exo.download.shard_downloader import ShardDownloader
from exo.shared.models.model_cards import MODEL_CARDS, ModelCard, ModelId
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.worker.shards import (
    PipelineShardMetadata,
    ShardMetadata,
)


def exo_shard_downloader(max_parallel_downloads: int = 8) -> ShardDownloader:
    return SingletonShardDownloader(
        CachedShardDownloader(ResumableShardDownloader(max_parallel_downloads))
    )


def exo_cluster_shard_downloader(
    *,
    node_id: NodeId,
    session_id: SessionId,
    node_address_book: NodeAddressBook,
    model_store_port: int,
    max_parallel_downloads: int = 8,
) -> ShardDownloader:
    return SingletonShardDownloader(
        CachedShardDownloader(
            ClusterAwareShardDownloader(
                node_id=node_id,
                session_id=session_id,
                node_address_book=node_address_book,
                model_store_port=model_store_port,
                max_parallel_downloads=max_parallel_downloads,
            )
        )
    )


async def build_base_shard(model_id: ModelId) -> ShardMetadata:
    model_card = await ModelCard.from_hf(model_id)
    return PipelineShardMetadata(
        model_card=model_card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=model_card.n_layers,
        n_layers=model_card.n_layers,
    )


async def build_full_shard(model_id: ModelId) -> PipelineShardMetadata:
    base_shard = await build_base_shard(model_id)
    return PipelineShardMetadata(
        model_card=base_shard.model_card,
        device_rank=base_shard.device_rank,
        world_size=base_shard.world_size,
        start_layer=base_shard.start_layer,
        end_layer=base_shard.n_layers,
        n_layers=base_shard.n_layers,
    )


class SingletonShardDownloader(ShardDownloader):
    def __init__(self, shard_downloader: ShardDownloader):
        self.shard_downloader = shard_downloader
        self.active_downloads: dict[ShardMetadata, asyncio.Task[Path]] = {}

    def on_progress(
        self,
        callback: Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]],
    ) -> None:
        self.shard_downloader.on_progress(callback)

    async def ensure_shard(
        self, shard: ShardMetadata, config_only: bool = False
    ) -> Path:
        if shard not in self.active_downloads:
            self.active_downloads[shard] = asyncio.create_task(
                self.shard_downloader.ensure_shard(shard, config_only)
            )
        try:
            return await self.active_downloads[shard]
        finally:
            if shard in self.active_downloads and self.active_downloads[shard].done():
                del self.active_downloads[shard]

    async def get_shard_download_status(
        self,
    ) -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:
        async for path, status in self.shard_downloader.get_shard_download_status():
            yield path, status

    async def get_shard_download_status_for_shard(
        self, shard: ShardMetadata
    ) -> RepoDownloadProgress:
        return await self.shard_downloader.get_shard_download_status_for_shard(shard)


class CachedShardDownloader(ShardDownloader):
    def __init__(self, shard_downloader: ShardDownloader):
        self.shard_downloader = shard_downloader
        self.cache: dict[tuple[str, ShardMetadata], Path] = {}

    def on_progress(
        self,
        callback: Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]],
    ) -> None:
        self.shard_downloader.on_progress(callback)

    async def ensure_shard(
        self, shard: ShardMetadata, config_only: bool = False
    ) -> Path:
        if (shard.model_card.model_id, shard) in self.cache:
            return self.cache[(shard.model_card.model_id, shard)]

        target_dir = await self.shard_downloader.ensure_shard(shard, config_only)
        self.cache[(shard.model_card.model_id, shard)] = target_dir
        return target_dir

    async def get_shard_download_status(
        self,
    ) -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:
        async for path, status in self.shard_downloader.get_shard_download_status():
            yield path, status

    async def get_shard_download_status_for_shard(
        self, shard: ShardMetadata
    ) -> RepoDownloadProgress:
        return await self.shard_downloader.get_shard_download_status_for_shard(shard)


class ResumableShardDownloader(ShardDownloader):
    def __init__(self, max_parallel_downloads: int = 8):
        self.max_parallel_downloads = max_parallel_downloads
        self.on_progress_callbacks: list[
            Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]]
        ] = []

    async def on_progress_wrapper(
        self, shard: ShardMetadata, progress: RepoDownloadProgress
    ) -> None:
        for callback in self.on_progress_callbacks:
            await callback(shard, progress)

    def on_progress(
        self,
        callback: Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]],
    ) -> None:
        self.on_progress_callbacks.append(callback)

    async def ensure_shard(
        self, shard: ShardMetadata, config_only: bool = False
    ) -> Path:
        allow_patterns = ["config.json"] if config_only else None

        target_dir, _ = await download_shard(
            shard,
            self.on_progress_wrapper,
            max_parallel_downloads=self.max_parallel_downloads,
            allow_patterns=allow_patterns,
        )
        return target_dir

    async def get_shard_download_status(
        self,
    ) -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:
        async def _status_for_model(
            model_id: ModelId,
        ) -> tuple[Path, RepoDownloadProgress]:
            """Helper coroutine that builds the shard for a model and gets its download status."""
            shard = await build_full_shard(model_id)
            return await download_shard(
                shard, self.on_progress_wrapper, skip_download=True
            )

        # Kick off download status coroutines concurrently
        tasks = [
            asyncio.create_task(_status_for_model(model_card.model_id))
            for model_card in MODEL_CARDS.values()
        ]

        for task in asyncio.as_completed(tasks):
            try:
                yield await task
            # TODO: except Exception
            except Exception as e:
                logger.error("Error downloading shard:", e)

    async def get_shard_download_status_for_shard(
        self, shard: ShardMetadata
    ) -> RepoDownloadProgress:
        _, progress = await download_shard(
            shard, self.on_progress_wrapper, skip_download=True
        )
        return progress


class ClusterAwareShardDownloader(ShardDownloader):
    """
    A shard downloader that downloads from Hugging Face on the elected master node,
    and from the master's model-store server on all other nodes.
    """

    def __init__(
        self,
        *,
        node_id: NodeId,
        session_id: SessionId,
        node_address_book: NodeAddressBook,
        model_store_port: int,
        max_parallel_downloads: int = 8,
    ) -> None:
        self.node_id = node_id
        self.session_id = session_id
        self.node_address_book = node_address_book
        self.model_store_port = model_store_port
        self.max_parallel_downloads = max_parallel_downloads
        self.on_progress_callbacks: list[
            Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]]
        ] = []

    def _resolve_model_store_endpoint(self) -> str | None:
        if self.node_id == self.session_id.master_node_id:
            return None

        master_ipv4 = self.node_address_book.get_ipv4(self.session_id.master_node_id)
        if master_ipv4 is None:
            return None

        return f"http://{master_ipv4}:{self.model_store_port}"

    async def on_progress_wrapper(
        self, shard: ShardMetadata, progress: RepoDownloadProgress
    ) -> None:
        for callback in self.on_progress_callbacks:
            await callback(shard, progress)

    def on_progress(
        self,
        callback: Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]],
    ) -> None:
        self.on_progress_callbacks.append(callback)

    async def ensure_shard(
        self, shard: ShardMetadata, config_only: bool = False
    ) -> Path:
        allow_patterns = ["config.json"] if config_only else None
        endpoint = self._resolve_model_store_endpoint()

        if endpoint is None and self.node_id != self.session_id.master_node_id:
            logger.warning(
                "Model store endpoint unavailable; falling back to Hugging Face downloads."
            )

        target_dir, _ = await download_shard(
            shard,
            self.on_progress_wrapper,
            max_parallel_downloads=self.max_parallel_downloads,
            allow_patterns=allow_patterns,
            endpoint=endpoint,
            retry_on_not_found=(endpoint is not None),
        )
        return target_dir

    async def get_shard_download_status(
        self,
    ) -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:
        async def _status_for_model(
            model_id: ModelId,
        ) -> tuple[Path, RepoDownloadProgress]:
            shard = await build_full_shard(model_id)
            endpoint = self._resolve_model_store_endpoint()
            return await download_shard(
                shard,
                self.on_progress_wrapper,
                skip_download=True,
                endpoint=endpoint,
                retry_on_not_found=(endpoint is not None),
            )

        tasks = [
            asyncio.create_task(_status_for_model(model_card.model_id))
            for model_card in MODEL_CARDS.values()
        ]

        for task in asyncio.as_completed(tasks):
            try:
                yield await task
            except Exception as e:
                logger.error("Error downloading shard:", e)

    async def get_shard_download_status_for_shard(
        self, shard: ShardMetadata
    ) -> RepoDownloadProgress:
        endpoint = self._resolve_model_store_endpoint()
        _, progress = await download_shard(
            shard,
            self.on_progress_wrapper,
            skip_download=True,
            endpoint=endpoint,
            retry_on_not_found=(endpoint is not None),
        )
        return progress
