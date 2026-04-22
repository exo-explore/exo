import asyncio
from asyncio import create_task
from collections.abc import Awaitable
from pathlib import Path
from typing import AsyncIterator, Callable

from loguru import logger

from exo.download.download_utils import (
    RepoDownloadProgress,
    download_shard,
)
from exo.download.shard_downloader import ShardDownloader
from exo.shared.models.model_cards import (
    ModelCard,
    ModelId,
    ModelTask,
    get_model_cards,
)
from exo.shared.types.memory import Memory
from exo.shared.types.worker.shards import (
    PipelineShardMetadata,
    ShardMetadata,
)


def exo_shard_downloader(
    max_parallel_downloads: int = 8, offline: bool = False
) -> ShardDownloader:
    return SingletonShardDownloader(
        ResumableShardDownloader(max_parallel_downloads, offline=offline)
    )


async def build_base_shard(model_id: ModelId) -> ShardMetadata:
    model_card = await ModelCard.load(model_id)
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


class ResumableShardDownloader(ShardDownloader):
    def __init__(self, max_parallel_downloads: int = 8, offline: bool = False):
        self.max_parallel_downloads = max_parallel_downloads
        self.offline = offline
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

        has_vision_sibling = (
            not config_only
            and not self.offline
            and shard.model_card.vision is not None
            and shard.model_card.vision.weights_repo != str(shard.model_card.model_id)
        )

        async def main_progress(
            cb_shard: ShardMetadata, progress: RepoDownloadProgress
        ) -> None:
            if has_vision_sibling and progress.status == "complete":
                return
            await self.on_progress_wrapper(cb_shard, progress)

        target_dir, _ = await download_shard(
            shard,
            main_progress,
            max_parallel_downloads=self.max_parallel_downloads,
            allow_patterns=allow_patterns,
            skip_internet=self.offline,
        )

        if has_vision_sibling:
            vision_shard = self._build_vision_shard(shard)

            async def vision_progress(
                _cb_shard: ShardMetadata, progress: RepoDownloadProgress
            ) -> None:
                await self.on_progress_wrapper(shard, progress)

            await download_shard(
                vision_shard,
                vision_progress,
                max_parallel_downloads=self.max_parallel_downloads,
                allow_patterns=["*.safetensors", "config.json"],
                skip_internet=self.offline,
            )

        return target_dir

    async def _status_for_shard(
        self, shard: ShardMetadata
    ) -> tuple[Path, RepoDownloadProgress]:
        async def _noop(
            _cb_shard: ShardMetadata, _progress: RepoDownloadProgress
        ) -> None:
            return

        path, main_progress = await download_shard(
            shard,
            _noop,
            skip_download=True,
            skip_internet=self.offline,
        )

        has_vision_sibling = (
            shard.model_card.vision is not None
            and shard.model_card.vision.weights_repo != str(shard.model_card.model_id)
        )
        if not has_vision_sibling:
            return path, main_progress

        vision_shard = self._build_vision_shard(shard)
        _, vision_progress = await download_shard(
            vision_shard,
            _noop,
            skip_download=True,
            skip_internet=self.offline,
        )
        combined = self._combine_progress(shard, main_progress, vision_progress)
        return path, combined

    @staticmethod
    def _build_vision_shard(shard: ShardMetadata) -> PipelineShardMetadata:
        assert shard.model_card.vision is not None
        vision_card = ModelCard(
            model_id=ModelId(shard.model_card.vision.weights_repo),
            storage_size=Memory.from_bytes(0),
            n_layers=1,
            hidden_size=1,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
        )
        return PipelineShardMetadata(
            model_card=vision_card,
            device_rank=0,
            world_size=1,
            start_layer=0,
            end_layer=1,
            n_layers=1,
        )

    @staticmethod
    def _combine_progress(
        shard: ShardMetadata,
        main: RepoDownloadProgress,
        vision: RepoDownloadProgress,
    ) -> RepoDownloadProgress:
        status_rank = {"not_started": 0, "in_progress": 1, "complete": 2}
        combined_status = min(
            (main.status, vision.status), key=lambda s: status_rank[s]
        )
        file_progress = dict(main.file_progress)
        for file_path, fp in vision.file_progress.items():
            file_progress[f"{vision.repo_id}/{file_path}"] = fp
        return RepoDownloadProgress(
            repo_id=main.repo_id,
            repo_revision=main.repo_revision,
            shard=shard,
            completed_files=main.completed_files + vision.completed_files,
            total_files=main.total_files + vision.total_files,
            downloaded=main.downloaded + vision.downloaded,
            downloaded_this_session=main.downloaded_this_session
            + vision.downloaded_this_session,
            total=main.total + vision.total,
            overall_speed=main.overall_speed + vision.overall_speed,
            overall_eta=max(main.overall_eta, vision.overall_eta),
            status=combined_status,
            file_progress=file_progress,
        )

    async def get_shard_download_status(
        self,
    ) -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:
        async def _status_for_model(
            model_id: ModelId,
        ) -> tuple[Path, RepoDownloadProgress]:
            """Helper coroutine that builds the shard for a model and gets its download status."""
            shard = await build_full_shard(model_id)
            return await self._status_for_shard(shard)

        semaphore = asyncio.Semaphore(self.max_parallel_downloads)

        async def download_with_semaphore(
            model_card: ModelCard,
        ) -> tuple[Path, RepoDownloadProgress]:
            async with semaphore:
                return await _status_for_model(model_card.model_id)

        tasks = [
            create_task(download_with_semaphore(model_card))
            for model_card in await get_model_cards()
        ]

        for task in asyncio.as_completed(tasks):
            try:
                yield await task
            except Exception as e:
                logger.warning(f"Error downloading shard: {type(e).__name__}")

    async def get_shard_download_status_for_shard(
        self, shard: ShardMetadata
    ) -> RepoDownloadProgress:
        _, progress = await self._status_for_shard(shard)
        return progress
