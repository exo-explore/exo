"""Peer-aware shard downloader that tries LAN peers before HuggingFace.

Wraps an existing ShardDownloader and adds a peer-download step: before
hitting HuggingFace, check if any peer on the LAN already has the model
(or is downloading it) and fetch from them instead. Falls back to the
inner downloader (HF) if peer download fails.
"""

import asyncio
import time
from collections.abc import Awaitable
from datetime import timedelta
from pathlib import Path
from typing import AsyncIterator, Callable

from loguru import logger

from exo.download.download_utils import (
    RepoDownloadProgress,
    calculate_repo_progress,
    ensure_models_dir,
    fetch_file_list_with_cache,
    is_image_model,
    resolve_allow_patterns,
)
from exo.download.huggingface_utils import filter_repo_objects
from exo.download.peer_download import (
    download_file_from_peer,
    get_peer_file_status,
)
from exo.download.peer_state import PeerStateProvider
from exo.download.shard_downloader import ShardDownloader
from exo.shared.types.memory import Memory
from exo.shared.types.worker.downloads import RepoFileDownloadProgress
from exo.shared.types.worker.shards import ShardMetadata


class PeerAwareShardDownloader(ShardDownloader):
    """ShardDownloader that tries peer download before HuggingFace.

    Decorates an inner ShardDownloader (typically ResumableShardDownloader).
    On ensure_shard(), checks if any cluster peer already has the model
    and downloads from them over the LAN. Falls back to the inner
    downloader if no peer has it or the peer transfer fails.
    """

    def __init__(
        self,
        inner: ShardDownloader,
        peer_state_provider: PeerStateProvider,
    ) -> None:
        self._inner = inner
        self._peer_state = peer_state_provider
        self._progress_callbacks: list[
            Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]]
        ] = []

    def on_progress(
        self,
        callback: Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]],
    ) -> None:
        self._inner.on_progress(callback)
        self._progress_callbacks.append(callback)

    async def ensure_shard(
        self, shard: ShardMetadata, config_only: bool = False
    ) -> Path:
        if config_only:
            return await self._inner.ensure_shard(shard, config_only=True)

        model_id = shard.model_card.model_id
        normalized = model_id.normalize()

        # Check if any peer has this model
        peers = self._peer_state.get_peers_for_model(normalized)
        if not peers:
            logger.debug(f"No peers have {model_id}, downloading from HuggingFace")
            return await self._inner.ensure_shard(shard, config_only=False)

        # Try each peer (completed peers first)
        for peer in peers:
            logger.info(
                f"Attempting peer download of {model_id} from "
                f"{peer.ip} (status: {peer.status})"
            )
            result = await self._try_peer_download(
                shard, peer.ip, self._peer_state.peer_download_port, normalized
            )
            if result is not None:
                logger.info(
                    f"Successfully downloaded {model_id} from peer {peer.ip}"
                )
                return result
            logger.info(
                f"Peer download from {peer.ip} failed, trying next peer or HuggingFace"
            )

        # All peers failed, fall back to HuggingFace
        logger.info(f"All peer downloads failed for {model_id}, falling back to HuggingFace")
        return await self._inner.ensure_shard(shard, config_only=False)

    async def _try_peer_download(
        self,
        shard: ShardMetadata,
        peer_ip: str,
        peer_port: int,
        model_id_normalized: str,
    ) -> Path | None:
        """Attempt to download all model files from a single peer.

        Returns the model directory path on success, None on failure.
        """
        # First, check what the peer has
        peer_files = await get_peer_file_status(
            peer_ip, peer_port, model_id_normalized
        )
        if not peer_files:
            return None

        peer_file_map = {f.path: f for f in peer_files}

        # Get the file list we need (same logic as download_shard)
        revision = "main"
        target_dir = await ensure_models_dir() / model_id_normalized

        try:
            file_list = await fetch_file_list_with_cache(
                shard.model_card.model_id,
                revision,
                recursive=True,
                skip_internet=False,
            )
        except Exception:
            # Can't get file list - fall back
            return None

        allow_patterns = await resolve_allow_patterns(shard)
        filtered_file_list = list(
            filter_repo_objects(
                file_list, allow_patterns=allow_patterns, key=lambda x: x.path
            )
        )

        if is_image_model(shard):
            filtered_file_list = [
                f
                for f in filtered_file_list
                if "/" in f.path or not f.path.endswith(".safetensors")
            ]

        # Check the peer has all (or most) files we need
        files_on_peer = 0
        for f in filtered_file_list:
            if f.path in peer_file_map:
                files_on_peer += 1

        if files_on_peer == 0:
            logger.debug(f"Peer has no files we need for {model_id_normalized}")
            return None

        # Download from peer with progress tracking
        all_start_time = time.time()
        file_progress: dict[str, RepoFileDownloadProgress] = {}
        semaphore = asyncio.Semaphore(8)
        failed = False

        async def download_one(file_path: str, expected_size: int) -> bool:
            """Download a single file from peer. Returns True on success."""

            def on_file_progress(
                curr_bytes: int, total_bytes: int, is_renamed: bool
            ) -> None:
                file_progress[file_path] = RepoFileDownloadProgress(
                    repo_id=str(shard.model_card.model_id),
                    repo_revision=revision,
                    file_path=file_path,
                    downloaded=Memory.from_bytes(curr_bytes),
                    downloaded_this_session=Memory.from_bytes(curr_bytes),
                    total=Memory.from_bytes(total_bytes),
                    speed=curr_bytes / max(time.time() - all_start_time, 0.1),
                    eta=timedelta(
                        seconds=(total_bytes - curr_bytes)
                        / max(
                            curr_bytes / max(time.time() - all_start_time, 0.1),
                            0.1,
                        )
                    ),
                    status="complete" if is_renamed else "in_progress",
                    start_time=all_start_time,
                )
                # Fire progress callbacks
                progress = calculate_repo_progress(
                    shard,
                    shard.model_card.model_id,
                    revision,
                    file_progress,
                    all_start_time,
                )
                for cb in self._progress_callbacks:
                    asyncio.create_task(cb(shard, progress))

            async with semaphore:
                result = await download_file_from_peer(
                    peer_ip,
                    peer_port,
                    model_id_normalized,
                    file_path,
                    target_dir,
                    expected_size,
                    on_progress=on_file_progress,
                )
                return result is not None

        # Initialize progress for all files
        for f in filtered_file_list:
            file_progress[f.path] = RepoFileDownloadProgress(
                repo_id=str(shard.model_card.model_id),
                repo_revision=revision,
                file_path=f.path,
                downloaded=Memory.from_bytes(0),
                downloaded_this_session=Memory.from_bytes(0),
                total=Memory.from_bytes(f.size or 0),
                speed=0,
                eta=timedelta(0),
                status="not_started",
                start_time=all_start_time,
            )

        # Download all files in parallel
        tasks = []
        for f in filtered_file_list:
            if f.size is None or f.size == 0:
                continue
            peer_info = peer_file_map.get(f.path)
            if peer_info and peer_info.safe_bytes > 0:
                tasks.append(download_one(f.path, f.size))
            else:
                # Peer doesn't have this file yet - this means incomplete peer
                # We could still try for the files it has, but for simplicity
                # fail the whole peer download if any file is missing
                failed = True
                break

        if failed:
            return None

        results = await asyncio.gather(*tasks, return_exceptions=True)
        if any(isinstance(r, Exception) or r is False for r in results):
            return None

        # Emit final progress
        final_progress = calculate_repo_progress(
            shard,
            shard.model_card.model_id,
            revision,
            file_progress,
            all_start_time,
        )
        for cb in self._progress_callbacks:
            await cb(shard, final_progress)

        # Return path (same as download_shard does)
        gguf = next(
            (f for f in filtered_file_list if f.path.endswith(".gguf")), None
        )
        return (target_dir / gguf.path) if gguf else target_dir

    async def get_shard_download_status(
        self,
    ) -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:
        async for path, status in self._inner.get_shard_download_status():
            yield path, status

    async def get_shard_download_status_for_shard(
        self, shard: ShardMetadata
    ) -> RepoDownloadProgress:
        return await self._inner.get_shard_download_status_for_shard(shard)
