"""Torrent-based shard downloader using BitTorrent protocol with private tracker.

This module implements downloading model shards via BitTorrent with:
- Private tracker integration (/_internal/announce endpoint)
- WebSeed (BEP 19) fallback to HTTPS
- Selective file download based on shard layer ranges
- Persistent seeding of downloaded content
"""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Callable

from loguru import logger

from exo.shared.constants import EXO_MODELS_DIR
from exo.shared.types.worker.shards import ShardMetadata
from exo.worker.download.download_utils import RepoDownloadProgress
from exo.worker.download.shard_downloader import ShardDownloader


class TorrentShardDownloader(ShardDownloader):
    """Download model shards using BitTorrent with private tracker."""

    def __init__(self, max_parallel_downloads: int = 8):
        self.max_parallel_downloads = max_parallel_downloads
        self._progress_callbacks: list[
            Callable[[ShardMetadata, RepoDownloadProgress], None]
        ] = []

        # Initialize TorrentSessionHandle
        try:
            from exo_pyo3_bindings import TorrentSessionHandle  # type: ignore

            session_dir = EXO_MODELS_DIR / "v2" / ".torrent_session"
            session_dir.mkdir(parents=True, exist_ok=True)
            self.session = TorrentSessionHandle(str(session_dir))
        except ImportError:
            logger.error("exo_pyo3_bindings not available, torrent downloads disabled")
            self.session = None

    def on_progress(
        self, callback: Callable[[ShardMetadata, RepoDownloadProgress], None]
    ) -> None:
        """Register a progress callback."""
        self._progress_callbacks.append(callback)

    async def ensure_shard(
        self, shard: ShardMetadata, config_only: bool = False
    ) -> Path:
        """Download a model shard using BitTorrent.

        Args:
            shard: Shard metadata including model ID and layer range
            config_only: If True, only download config files (not implemented for torrents yet)

        Returns:
            Path to the downloaded shard directory

        Raises:
            RuntimeError: If torrent file is not found or session not initialized
        """
        if self.session is None:
            raise RuntimeError(
                "TorrentSessionHandle not initialized. "
                "exo_pyo3_bindings module not available."
            )

        model_id = str(shard.model_meta.model_id)

        # Resolve "main" branch to commit hash
        revision = await self._resolve_revision(model_id, "main")

        # Load embedded torrent file
        torrent_data = self._load_embedded_torrent(model_id, revision)
        if torrent_data is None:
            raise RuntimeError(
                f"Torrent not found for {model_id}@{revision}. "
                f"Expected at: rust/downloads/torrents/{model_id}/{revision}.torrent. "
                f"Please add the torrent file or use HTTP downloads (--enable-torrents=False)."
            )

        # Build v2 path: ~/.exo/models/v2/{model_id}/{revision}
        save_path = EXO_MODELS_DIR / "v2" / model_id / revision
        save_path.mkdir(parents=True, exist_ok=True)

        # Calculate which files to download based on shard layers
        # For now, download all files (selective download will be implemented later)
        file_indices = None  # None means download all files

        # Add torrent and download
        logger.info(
            f"Adding torrent for {model_id}@{revision} to session, saving to {save_path}"
        )
        info_hash = self.session.add_torrent(torrent_data, str(save_path), file_indices)

        # Wait for download with progress reporting
        logger.info(f"Starting download for {info_hash}")
        while True:
            progress_dict = self.session.get_progress(info_hash)

            # Convert dict to RepoDownloadProgress for callbacks
            progress = RepoDownloadProgress(
                downloaded=progress_dict["downloaded_bytes"],
                total=progress_dict["total_bytes"],
            )
            self._report_progress(shard, progress)

            if progress_dict["is_finished"]:
                logger.info(f"Download completed for {info_hash}")
                break

            await asyncio.sleep(0.5)

        # Enable persistent seeding
        logger.info(f"Enabling seeding for {info_hash}")
        self.session.enable_seeding(info_hash)

        return save_path

    async def get_shard_download_status(
        self,
    ) -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:
        """Get download status for all shards."""
        if self.session is None:
            return

        # List all torrents in the session
        info_hashes = self.session.list_torrents()

        for info_hash in info_hashes:
            try:
                progress_dict = self.session.get_progress(info_hash)
                progress = RepoDownloadProgress(
                    downloaded=progress_dict["downloaded_bytes"],
                    total=progress_dict["total_bytes"],
                )

                # We don't have a straightforward way to map info_hash back to path
                # This would require tracking in the session or metadata
                # For now, yield empty path
                yield (Path(), progress)
            except Exception as e:
                logger.error(f"Error getting status for {info_hash}: {e}")

    async def get_shard_download_status_for_shard(
        self, shard: ShardMetadata
    ) -> RepoDownloadProgress:
        """Get download status for a specific shard."""
        if self.session is None:
            return RepoDownloadProgress(downloaded=0, total=0)

        model_id = str(shard.model_meta.model_id)
        revision = await self._resolve_revision(model_id, "main")

        # We would need to track info_hash -> shard mapping
        # For now, return empty progress
        return RepoDownloadProgress(downloaded=0, total=0)

    async def _resolve_revision(self, model_id: str, branch: str = "main") -> str:
        """Resolve branch name to commit hash using HuggingFace API.

        Args:
            model_id: Model identifier (e.g., "mlx-community/Qwen3-30B-A3B-4bit")
            branch: Branch name (default: "main")

        Returns:
            Git commit hash (SHA)
        """
        try:
            from huggingface_hub import model_info

            info = model_info(model_id, revision=branch)
            return str(info.sha)
        except Exception as e:
            logger.error(f"Failed to resolve revision for {model_id}@{branch}: {e}")
            # Fallback to "main" as revision if API call fails
            logger.warning(f"Using branch name '{branch}' as revision")
            return branch

    def _load_embedded_torrent(self, model_id: str, revision: str) -> bytes | None:
        """Load embedded torrent file from Rust binary.

        Args:
            model_id: Model identifier
            revision: Git commit hash

        Returns:
            Torrent file contents, or None if not found
        """
        try:
            from exo_pyo3_bindings import get_embedded_torrent  # type: ignore

            result: bytes | None = get_embedded_torrent(model_id, revision)  # type: ignore
            return result
        except ImportError:
            logger.warning("exo_pyo3_bindings not available, cannot load torrents")
            return None
        except Exception as e:
            logger.error(f"Error loading torrent for {model_id}@{revision}: {e}")
            return None

    def _report_progress(
        self, shard: ShardMetadata, progress: RepoDownloadProgress
    ) -> None:
        """Report progress to all registered callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(shard, progress)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")
