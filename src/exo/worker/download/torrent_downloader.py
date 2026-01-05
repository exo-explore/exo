"""Torrent-based shard downloader using BitTorrent protocol with private tracker.

This module implements downloading model shards via BitTorrent with:
- Private tracker integration (/_internal/announce endpoint)
- WebSeed (BEP 19) fallback to HTTPS
- Selective file download based on shard layer ranges
- Persistent seeding of downloaded content
"""

import asyncio
from collections.abc import AsyncIterator
from datetime import timedelta
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable

from loguru import logger

from exo.shared.constants import EXO_MODELS_DIR
from exo.shared.types.memory import Memory
from exo.shared.types.worker.shards import ShardMetadata
from exo.worker.download.download_utils import RepoDownloadProgress, resolve_allow_patterns
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

        Downloads all torrent variants for the model and applies file filtering
        based on the shard's layer range.

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

        # Load all embedded torrent variants
        variants = self._load_embedded_torrents(model_id, revision)
        if not variants:
            raise RuntimeError(
                f"No torrents found for {model_id}@{revision}. "
                f"Expected at: rust/downloads/torrents/{model_id}/{revision}.*.torrent. "
                f"Please add torrent files or use HTTP downloads (--enable-torrents=False)."
            )

        # Build v2 path: ~/.exo/models/v2/{model_id}/{revision}
        save_path = EXO_MODELS_DIR / "v2" / model_id / revision
        save_path.mkdir(parents=True, exist_ok=True)

        # Get allow patterns for file filtering (same as HTTP downloads)
        allow_patterns = await resolve_allow_patterns(shard)
        logger.debug(f"Torrent download allow patterns: {allow_patterns}")

        # Track aggregate progress across all variants
        total_downloaded = 0
        total_size = 0

        # Download each variant
        for variant_name, torrent_data in variants:
            # Get file list from torrent
            file_list = self._get_torrent_file_list(torrent_data)
            if not file_list:
                logger.warning(f"No files in variant '{variant_name}', skipping")
                continue

            # Filter files using allow patterns
            file_indices = self._filter_files(file_list, allow_patterns)
            if not file_indices:
                logger.debug(f"No matching files in variant '{variant_name}' after filtering")
                continue

            logger.info(
                f"Adding torrent variant '{variant_name}' for {model_id}@{revision} "
                f"({len(file_indices)} of {len(file_list)} files)"
            )

            try:
                info_hash = self.session.add_torrent(
                    torrent_data, str(save_path), file_indices
                )
            except Exception as e:
                # Torrent may already be added
                logger.debug(f"Could not add variant '{variant_name}': {e}")
                continue

            # Wait for download with progress reporting
            logger.info(f"Starting download for variant '{variant_name}' ({info_hash})")
            while True:
                progress_dict = self.session.get_progress(info_hash)

                # Update aggregate progress
                variant_downloaded = progress_dict["downloaded_bytes"]
                variant_total = progress_dict["total_bytes"]

                # Report combined progress
                progress = self._make_progress(
                    shard=shard,
                    model_id=model_id,
                    revision=revision,
                    downloaded=total_downloaded + variant_downloaded,
                    total=total_size + variant_total,
                    status="in_progress",
                )
                self._report_progress(shard, progress)

                if progress_dict["is_finished"]:
                    logger.info(f"Variant '{variant_name}' download completed")
                    total_downloaded += variant_total
                    total_size += variant_total
                    break

                await asyncio.sleep(0.5)

            # Enable persistent seeding for this variant
            try:
                self.session.enable_seeding(info_hash)
            except Exception as e:
                logger.warning(f"Could not enable seeding for '{variant_name}': {e}")

        return save_path

    async def get_shard_download_status(
        self,
    ) -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:
        """Get download status for all shards."""
        if self.session is None:
            return

        # List all torrents in the session
        info_hashes = self.session.list_torrents()

        # We don't have a straightforward way to map info_hash back to shard/path
        # This would require tracking in the session or metadata
        # For now, this method doesn't yield anything meaningful for torrents
        for info_hash in info_hashes:
            try:
                _progress_dict = self.session.get_progress(info_hash)
                # Can't create proper progress without shard context
                # Skip yielding until we have proper tracking
                logger.debug(f"Torrent {info_hash} status check - not implemented")
            except Exception as e:
                logger.error(f"Error getting status for {info_hash}: {e}")
        # Yield nothing - torrents don't support this method well yet
        return
        yield  # Make this an async generator

    async def get_shard_download_status_for_shard(
        self, shard: ShardMetadata
    ) -> RepoDownloadProgress:
        """Get download status for a specific shard."""
        model_id = str(shard.model_meta.model_id)

        if self.session is None:
            return self._make_progress(
                shard=shard,
                model_id=model_id,
                revision="unknown",
                downloaded=0,
                total=0,
                status="not_started",
            )

        # We would need to track info_hash -> shard mapping
        # For now, return empty progress
        return self._make_progress(
            shard=shard,
            model_id=model_id,
            revision="unknown",
            downloaded=0,
            total=0,
            status="not_started",
        )

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

    def _load_embedded_torrents(
        self, model_id: str, revision: str
    ) -> list[tuple[str, bytes]]:
        """Load all embedded torrent variants from Rust binary.

        Args:
            model_id: Model identifier
            revision: Git commit hash

        Returns:
            List of (variant_name, torrent_bytes) tuples, e.g., [("small", ...), ("large", ...)]
        """
        try:
            from exo_pyo3_bindings import get_embedded_torrents  # type: ignore

            result: list[tuple[str, bytes]] = get_embedded_torrents(model_id, revision)
            return result
        except ImportError:
            logger.warning("exo_pyo3_bindings not available, cannot load torrents")
            return []
        except Exception as e:
            logger.error(f"Error loading torrents for {model_id}@{revision}: {e}")
            return []

    def _get_torrent_file_list(
        self, torrent_data: bytes
    ) -> list[tuple[int, str, int]]:
        """Get file list from torrent data.

        Args:
            torrent_data: Raw torrent file bytes

        Returns:
            List of (index, path, size_bytes) tuples
        """
        try:
            from exo_pyo3_bindings import get_torrent_file_list  # type: ignore

            return get_torrent_file_list(torrent_data)
        except ImportError:
            logger.warning("exo_pyo3_bindings not available")
            return []
        except Exception as e:
            logger.error(f"Error parsing torrent file list: {e}")
            return []

    def _filter_files(
        self, file_list: list[tuple[int, str, int]], allow_patterns: list[str]
    ) -> list[int]:
        """Filter file list using glob patterns.

        Args:
            file_list: List of (index, path, size) tuples
            allow_patterns: Glob patterns to match (e.g., ["*.json", "*.safetensors"])

        Returns:
            List of file indices that match at least one pattern
        """
        matching_indices = []
        for index, path, _size in file_list:
            for pattern in allow_patterns:
                if fnmatch(path, pattern):
                    matching_indices.append(index)
                    break
        return matching_indices

    def _make_progress(
        self,
        shard: ShardMetadata,
        model_id: str,
        revision: str,
        downloaded: int,
        total: int,
        status: str,
    ) -> RepoDownloadProgress:
        """Create a RepoDownloadProgress object with proper fields.

        Args:
            shard: Shard metadata
            model_id: Model identifier
            revision: Git revision
            downloaded: Downloaded bytes
            total: Total bytes
            status: Download status ("not_started", "in_progress", "complete")

        Returns:
            RepoDownloadProgress object
        """
        return RepoDownloadProgress(
            repo_id=model_id,
            repo_revision=revision,
            shard=shard,
            completed_files=1 if status == "complete" else 0,
            total_files=1,
            downloaded_bytes=Memory.from_bytes(downloaded),
            downloaded_bytes_this_session=Memory.from_bytes(downloaded),
            total_bytes=Memory.from_bytes(total),
            overall_speed=0.0,  # Not tracked for torrents yet
            overall_eta=timedelta(0),
            status=status,  # type: ignore
        )

    def _report_progress(
        self, shard: ShardMetadata, progress: RepoDownloadProgress
    ) -> None:
        """Report progress to all registered callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(shard, progress)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")
