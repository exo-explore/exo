"""Peer-aware shard downloader that tries LAN peers before HuggingFace.

Wraps an existing ShardDownloader and adds a peer-download step: before
hitting HuggingFace, try peers provided in the available_peers list.
Falls back to the inner downloader (HF) if peer download fails.

The peer list is computed by the Worker at command-emit time and passed
through the StartDownload command, keeping the download coordinator
decoupled from Worker state.
"""

import asyncio
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Coroutine
from datetime import timedelta
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Literal

import aiofiles
import aiofiles.os as aios
from loguru import logger

from exo.download.download_utils import (
    RepoDownloadProgress,
    calc_hash,
    calculate_repo_progress,
    fetch_file_list_with_cache,
    file_meta,
    is_image_model,
    resolve_allow_patterns,
    resolve_model_dir,
)
from exo.download.huggingface_utils import filter_repo_objects
from exo.download.peer_download import (
    download_file_from_peer,
    get_peer_file_status,
)
from exo.download.shard_downloader import ShardDownloader
from exo.shared.types.commands import PeerEndpoint
from exo.shared.types.memory import Memory
from exo.shared.types.worker.downloads import FileListEntry, RepoFileDownloadProgress
from exo.shared.types.worker.shards import ShardMetadata

ShardPeerKey = str


async def _run_progress_callback(
    callback: Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]],
    shard: ShardMetadata,
    progress: RepoDownloadProgress,
) -> None:
    await callback(shard, progress)


class PeerAwareShardDownloader(ShardDownloader):
    """ShardDownloader that tries peer download before HuggingFace.

    Decorates an inner ShardDownloader (typically ResumableShardDownloader).
    On ensure_shard(), if available_peers were provided, tries downloading
    from them over the LAN first. Falls back to the inner downloader if
    no peer has it or the transfer fails.
    """

    def __init__(self, inner: ShardDownloader, offline: bool = False) -> None:
        self._inner = inner
        # ``offline`` mirrors ``ResumableShardDownloader.offline`` and is
        # forwarded to ``fetch_file_list_with_cache`` so that a node
        # configured for offline operation never reaches out to
        # HuggingFace before attempting a peer download. Pre-fix the
        # peer path hard-coded ``skip_internet=False`` and would raise
        # on cold/offline nodes that lacked a cached file list, ending
        # the peer attempt before it could even start. Codex flagged
        # this as a P1 (PR #16 round 2).
        self._offline = offline
        self._progress_callbacks: list[
            Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]]
        ] = []
        # Peers are set per-download by the coordinator before calling ensure_shard.
        self._peers_by_shard: defaultdict[ShardPeerKey, deque[list[PeerEndpoint]]] = (
            defaultdict(deque)
        )

    def set_available_peers(
        self, shard: ShardMetadata, peers: list[PeerEndpoint]
    ) -> None:
        """Set the peers to try for a specific ensure_shard call.

        Called by DownloadCoordinator before triggering a download, based
        on the peers embedded in the StartDownload command.
        """
        self._peers_by_shard[_peer_key(shard)].append(list(peers))

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
        peers = self._pop_available_peers(shard)

        if not peers:
            logger.debug(
                f"No peers available for {model_id}, downloading from HuggingFace"
            )
            return await self._inner.ensure_shard(shard, config_only=False)

        # Try each peer (already sorted by priority: RDMA first, completed first)
        for peer in peers:
            logger.info(
                f"Attempting peer download of {model_id} from "
                f"{peer.ip}:{peer.port} (status: {peer.status}, link: {peer.connection_type})"
            )
            result = await self._try_peer_download(
                shard, peer.ip, peer.port, normalized
            )
            if result is not None:
                logger.info(f"Successfully downloaded {model_id} from peer {peer.ip}")
                return result
            logger.info(
                f"Peer download from {peer.ip} failed, trying next peer or HuggingFace"
            )

        # All peers failed, fall back to HuggingFace
        logger.info(
            f"All peer downloads failed for {model_id}, falling back to HuggingFace"
        )
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
        peer_files = await get_peer_file_status(peer_ip, peer_port, model_id_normalized)
        if not peer_files:
            return None

        peer_file_map = {f.path: f for f in peer_files}

        # Get the file list we need (same logic as download_shard)
        revision = "main"
        target_dir = await resolve_model_dir(shard.model_card.model_id)

        try:
            file_list = await fetch_file_list_with_cache(
                shard.model_card.model_id,
                revision,
                recursive=True,
                # Honor the coordinator's offline setting so a cold
                # offline node can still satisfy a peer download from
                # the LAN without reaching out to HuggingFace for the
                # initial file-list fetch (Codex P1, PR #16 round 2).
                skip_internet=self._offline,
            )
        except Exception:
            return None

        allow_patterns = await resolve_allow_patterns(shard)
        # Mirror ``download_shard``'s selection logic exactly: it filters
        # by ``allow_patterns`` AND ``ignore_patterns`` before deciding
        # which files to fetch. Pre-fix the peer path applied
        # ``allow_patterns`` only and missed the ignore set, so for any
        # repo containing ``original/*`` or ``metal/*`` (e.g. Llama 3.x
        # repos) the peer would not have those files locally, and the
        # later strict ``peer_info`` missing => fail check would abort
        # the whole peer transfer and force a HuggingFace fallback for
        # every download (Codex P1, PR #16 round 2). Keep this list in
        # sync with ``download_shard`` (download_utils.py:983).
        ignore_patterns = ["original/*", "metal/*"]
        filtered_file_list: list[FileListEntry] = list(
            filter_repo_objects(
                file_list,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                key=lambda x: x.path,
            )
        )

        if is_image_model(shard):
            filtered_file_list = [
                f
                for f in filtered_file_list
                if "/" in f.path or not f.path.endswith(".safetensors")
            ]

        # Check the peer has all (or most) files we need
        files_on_peer = sum(1 for f in filtered_file_list if f.path in peer_file_map)
        if files_on_peer == 0:
            logger.debug(f"Peer has no files we need for {model_id_normalized}")
            return None

        # Download from peer with progress tracking
        all_start_time = time.time()
        file_progress: dict[str, RepoFileDownloadProgress] = {}
        semaphore = asyncio.Semaphore(8)
        failed = False

        async def download_one(file_path: str, expected_size: int) -> bool:
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
                progress = calculate_repo_progress(
                    shard,
                    shard.model_card.model_id,
                    revision,
                    file_progress,
                    all_start_time,
                )
                for cb in self._progress_callbacks:
                    asyncio.create_task(_run_progress_callback(cb, shard, progress))

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
                if result is None:
                    return False
                # Offline / air-gapped deployments have explicitly opted
                # out of contacting HuggingFace. Codex flagged (P1, PR
                # #16 round 3) that calling ``file_meta`` here silently
                # broke peer transfers in offline mode: any exception
                # (e.g. DNS failure, blocked egress) was treated as
                # integrity-check failure and the peer copy was
                # deleted, leaving the cold node with no path to
                # complete model sync. When the operator runs with
                # ``--offline``/``EXO_OFFLINE=true`` we trust the LAN
                # peer's bytes (size already enforced by
                # ``download_file_from_peer``) and skip the HF
                # canonical-hash check entirely.
                if self._offline:
                    return True

                # Codex flagged (P2, PR #16 round 2) that peer downloads
                # were marked successful as soon as ``n_read ==
                # expected_size``, with no content-integrity check. A
                # peer serving wrong bytes with the right length
                # (stale/corrupt/malicious) would otherwise be
                # silently accepted as model data, causing
                # hard-to-diagnose inference failures.
                #
                # Validate against HuggingFace's authoritative hash:
                # we already need internet for ``fetch_file_list_with_cache``
                # in online mode, so the extra ``file_meta()`` HEAD is
                # cheap. Trusting a hash advertised by the peer would
                # leave us vulnerable to a malicious peer that lies
                # about both bytes and hash; HF is the canonical
                # source.
                #
                # On mismatch the partial-or-renamed file is removed
                # so the caller's HF fallback (``self._inner.ensure_shard``)
                # starts from a clean slate.
                try:
                    _expected_size, expected_etag = await file_meta(
                        shard.model_card.model_id, revision, file_path
                    )
                except Exception as exc:
                    # If we can't reach HF for metadata, the file
                    # might still be valid -- but we can't prove it.
                    # Fall back to HF download where the same call
                    # would have happened anyway.
                    logger.warning(
                        f"Peer download integrity-check failed: could not "
                        f"fetch HF metadata for {file_path}: {exc}; "
                        f"discarding peer-downloaded copy"
                    )
                    try:
                        await aios.remove(result)
                    except Exception as cleanup_exc:
                        logger.debug(
                            f"Could not remove unverified peer download "
                            f"{result}: {cleanup_exc}"
                        )
                    return False

                hash_type: Literal["sha1", "sha256"] = (
                    "sha256" if len(expected_etag) == 64 else "sha1"
                )
                final_hash = await calc_hash(result, hash_type=hash_type)
                if final_hash != expected_etag:
                    logger.warning(
                        f"Peer-downloaded {file_path} from {peer_ip} has "
                        f"hash {final_hash} but HF authoritative hash is "
                        f"{expected_etag} ({hash_type}); discarding and "
                        f"falling back to HF"
                    )
                    try:
                        await aios.remove(result)
                    except Exception as exc:
                        logger.error(
                            f"Failed to remove corrupt peer download {result}: {exc}"
                        )
                    return False
                return True

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

        # Codex P2 (PR #16 round-(N+10), peer_shard_downloader.py:354):
        # zero-byte files (e.g. ``.gitattributes`` markers, empty
        # ``__init__.py`` shims) MUST still be materialized so the
        # local snapshot mirrors the filtered file list HF would
        # have produced. Pre-fix the peer path silently skipped any
        # file with ``size in (None, 0)`` and reported success, so
        # ``DownloadCompleted`` was published with an incomplete
        # local model directory -- subsequent loads that touched
        # those marker files (model loaders, processors that probe
        # for ``chat_template.json``, etc.) would then fail in ways
        # that don't point back at the peer step.
        #
        # Codex P1 (PR #16 round-(N+11), peer_shard_downloader.py:354):
        # ``size is None`` is *not* the same as ``size == 0``.
        # ``fetch_file_list_with_cache`` returns ``FileListEntry(size=None)``
        # for files discovered via the safetensors index (e.g. weight
        # shards whose size is not in the HF API response). Pre-fix
        # the previous round lumped ``None`` together with literal
        # zero and materialized those weight files as empty,
        # producing a "DownloadCompleted" snapshot with corrupted /
        # incomplete weights that failed only at load/inference
        # time. Split the cases: ``== 0`` is materialized as an
        # empty marker; ``is None`` aborts the peer transfer and
        # forces the HF fallback so the file gets a real download
        # path.
        #
        # Pre-pass: detect bail-out conditions before constructing any
        # ``download_one`` coroutines so we don't leak un-awaited
        # coroutines on the unknown-size or missing-peer-info paths.
        for f in filtered_file_list:
            if f.size is None:
                logger.info(
                    f"Peer transfer for {model_id_normalized} aborted: "
                    f"unknown-size entry {f.path!r} (size=None) cannot "
                    f"be safely transferred over peer; falling back to HF"
                )
                return None
            if f.size == 0:
                continue
            peer_info = peer_file_map.get(f.path)
            if not peer_info or peer_info.safe_bytes <= 0:
                # Real-size file the peer doesn't have => abort transfer.
                return None

        zero_byte_files: list[str] = []
        tasks: list[Coroutine[Any, Any, bool]] = []
        for f in filtered_file_list:
            if f.size is None:
                # Pre-pass already bailed; safety net for type-narrowing.
                return None
            if f.size == 0:
                # Defer the local touch until after we know the rest
                # of the peer transfer succeeded; a partial peer
                # transfer should not leave behind orphan empty
                # marker files that masquerade as a complete download.
                zero_byte_files.append(f.path)
                continue
            peer_info = peer_file_map.get(f.path)
            if peer_info and peer_info.safe_bytes > 0:
                tasks.append(download_one(f.path, f.size))
            else:
                failed = True
                break

        if failed:
            return None

        results = await asyncio.gather(*tasks, return_exceptions=True)
        if any(isinstance(r, Exception) or r is False for r in results):
            return None

        for marker_path in zero_byte_files:
            full_path = target_dir / marker_path
            try:
                await aios.makedirs(full_path.parent, exist_ok=True)
                # ``aios.path.exists`` first to avoid an unnecessary
                # touch (and the corresponding mtime bump) when
                # resume-from-partial finds the marker already on
                # disk. ``aios.open`` in append mode is the safest
                # way to materialize the empty file without
                # truncating an already-present marker.
                if not await aios.path.exists(full_path):
                    async with aiofiles.open(full_path, mode="a"):
                        pass
            except Exception as exc:
                logger.warning(
                    f"Could not materialize zero-byte marker file "
                    f"{full_path} after peer transfer: {exc}; "
                    f"falling back to HF for full snapshot integrity"
                )
                return None
            # Codex P2 (PR #16 round-(N+13), peer_shard_downloader.py:407):
            # ``download_one`` -> ``on_file_progress`` is the only
            # writer of the per-file ``status="complete"`` marker;
            # the zero-byte branch never invokes it (there are no
            # bytes to stream), so the file_progress entry seeded
            # at line 338 stays at ``status="not_started"``. The
            # final ``calculate_repo_progress`` call below then
            # rolls those up into a non-``complete`` overall status,
            # which means ``_download_progress_callback`` does NOT
            # publish ``DownloadCompleted`` -- the model gets stuck
            # in ``DownloadOngoing`` until the periodic
            # reconciliation loop in ``DownloadCoordinator`` notices
            # the on-disk snapshot and force-updates the status.
            # Mirror the regular file completion path by overwriting
            # the seeded entry with a fully-complete one once the
            # marker is on disk. ``RepoFileDownloadProgress`` is
            # frozen, so we replace the dict slot rather than
            # mutating the existing instance.
            file_progress[marker_path] = RepoFileDownloadProgress(
                repo_id=str(shard.model_card.model_id),
                repo_revision=revision,
                file_path=marker_path,
                downloaded=Memory.from_bytes(0),
                downloaded_this_session=Memory.from_bytes(0),
                total=Memory.from_bytes(0),
                speed=0,
                eta=timedelta(0),
                status="complete",
                start_time=all_start_time,
            )

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

        gguf = next((f for f in filtered_file_list if f.path.endswith(".gguf")), None)
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

    def _pop_available_peers(self, shard: ShardMetadata) -> list[PeerEndpoint]:
        key = _peer_key(shard)
        queue = self._peers_by_shard.get(key)
        if not queue:
            return []
        peers = queue.popleft()
        if not queue:
            del self._peers_by_shard[key]
        return peers


def _peer_key(shard: ShardMetadata) -> ShardPeerKey:
    return shard.model_dump_json()
