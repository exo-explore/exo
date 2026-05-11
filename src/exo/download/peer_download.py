"""HTTP client for downloading model files from peer nodes.

Instead of downloading from HuggingFace, nodes can fetch model files from
peers on the same LAN that already have them (or are still downloading them).
Falls back gracefully if the peer is unreachable or the transfer fails.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, cast

import aiofiles
import aiofiles.os as aios
import aiohttp
from loguru import logger


@dataclass(frozen=True)
class PeerFileInfo:
    """Status of a single file on a peer node."""

    path: str
    size: int
    complete: bool
    safe_bytes: int


def _as_int(value: object) -> int:
    return value if isinstance(value, int) else 0


async def get_peer_file_status(
    peer_host: str,
    peer_port: int,
    model_id_normalized: str,
    timeout: float = 5.0,
) -> list[PeerFileInfo] | None:
    """Query a peer's file server for available files for a model.

    Returns None if the peer is unreachable.
    """
    url = f"http://{peer_host}:{peer_port}/status/{model_id_normalized}"
    try:
        async with (
            aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as session,
            session.get(url) as r,
        ):
            if r.status != 200:
                return None
            data = cast(dict[str, object], await r.json())
            files = data.get("files", [])
            if not isinstance(files, list):
                return []
            raw_files = cast(list[object], files)
            out: list[PeerFileInfo] = []
            required = {"path", "size", "complete", "safe_bytes"}
            for raw_file in raw_files:
                if not isinstance(raw_file, dict):
                    continue
                file_info = cast(dict[str, object], raw_file)
                if not required.issubset(file_info):
                    continue
                out.append(
                    PeerFileInfo(
                        path=str(file_info["path"]),
                        size=_as_int(file_info["size"]),
                        complete=bool(file_info["complete"]),
                        safe_bytes=_as_int(file_info["safe_bytes"]),
                    )
                )
            return out
    except Exception as e:
        logger.debug(f"Could not reach peer {peer_host}:{peer_port}: {e}")
        return None


async def download_file_from_peer(
    peer_host: str,
    peer_port: int,
    model_id_normalized: str,
    file_path: str,
    target_dir: Path,
    expected_size: int,
    on_progress: Callable[[int, int, bool], None] = lambda _a, _b, _c: None,
    max_poll_attempts: int = 60,
    poll_interval: float = 3.0,
) -> Path | None:
    """Download a single file from a peer's file server.

    Supports streaming relay: if the peer is still downloading the file,
    we fetch available bytes, wait, and poll for more until the file is
    complete.

    Returns the final file path on success, or None on failure (caller
    should fall back to HuggingFace).
    """
    target_path = target_dir / file_path
    partial_path = target_dir / f"{file_path}.partial"

    # Check if already complete locally
    if await aios.path.exists(target_path):
        local_size = (await aios.stat(target_path)).st_size
        if local_size == expected_size:
            on_progress(expected_size, expected_size, True)
            return target_path

    await aios.makedirs((target_dir / file_path).parent, exist_ok=True)

    url = f"http://{peer_host}:{peer_port}/files/{model_id_normalized}/{file_path}"
    n_read = 0

    # Resume from existing partial.
    #
    # Codex P1 (PR #16 round 5): a stale ``.partial`` left over from a
    # previous run can be larger than ``expected_size`` (e.g. the peer
    # was serving the wrong revision, the on-disk file was truncated
    # to a different blob, or the user manually replaced it). In that
    # case ``n_read >= expected_size`` skips the resume loop entirely
    # and we'd then ``rename`` a too-large file as the "successful"
    # result. With offline mode we explicitly skip hash verification,
    # so the bad bytes would never get caught downstream and would
    # poison the model cache. Fail fast: drop the stale partial and
    # restart from zero on this peer.
    if await aios.path.exists(partial_path):
        existing_size = (await aios.stat(partial_path)).st_size
        if existing_size > expected_size:
            logger.warning(
                f"Discarding stale oversized peer partial for {file_path} "
                f"({existing_size} > expected {expected_size}); "
                "restarting download from zero"
            )
            await aios.remove(partial_path)
            n_read = 0
        else:
            n_read = existing_size

    poll_count = 0
    chunk_size = 8 * 1024 * 1024  # 8MB, matching HF download

    try:
        while n_read < expected_size and poll_count < max_poll_attempts:
            headers: dict[str, str] = {}
            if n_read > 0:
                headers["Range"] = f"bytes={n_read}-"

            got_bytes = False
            range_was_requested = n_read > 0
            async with (
                aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=300, sock_read=60)
                ) as session,
                session.get(url, headers=headers) as r,
            ):
                if r.status == 416:
                    # Range not satisfiable - peer doesn't have more yet
                    pass
                elif range_was_requested and r.status == 200:
                    # Codex P1 (PR #16 round-(N+3), peer_download.py:162):
                    # we sent a ``Range`` header (we have a partial), but
                    # the peer ignored it and returned full content with
                    # 200. Appending the body would duplicate the
                    # already-downloaded prefix, push ``n_read`` past
                    # ``expected_size``, and -- because offline mode
                    # skips hash verification -- silently poison the
                    # model file. Drop the partial and restart from
                    # zero on the next loop iteration so the next
                    # request gets fresh, intact bytes.
                    logger.warning(
                        f"Peer {peer_host} ignored Range header for "
                        f"{file_path} (returned 200 instead of 206); "
                        "discarding partial and restarting from zero"
                    )
                    await aios.remove(partial_path)
                    n_read = 0
                elif r.status in (200, 206):
                    # Codex P1 (PR #16 round-(N+8), peer_download.py:187):
                    # bound the inner read by ``expected_size - n_read``
                    # and treat any extra bytes as a peer protocol
                    # violation. Pre-fix the loop kept appending until
                    # EOF and only checked ``n_read < expected_size``
                    # afterward, so an oversized response (peer
                    # serving a stale/wrong blob) was accepted as
                    # success and renamed into the model cache. In
                    # offline mode hash verification is skipped, so
                    # this silently poisoned local weights. Now we
                    # cap each chunk at the remaining budget and bail
                    # out the moment a peer tries to send extra data.
                    oversized_response = False
                    async with aiofiles.open(
                        partial_path, "ab" if n_read > 0 else "wb"
                    ) as f:
                        while True:
                            remaining = expected_size - n_read
                            if remaining <= 0:
                                # We have everything we need. Read one
                                # more byte to detect peer
                                # over-supplying; if the stream isn't
                                # EOF, the peer is sending more bytes
                                # than ``expected_size`` claims.
                                tail = await r.content.read(1)
                                if tail:
                                    oversized_response = True
                                break
                            chunk = await r.content.read(min(chunk_size, remaining))
                            if not chunk:
                                break
                            written = await f.write(chunk)
                            n_read += written
                            got_bytes = True
                            on_progress(n_read, expected_size, False)
                    if oversized_response:
                        # Discard the partial: we cannot trust any
                        # bytes from a peer that violates the
                        # advertised file size, especially in
                        # offline mode where hash verification is
                        # skipped. Restart from zero on the next
                        # iteration so a fresh request gets a
                        # well-bounded response.
                        logger.warning(
                            f"Peer {peer_host} returned oversized response for "
                            f"{file_path} (advertised {expected_size} bytes, "
                            "stream still had data when budget was exhausted); "
                            "discarding partial and restarting from zero"
                        )
                        await aios.remove(partial_path)
                        n_read = 0
                elif r.status == 404:
                    logger.debug(f"File {file_path} not found on peer {peer_host}")
                    return None
                else:
                    logger.warning(
                        f"Unexpected status {r.status} from peer {peer_host}"
                    )
                    return None

            # Check if we're done
            if n_read >= expected_size:
                break

            # If we got no new bytes, the peer might still be downloading
            if not got_bytes:
                poll_count += 1
                logger.debug(
                    f"Waiting for peer {peer_host} to download more of {file_path} "
                    f"({n_read}/{expected_size}, poll {poll_count}/{max_poll_attempts})"
                )
                await asyncio.sleep(poll_interval)
            else:
                # Got data, reset poll counter
                poll_count = 0

        if n_read < expected_size:
            logger.warning(
                f"Peer download incomplete for {file_path}: {n_read}/{expected_size}"
            )
            return None

        # Rename partial to final
        await aios.rename(partial_path, target_path)
        on_progress(expected_size, expected_size, True)
        logger.info(
            f"Downloaded {file_path} from peer {peer_host} ({expected_size} bytes)"
        )
        return target_path

    except Exception as e:
        logger.warning(f"Peer download failed for {file_path} from {peer_host}: {e}")
        return None
