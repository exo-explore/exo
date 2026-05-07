"""HTTP client for downloading model files from peer nodes.

Instead of downloading from HuggingFace, nodes can fetch model files from
peers on the same LAN that already have them (or are still downloading them).
Falls back gracefully if the peer is unreachable or the transfer fails.
"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as session:
            async with session.get(url) as r:
                if r.status != 200:
                    return None
                data = await r.json()
                return [PeerFileInfo(**f) for f in data.get("files", [])]
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

    # Resume from existing partial
    if await aios.path.exists(partial_path):
        n_read = (await aios.stat(partial_path)).st_size

    poll_count = 0
    chunk_size = 8 * 1024 * 1024  # 8MB, matching HF download

    try:
        while n_read < expected_size and poll_count < max_poll_attempts:
            headers: dict[str, str] = {}
            if n_read > 0:
                headers["Range"] = f"bytes={n_read}-"

            got_bytes = False
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300, sock_read=60)
            ) as session:
                async with session.get(url, headers=headers) as r:
                    if r.status == 416:
                        # Range not satisfiable - peer doesn't have more yet
                        pass
                    elif r.status in (200, 206):
                        peer_complete = r.headers.get("X-Exo-Complete") == "true"
                        safe_bytes = int(r.headers.get("X-Exo-Safe-Bytes", "0"))

                        async with aiofiles.open(
                            partial_path, "ab" if n_read > 0 else "wb"
                        ) as f:
                            while True:
                                chunk = await r.content.read(chunk_size)
                                if not chunk:
                                    break
                                written = await f.write(chunk)
                                n_read += written
                                got_bytes = True
                                on_progress(n_read, expected_size, False)
                    elif r.status == 404:
                        logger.debug(
                            f"File {file_path} not found on peer {peer_host}"
                        )
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
