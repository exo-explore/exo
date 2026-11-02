"""Model-file helpers for the rest of exo: finding a model on disk, fetching a
single file (model cards), and an HTTP session for other callers."""

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import aiohttp
from loguru import logger

from exo.download import store
from exo.download.hub import Hub, NotFoundError, TransientError, chunks, new_session
from exo.shared.models.model_cards import ModelCard
from exo.shared.types.common import ModelId
from exo.shared.types.worker.downloads import FileListEntry


def resolve_existing_model(
    model_id: ModelId, card: ModelCard | None = None
) -> Path | None:
    """The dir of a complete copy of the model (read-only dirs first), if any."""
    return store.find_complete(model_id, card)


def is_read_only_model_dir(model_dir: Path) -> bool:
    return store.is_read_only(model_dir)


def build_model_path(model_id: ModelId) -> Path:
    found = store.find_complete(model_id)
    return found if found is not None else store.default_dir() / model_id.normalize()


async def resolve_model_dir(model_id: ModelId) -> Path:
    """The model's dir (a complete copy, else the default one), created if needed."""
    target = await asyncio.to_thread(build_model_path, model_id)
    await asyncio.to_thread(target.mkdir, parents=True, exist_ok=True)
    return target


def create_http_session(
    auto_decompress: bool = False,
    timeout_profile: Literal["short", "long"] = "long",
) -> aiohttp.ClientSession:
    if timeout_profile == "short":
        return new_session(
            connect_timeout=10,
            idle_timeout=30,
            total_timeout=30,
            auto_decompress=auto_decompress,
        )
    return new_session(
        connect_timeout=60, idle_timeout=60, auto_decompress=auto_decompress
    )


async def download_file_with_retry(
    model_id: ModelId,
    revision: str,
    path: str,
    target_dir: Path,
    on_progress: Callable[[int, int, bool], None] = lambda _, __, ___: None,
    attempts: int = 3,
) -> Path:
    """Fetch one (small) file into ``target_dir`` unless it is already there.

    Raises ``FileNotFoundError`` when the Hub doesn't have it.
    """
    target = target_dir / path
    if await asyncio.to_thread(target.exists):
        return target
    partial = target_dir / (path + store.PARTIAL_SUFFIX)
    for attempt in range(1, attempts + 1):
        try:
            async with Hub.connect(
                connect_timeout=30, idle_timeout=60, metadata_timeout=60
            ) as hub:
                received = 0
                await asyncio.to_thread(
                    partial.parent.mkdir, parents=True, exist_ok=True
                )
                async with hub.open(model_id, revision, path) as response:
                    total = response.content_length or 0
                    with open(partial, "wb") as file:  # noqa: ASYNC230 - small files
                        async for chunk in chunks(response, 1 << 20):
                            file.write(chunk)
                            received += len(chunk)
                            on_progress(received, total, False)
            await asyncio.to_thread(partial.replace, target)
            on_progress(received, received, True)
            return target
        except NotFoundError as error:
            raise FileNotFoundError(error.message) from error
        except TransientError as error:
            if attempt == attempts:
                raise
            logger.info(f"Fetching {model_id}/{path}: {error.message}; retrying")
            await asyncio.sleep(attempt)
    raise AssertionError("unreachable")


async def fetch_file_list_with_cache(
    model_id: ModelId, revision: str = "main", recursive: bool = True
) -> list[FileListEntry]:
    """Every file of the repo at ``revision``."""
    del recursive  # the listing is always recursive
    async with Hub.connect(
        connect_timeout=30, idle_timeout=60, metadata_timeout=60
    ) as hub:
        files = await hub.list_files(model_id, revision)
    return [FileListEntry(type="file", path=f.path, size=f.size) for f in files]
