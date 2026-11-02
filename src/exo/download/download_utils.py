"""Model-file helpers for the rest of exo: finding a model on disk, fetching a
single file (model cards), and an HTTP session for other callers."""

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import aiohttp
from anyio import to_thread

from exo.download import store, transfer
from exo.download.hub import Hub, NotFoundError, new_session, proxy_for
from exo.download.store import Manifest
from exo.download.transfer import DownloadSettings, Locks, Part, Progress
from exo.shared.models.model_cards import ModelCard
from exo.shared.types.common import ModelId


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
    target = await to_thread.run_sync(build_model_path, model_id)
    await to_thread.run_sync(lambda: target.mkdir(parents=True, exist_ok=True))
    return target


def create_http_session(
    auto_decompress: bool = False,
    timeout_profile: Literal["short", "long"] = "long",
) -> aiohttp.ClientSession:
    # one proxy for the session's requests, as before (any host)
    proxy = proxy_for("https://")
    if timeout_profile == "short":
        return new_session(
            connect_timeout=10,
            idle_timeout=30,
            total_timeout=30,
            auto_decompress=auto_decompress,
            proxy=proxy,
        )
    return new_session(
        connect_timeout=60,
        idle_timeout=60,
        auto_decompress=auto_decompress,
        proxy=proxy,
    )


_TRIES = 3


async def download_file_with_retry(
    model_id: ModelId,
    revision: str,
    path: str,
    target_dir: Path,
    on_progress: Callable[[int, int, bool], None] = lambda _, __, ___: None,
) -> Path:
    """Fetch one file of the repo into ``target_dir`` unless it is already there,
    the same way a model download does (resumed, hash-checked, renamed into place).

    Uses the commit a download of this model pinned, if any, else ``revision``.
    Raises ``FileNotFoundError`` when the Hub doesn't have it.
    """
    target = target_dir / path
    if await to_thread.run_sync(target.exists):
        return target
    settings = DownloadSettings()
    progress = Progress()
    async with Hub.connect(
        connect_timeout=settings.connect_timeout,
        idle_timeout=settings.idle_timeout,
        metadata_timeout=settings.metadata_timeout,
    ) as hub:

        async def locate() -> Manifest:
            pinned = await to_thread.run_sync(store.read_manifest, target_dir)
            if pinned is not None:
                return pinned
            commit = await hub.resolve_commit(model_id, revision)
            return Manifest(
                repo=str(model_id),
                commit=commit,
                files=await hub.list_files(model_id, commit),
            )

        try:
            # it answers an API call: a few tries, as before, not a long wait
            manifest = await transfer.retrying(
                locate, progress, settings, max_tries=_TRIES
            )
            file = next((f for f in manifest.files if f.path == path), None)
            if file is None:
                raise FileNotFoundError(f"{path} is not in {model_id}.")
            part = Part(model_id, target_dir, manifest, check_index=False)
            await transfer.fetch_file(
                hub, part, file, progress, settings, Locks(), max_tries=_TRIES
            )
        except NotFoundError as error:
            raise FileNotFoundError(error.message) from error
    on_progress(file.size, file.size, True)
    return target
