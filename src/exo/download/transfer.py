"""Downloading a model's files of one pinned commit into one directory.

Each file is fetched into ``<path>.partial`` from wherever the last attempt
stopped, hashed as it is written, and renamed into place once the hash matches.
Network trouble is retried for as long as bytes keep coming (the backoff resets
after every attempt that made progress); only a long stretch with no progress at
all, or an error retrying can't fix, ends the download.

Writes are addressed by offset, so a large file could later be fetched as
several ranges in parallel.
"""

from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import os
import random
import time
from collections import deque
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import aclosing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, final

from loguru import logger

from exo.download import store
from exo.download.hub import (
    Hub,
    PermanentError,
    RateLimitedError,
    RepoFile,
    TransientError,
    chunks,
)
from exo.download.store import PARTIAL_SUFFIX, Manifest
from exo.shared.models.model_cards import ModelCard
from exo.shared.types.common import ModelId


@dataclass(frozen=True)
class DownloadSettings:
    # A connection that delivers nothing for this long is dropped and retried.
    idle_timeout: float = 60.0
    connect_timeout: float = 30.0
    # Cap on listing files and resolving the commit.
    metadata_timeout: float = 60.0
    parallel_files: int = 8
    # Backoff between retries of a file: doubles from base to cap, and goes back
    # to base after any attempt that made progress.
    retry_base: float = 1.0
    retry_cap: float = 30.0
    # Longest wait asked for by a 429, and a multiplier on such waits (for tests).
    rate_limit_cap: float = 300.0
    server_wait_scale: float = 1.0
    # A download with no new bytes for this long fails with the last error.
    give_up_after: float = 30 * 60.0
    # A failed download is retried after this, doubling up to the cap.
    failed_retry_base: float = 60.0
    failed_retry_cap: float = 600.0
    progress_interval: float = 1.0
    rescan_interval: float = 60.0
    chunk_size: int = 1 << 20


class IntegrityError(PermanentError):
    """A file failed its hash check twice."""


# --- progress ------------------------------------------------------------------------


@dataclass
class FileProgress:
    size: int
    downloaded: int = 0
    done: bool = False
    # bytes received this session, and when the first of them arrived
    fetched: int = 0
    started: float | None = None

    def speed(self, now: float) -> float:
        if self.done or self.started is None or now <= self.started:
            return 0.0
        return self.fetched / (now - self.started)


@final
class Progress:
    """Byte counters of one download, written by the file tasks, read by the reporter."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.files: dict[str, FileProgress] = {}
        self.fetched = 0
        self.last_progress = clock()
        self.last_error: str | None = None
        self.model_dir: Path | None = None
        # the bytes each running download still needs, by models dir
        self.needed: dict[Path, int] = {}
        self._samples: deque[tuple[float, int]] = deque([(self.clock(), 0)])

    @property
    def total(self) -> int:
        return sum(f.size for f in self.files.values())

    @property
    def downloaded(self) -> int:
        return sum(f.downloaded for f in self.files.values())

    @property
    def completed_files(self) -> int:
        return sum(1 for f in self.files.values() if f.done)

    def remaining(self) -> dict[Path, int]:
        left = max(0, sum(self.needed.values()) - self.fetched)
        total = sum(self.needed.values()) or 1
        return {root: left * needed // total for root, needed in self.needed.items()}

    def advance(self, key: str, n: int) -> None:
        file = self.files[key]
        now = self.clock()
        if file.started is None:
            file.started = now
        file.downloaded += n
        file.fetched += n
        self.fetched += n
        self.last_progress = now

    def rewind(self, key: str) -> None:
        self.files[key].downloaded = 0

    def speed(self) -> float:
        """Bytes per second over about the last 10 seconds."""
        now = self.clock()
        self._samples.append((now, self.fetched))
        while len(self._samples) > 2 and now - self._samples[1][0] >= 10:
            self._samples.popleft()
        start_time, start_bytes = self._samples[0]
        return (
            (self.fetched - start_bytes) / (now - start_time)
            if now > start_time
            else 0.0
        )


# --- planning: which commit, which files, which dir ----------------------------------


@dataclass(frozen=True)
class Part:
    """One repo's share of a download (a model, or its separate vision weights)."""

    repo: ModelId
    model_dir: Path
    manifest: Manifest
    key_prefix: str = ""
    check_index: bool = True


def parts_of(card: ModelCard) -> list[tuple[ModelId, tuple[str, ...] | None]]:
    parts: list[tuple[ModelId, tuple[str, ...] | None]] = [(card.model_id, None)]
    if (vision := store.vision_repo(card)) is not None:
        parts.append((vision, ("*.safetensors", "config.json")))
    return parts


async def plan(
    hub: Hub,
    card: ModelCard,
    progress: Progress,
    reserved: Callable[[], Mapping[Path, int]],
) -> list[Part]:
    """Pin each repo of the model to a commit and pick where it goes.

    A repo with a manifest resumes its commit; otherwise ``main`` is resolved now
    and the manifest is written before any file is fetched.
    """
    result: list[Part] = []
    for repo, only in parts_of(card):
        existing = next(
            (
                (d, manifest)
                for d in store.candidates(repo)
                if not store.is_read_only(d) and (manifest := store.read_manifest(d))
            ),
            None,
        )
        if existing is not None:
            model_dir, manifest = existing
            root, needed = await asyncio.to_thread(
                store.choose_root, repo, manifest.files, reserved(), model_dir.parent
            )
        else:
            commit = await hub.resolve_commit(repo)
            files = store.wanted_files(
                await hub.list_files(repo, commit),
                image_model=store.is_image_model(card),
                only=only,
            )
            if not files:
                raise PermanentError(f"{repo} has no files to download.")
            root, needed = await asyncio.to_thread(
                store.choose_root, repo, files, reserved()
            )
            model_dir = root / repo.normalize()
            manifest = Manifest(repo=str(repo), commit=commit, files=files)
            await asyncio.to_thread(store.write_manifest, model_dir, manifest)
        progress.needed[root] = progress.needed.get(root, 0) + needed
        prefix = "" if repo == card.model_id else f"{repo}/"
        check_index = only is None and not store.is_image_model(card)
        result.append(Part(repo, model_dir, manifest, prefix, check_index))
        for file in manifest.files:
            progress.files[prefix + file.path] = FileProgress(
                size=file.size,
                downloaded=await asyncio.to_thread(
                    store.present_bytes, model_dir, file
                ),
            )
    progress.model_dir = result[0].model_dir
    return result


# --- downloading ---------------------------------------------------------------------


async def download(
    hub: Hub, parts: list[Part], progress: Progress, settings: DownloadSettings
) -> None:
    """Fetch every file of every part; mark each manifest complete when done.

    The first error retrying can't fix cancels the other files.
    """
    semaphore = asyncio.Semaphore(settings.parallel_files)

    async def one(part: Part, file: RepoFile) -> None:
        async with semaphore:
            await _fetch_file(hub, part, file, progress, settings)

    try:
        async with asyncio.TaskGroup() as group:
            for part in parts:
                # small files first: the index is checked before the weights arrive
                for file in sorted(part.manifest.files, key=lambda f: f.size):
                    group.create_task(one(part, file))
    except BaseExceptionGroup as group_error:
        raise _first_error(group_error) from None
    for part in parts:
        await asyncio.to_thread(
            store.write_manifest,
            part.model_dir,
            part.manifest.model_copy(update={"complete": True}),
        )


async def retrying[T](
    operation: Callable[[], Awaitable[T]],
    progress: Progress,
    settings: DownloadSettings,
) -> T:
    """Run ``operation`` until it succeeds, retrying network trouble.

    The wait between tries doubles from ``retry_base`` to ``retry_cap`` and starts
    over after a try that made progress; a 429 waits as long as Hugging Face asks.
    Gives up only when the whole download got no bytes for ``give_up_after``.
    """
    delay = settings.retry_base
    while True:
        before = progress.fetched
        try:
            return await operation()
        except RateLimitedError as error:
            progress.last_error = error.message
            asked = (
                error.retry_after * settings.server_wait_scale
                if error.retry_after
                else delay
            )
            wait = min(asked, settings.rate_limit_cap)
            delay = min(delay * 2, settings.retry_cap)
        except TransientError as error:
            progress.last_error = error.message
            if progress.fetched > before:
                delay = settings.retry_base
            wait = delay
            delay = min(delay * 2, settings.retry_cap)
        if progress.clock() - progress.last_progress > settings.give_up_after:
            raise PermanentError(
                f"No progress for {_duration(settings.give_up_after)}: {progress.last_error}."
            )
        logger.info(f"{progress.last_error}; retrying in {wait:.1f}s")
        await asyncio.sleep(wait * random.uniform(0.8, 1.2))


def _first_error(group_error: BaseExceptionGroup[BaseException]) -> BaseException:
    for error in group_error.exceptions:
        if isinstance(error, BaseExceptionGroup):
            return _first_error(error)  # pyright: ignore[reportUnknownArgumentType]
        if not isinstance(error, asyncio.CancelledError):
            return error
    return group_error.exceptions[0]


def _duration(seconds: float) -> str:
    return f"{seconds / 60:.0f} minutes" if seconds >= 120 else f"{seconds:.0f} seconds"


class Hasher(Protocol):
    def update(self, data: bytes, /) -> None: ...
    def hexdigest(self) -> str: ...


def _new_hasher(file: RepoFile) -> Hasher:
    if file.sha256 is not None:
        return hashlib.sha256()
    hasher = hashlib.sha1()
    hasher.update(b"blob %d\0" % file.size)
    return hasher


def _expected_hash(file: RepoFile) -> str:
    return file.sha256 if file.sha256 is not None else file.git_sha1


class _MismatchError(Exception):
    pass


class _StartOverError(Exception):
    """The partial file is useless; fetch the file from the beginning, now."""


async def _fetch_file(
    hub: Hub, part: Part, file: RepoFile, progress: Progress, settings: DownloadSettings
) -> None:
    key = part.key_prefix + file.path
    final = part.model_dir / file.path
    partial = part.model_dir / (file.path + PARTIAL_SUFFIX)
    with _disk_errors(part.repo):
        there = await asyncio.to_thread(_already_there, final, file)
    if not there:
        mismatches = 0

        async def attempt() -> None:
            nonlocal mismatches
            while True:
                try:
                    with _disk_errors(part.repo):
                        await _fetch_once(
                            hub, part, file, partial, key, progress, settings
                        )
                        await asyncio.to_thread(os.replace, partial, final)
                    return
                except _MismatchError:
                    mismatches += 1
                    logger.warning(f"{part.repo}/{file.path} failed its hash check")
                    if mismatches >= 2:
                        await asyncio.to_thread(_remove, partial)
                        progress.rewind(key)
                        raise IntegrityError(
                            f"{file.path} failed its hash check twice. Try again later."
                        ) from None
                except _StartOverError:
                    pass
                await asyncio.to_thread(_remove, partial)
                progress.rewind(key)

        await retrying(attempt, progress, settings)
    progress.files[key].downloaded = file.size
    progress.files[key].done = True
    if file.path.endswith(".safetensors.index.json"):
        await asyncio.to_thread(_check_index, part, final)


async def _fetch_once(
    hub: Hub,
    part: Part,
    file: RepoFile,
    partial: Path,
    key: str,
    progress: Progress,
    settings: DownloadSettings,
) -> None:
    """One attempt: bring ``partial`` up to the whole file and verify it."""
    offset = await asyncio.to_thread(store.file_size, partial) or 0
    if offset > file.size:
        raise _StartOverError()
    progress.files[key].downloaded = offset
    await asyncio.to_thread(partial.parent.mkdir, parents=True, exist_ok=True)
    descriptor = await asyncio.to_thread(
        os.open, partial, os.O_RDWR | os.O_CREAT, 0o644
    )
    try:
        hasher = _new_hasher(file)
        if offset > 0:
            await asyncio.to_thread(_hash_existing, descriptor, offset, hasher)
        if offset < file.size:
            async with hub.open(
                part.repo, part.manifest.commit, file.path, offset
            ) as response:
                if response.status == 416:
                    if offset == 0:
                        raise PermanentError(
                            f"Hugging Face won't send {file.path} (416)."
                        )
                    raise _StartOverError()
                if response.status == 200 and offset > 0:
                    # the server ignored the Range: take its bytes from the start
                    await asyncio.to_thread(os.ftruncate, descriptor, 0)
                    progress.rewind(key)
                    offset = 0
                    hasher = _new_hasher(file)
                elif response.status == 206 and not _range_starts_at(
                    response.headers.get("Content-Range", ""), offset
                ):
                    if offset == 0:
                        raise TransientError(
                            "Hugging Face sent the wrong part of a file"
                        )
                    raise _StartOverError()
                async with aclosing(chunks(response, settings.chunk_size)) as body:
                    async for chunk in body:
                        if offset + len(chunk) > file.size:
                            raise _MismatchError()
                        await asyncio.to_thread(
                            _write_at, descriptor, chunk, offset, hasher
                        )
                        offset += len(chunk)
                        progress.advance(key, len(chunk))
        if offset != file.size:
            raise TransientError("the connection was cut")
        if hasher.hexdigest() != _expected_hash(file):
            raise _MismatchError()
        await asyncio.to_thread(os.fsync, descriptor)
    finally:
        await asyncio.to_thread(os.close, descriptor)


def _write_at(descriptor: int, data: bytes, offset: int, hasher: Hasher) -> None:
    view = memoryview(data)
    while view:
        written = os.pwrite(descriptor, view, offset)
        view = view[written:]
        offset += written
    hasher.update(data)


def _hash_existing(descriptor: int, length: int, hasher: Hasher) -> None:
    position = 0
    while position < length:
        block = os.pread(descriptor, min(8 << 20, length - position), position)
        if not block:
            break
        hasher.update(block)
        position += len(block)


def _range_starts_at(content_range: str, offset: int) -> bool:
    # "bytes 100-199/1000"
    first = content_range.removeprefix("bytes ").partition("-")[0]
    return first.isdigit() and int(first) == offset


def _already_there(final: Path, file: RepoFile) -> bool:
    """Whether ``final`` is the file. Small files are hash-checked (cheap); large ones
    were hash-checked before they got their final name."""
    size = store.file_size(final)
    if size is None:
        return False
    if size == file.size and (
        file.sha256 is not None or _git_sha1(final, file.size) == file.git_sha1
    ):
        return True
    final.unlink()
    return False


def _git_sha1(path: Path, size: int) -> str:
    hasher = hashlib.sha1(b"blob %d\0" % size)
    hasher.update(path.read_bytes())
    return hasher.hexdigest()


def _check_index(part: Part, index_path: Path) -> None:
    """Fail early when the index names a weight file the repo doesn't have."""
    if not part.check_index:
        return
    try:
        weight_map: object = json.loads(index_path.read_text()).get("weight_map")  # pyright: ignore[reportAny]
    except (OSError, ValueError, AttributeError):
        return
    if not isinstance(weight_map, dict):
        return
    folder = Path(index_path.relative_to(part.model_dir).parent)
    listed = {f.path for f in part.manifest.files}
    for name in sorted({str(v) for v in weight_map.values()}):  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        if (folder / name).as_posix() not in listed:
            raise PermanentError(
                f"{part.repo} is incomplete on Hugging Face: {name} is in its index "
                "but not in the repo."
            )


def _remove(path: Path) -> None:
    path.unlink(missing_ok=True)


@contextmanager
def _disk_errors(repo: ModelId) -> Iterator[None]:
    """Turn file-system errors into a :class:`PermanentError` users can act on."""
    try:
        yield
    except TimeoutError:
        raise
    except OSError as error:
        logger.opt(exception=error).warning(f"Disk error downloading {repo}")
        if error.errno in (errno.ENOSPC, errno.EDQUOT):
            raise PermanentError(
                f"Ran out of disk space downloading {repo}."
            ) from error
        reason = os.strerror(error.errno) if error.errno is not None else str(error)
        if error.errno in (errno.EROFS, errno.EACCES, errno.EPERM):
            raise PermanentError(
                f"Can't write to the models folder: {reason}."
            ) from error
        raise PermanentError(f"Disk error downloading {repo}: {reason}.") from error
