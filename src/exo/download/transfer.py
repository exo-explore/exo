"""Downloading a model's files of one pinned commit into one directory.

Each file is fetched into ``<path>.partial`` from wherever the last attempt
stopped, hashed as it is written, and renamed into place once the hash matches.
Network trouble is retried for as long as bytes keep coming (the backoff resets
after every attempt that made progress); only a long stretch with no progress at
all, or an error retrying can't fix, ends the download.

File I/O runs in worker threads that a cancellation waits for, so nothing is
written, created or closed behind the back of whoever cancelled. Writes are
addressed by offset, so a large file could later be fetched as several ranges in
parallel.
"""

from __future__ import annotations

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
from functools import partial
from pathlib import Path
from typing import Protocol, final

import anyio
from anyio import to_thread
from loguru import logger

from exo.download import store
from exo.download.hub import (
    Hub,
    PermanentError,
    RateLimitedError,
    RepoFile,
    TransientError,
    chunks,
    is_safe_path,
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
    # Failing to reach the Hub before any file for this long fails the download
    # (and it is retried later); once files flow, only no new bytes for
    # ``give_up_after`` does.
    unreachable_after: float = 120.0
    give_up_after: float = 30 * 60.0
    progress_interval: float = 1.0
    rescan_interval: float = 60.0
    chunk_size: int = 1 << 20


class IntegrityError(PermanentError):
    """A file failed its hash check twice."""


@final
class Locks:
    """What concurrent downloads on one node must not do at the same time:
    choose a models dir (and reserve its space), and write the same repo dir."""

    def __init__(self) -> None:
        self.space = anyio.Lock()
        self._dirs: dict[Path, anyio.Lock] = {}
        self._io: anyio.CapacityLimiter | None = None

    def directory(self, path: Path) -> anyio.Lock:
        return self._dirs.setdefault(path, anyio.Lock())

    @property
    def io(self) -> anyio.CapacityLimiter:
        """Threads for file I/O, apart from anyio's shared pool, so a hung disk
        can't starve the rest of the node."""
        if self._io is None:
            self._io = anyio.CapacityLimiter(16)
        return self._io


async def _io[*A, R](locks: Locks, func: Callable[[*A], R], *args: *A) -> R:
    """Run blocking file I/O in a thread that a cancellation waits for."""
    return await to_thread.run_sync(func, *args, limiter=locks.io)


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
        # the bytes this download still needed in each repo dir when it started
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

    def reset(self) -> None:
        self.files.clear()
        self.needed.clear()

    def remaining(self) -> dict[Path, int]:
        """What each repo dir of this download still needs, roughly."""
        left = max(0, sum(self.needed.values()) - self.fetched)
        total = sum(self.needed.values()) or 1
        return {path: left * needed // total for path, needed in self.needed.items()}

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
    locks: Locks,
    reserved: Callable[[ModelId], Mapping[Path, int]],
) -> list[Part]:
    """Pin each repo of the model to a commit and pick where it goes.

    A repo with a manifest resumes its commit and files; otherwise ``main`` is
    resolved now. The dir is chosen, and its space reserved, under ``locks.space``
    (``reserved(repo)`` is what other downloads still need, by models dir, not
    counting this repo), and the manifest is written there before any file is
    fetched.
    """
    progress.reset()
    result: list[Part] = []
    for repo, only in parts_of(card):
        with _disk_errors(repo):
            existing = await _io(locks, _existing_manifest, repo)
        if existing is not None:
            commit, files = existing[1].commit, existing[1].files
        else:
            commit = await hub.resolve_commit(repo)
            files = store.wanted_files(
                await hub.list_files(repo, commit),
                image_model=store.is_image_model(card),
                only=only,
            )
            if not files:
                raise PermanentError(f"{repo} has no files to download.")
        if unsafe := next((f.path for f in files if not is_safe_path(f.path)), None):
            raise PermanentError(f"{repo} lists a file exo won't write: {unsafe!r}.")
        manifest = Manifest(repo=str(repo), commit=commit, files=files)
        with _disk_errors(repo):
            async with locks.space:
                root, needed = await _io(
                    locks, store.choose_root, repo, files, reserved(repo)
                )
                model_dir = root / repo.normalize()
                progress.needed[model_dir] = needed
            if existing is not None and existing[0] != model_dir:
                # the old dir ran out of room: its partial files are of no more use
                await _io(locks, store.drop_partials, existing[0])
            current = await _io(locks, store.read_manifest, model_dir)
            if current is None or (current.commit, current.files) != (commit, files):
                await _io(locks, store.write_manifest, model_dir, manifest)
            present = await _io(locks, _present, model_dir, files)
        prefix = "" if repo == card.model_id else f"{repo}/"
        check_index = only is None and not store.is_image_model(card)
        result.append(Part(repo, model_dir, manifest, prefix, check_index))
        for file, downloaded in zip(files, present, strict=True):
            progress.files[prefix + file.path] = FileProgress(
                size=file.size, downloaded=downloaded
            )
    progress.model_dir = result[0].model_dir
    return result


def _existing_manifest(repo: ModelId) -> tuple[Path, Manifest] | None:
    for model_dir in store.candidates(repo):
        if not store.is_read_only(model_dir) and (
            manifest := store.read_manifest(model_dir)
        ):
            return model_dir, manifest
    return None


def _present(model_dir: Path, files: list[RepoFile]) -> list[int]:
    return [store.present_bytes(model_dir, f) for f in files]


# --- downloading ---------------------------------------------------------------------


async def download(
    hub: Hub,
    parts: list[Part],
    progress: Progress,
    settings: DownloadSettings,
    locks: Locks,
) -> None:
    """Fetch every file of every part; mark each manifest complete when done.

    Only one download writes a repo dir at a time (models can share a vision
    weights repo). The first error retrying can't fix cancels the other files.
    """
    semaphore = anyio.Semaphore(settings.parallel_files)

    async def one(part: Part, file: RepoFile) -> None:
        async with semaphore:
            await fetch_file(hub, part, file, progress, settings, locks)

    async def whole(part: Part) -> None:
        async with locks.directory(part.model_dir):
            async with anyio.create_task_group() as group:
                # small files first: the index is checked before the weights arrive
                for file in sorted(part.manifest.files, key=lambda f: f.size):
                    group.start_soon(one, part, file)
            with _disk_errors(part.repo):
                await _io(
                    locks,
                    store.write_manifest,
                    part.model_dir,
                    part.manifest.model_copy(update={"complete": True}),
                )

    try:
        async with anyio.create_task_group() as group:
            for part in parts:
                group.start_soon(whole, part)
    except BaseExceptionGroup as group_error:
        raise _first_error(group_error) from None


async def retrying[T](
    operation: Callable[[], Awaitable[T]],
    progress: Progress,
    settings: DownloadSettings,
    *,
    max_tries: int | None = None,
) -> T:
    """Run ``operation`` until it succeeds, retrying network trouble.

    The wait between tries doubles from ``retry_base`` to ``retry_cap`` and starts
    over after a try that made progress; a 429 waits as long as Hugging Face asks.
    Gives up when nothing new arrived for long enough (from the first try or the
    last new byte, whichever is later): ``unreachable_after`` while no byte has
    arrived since the download (re)started, ``give_up_after`` once one has; or
    after ``max_tries``.
    """
    delay = settings.retry_base
    started = progress.clock()
    tries = 0
    while True:
        tries += 1
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
        if max_tries is not None and tries >= max_tries:
            raise PermanentError(f"Gave up after {tries} tries: {progress.last_error}.")
        if progress.fetched == 0:
            limit, giving_up = settings.unreachable_after, "Can't reach Hugging Face"
        else:
            limit = settings.give_up_after
            giving_up = f"No progress for {duration(limit)}"
        if progress.clock() - max(started, progress.last_progress) > limit:
            raise PermanentError(f"{giving_up}: {progress.last_error}.")
        logger.info(f"{progress.last_error}; retrying in {wait:.1f}s")
        await anyio.sleep(wait * random.uniform(0.8, 1.2))


def _first_error(group_error: BaseExceptionGroup[BaseException]) -> BaseException:
    for error in group_error.exceptions:
        if isinstance(error, BaseExceptionGroup):
            return _first_error(error)  # pyright: ignore[reportUnknownArgumentType]
        if not isinstance(error, anyio.get_cancelled_exc_class()):
            return error
    return group_error.exceptions[0]


def duration(seconds: float) -> str:
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


@dataclass
class FileState:
    """What one file's attempts share: the hash of the ``.partial`` so far."""

    file: RepoFile
    hasher: Hasher
    hashed: int = 0
    # the .partial was on disk before this run (a power cut, another revision's
    # leftovers): a mismatch then isn't the server's fault, and isn't counted
    inherited: bool = False
    mismatches: int = 0

    def restart(self) -> None:
        self.hasher = _new_hasher(self.file)
        self.hashed = 0


class _MismatchError(Exception):
    pass


class _StartOverError(Exception):
    """The partial file is useless; fetch the file from the beginning, now."""


async def fetch_file(
    hub: Hub,
    part: Part,
    file: RepoFile,
    progress: Progress,
    settings: DownloadSettings,
    locks: Locks,
    *,
    max_tries: int | None = None,
) -> None:
    """Make ``file`` of ``part`` present and verified in its dir."""
    key = part.key_prefix + file.path
    final = part.model_dir / file.path
    partial_path = part.model_dir / (file.path + PARTIAL_SUFFIX)
    progress.files.setdefault(key, FileProgress(size=file.size))
    with _disk_errors(part.repo):
        there = await _io(locks, _already_there, final, file)
        inherited = (await _io(locks, store.file_size, partial_path) or 0) > 0
    if not there:
        state = FileState(file, _new_hasher(file), inherited=inherited)

        async def attempt() -> None:
            while True:
                try:
                    with _disk_errors(part.repo):
                        await _fetch_once(
                            hub,
                            part,
                            state,
                            partial_path,
                            key,
                            progress,
                            settings,
                            locks,
                        )
                        await _io(locks, _rename, partial_path, final, file)
                    return
                except _MismatchError:
                    if not state.inherited:
                        state.mismatches += 1
                    logger.warning(
                        f"{part.repo}/{file.path} failed its hash check"
                        f" ({state.mismatches} bad downloads)"
                    )
                    if state.mismatches >= 2:
                        await _io(locks, _remove, partial_path)
                        progress.rewind(key)
                        raise IntegrityError(
                            f"{file.path} failed its hash check twice. Try again later."
                        ) from None
                except _StartOverError:
                    pass
                await _io(locks, _remove, partial_path)
                state.restart()
                progress.rewind(key)

        await retrying(attempt, progress, settings, max_tries=max_tries)
    progress.files[key].downloaded = file.size
    progress.files[key].done = True
    if file.path.endswith(".safetensors.index.json"):
        await _io(locks, _check_index, part, final)


async def _fetch_once(
    hub: Hub,
    part: Part,
    state: FileState,
    partial_path: Path,
    key: str,
    progress: Progress,
    settings: DownloadSettings,
    locks: Locks,
) -> None:
    """One attempt: bring the ``.partial`` up to the whole file and verify it."""
    file = state.file
    offset = await _io(locks, store.file_size, partial_path) or 0
    if offset > file.size:
        raise _StartOverError()
    if offset < state.hashed:
        state.restart()  # shorter than what was hashed: it changed under us
    if offset == 0:
        state.inherited = False
    progress.files[key].downloaded = offset
    await _io(locks, partial(partial_path.parent.mkdir, parents=True, exist_ok=True))
    # not abandoned on cancel: the descriptor is always ours to close
    descriptor = await _io(locks, os.open, partial_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        while state.hashed < offset:
            # in steps, so a cancellation waits for one step at most
            await _io(locks, _hash_range, descriptor, state, offset)
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
                    await _io(locks, os.ftruncate, descriptor, 0)
                    progress.rewind(key)
                    offset = 0
                    state.restart()
                    state.inherited = False
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
                        await _io(locks, _write_at, descriptor, chunk, offset, state)
                        offset += len(chunk)
                        progress.advance(key, len(chunk))
        if offset != file.size:
            raise TransientError("the connection was cut")
        if state.hasher.hexdigest() != _expected_hash(file):
            raise _MismatchError()
        await _io(locks, os.fsync, descriptor)
    finally:
        with anyio.CancelScope(shield=True):
            await _io(locks, os.close, descriptor)


def _write_at(descriptor: int, data: bytes, offset: int, state: FileState) -> None:
    view = memoryview(data)
    position = offset
    while view:
        written = os.pwrite(descriptor, view, position)
        view = view[written:]
        position += written
    if state.hashed == offset:
        state.hasher.update(data)
        state.hashed = offset + len(data)


def _hash_range(descriptor: int, state: FileState, end: int) -> None:
    """Hash up to 64 MiB more of the ``.partial``, towards ``end``."""
    block = os.pread(descriptor, min(64 << 20, end - state.hashed), state.hashed)
    if not block:
        raise _StartOverError()
    state.hasher.update(block)
    state.hashed += len(block)


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


def _rename(partial_path: Path, final: Path, file: RepoFile) -> None:
    try:
        os.replace(partial_path, final)
    except FileNotFoundError:
        # someone else (fetching the same commit) got there first
        if store.file_size(final) != file.size:
            raise


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
        raise PermanentError(
            disk_error_message(error, f"downloading {repo}")
        ) from error


def disk_error_message(error: OSError, doing: str) -> str:
    if error.errno in (errno.ENOSPC, errno.EDQUOT):
        return f"Ran out of disk space {doing}."
    reason = os.strerror(error.errno) if error.errno is not None else str(error)
    if error.errno in (errno.EROFS, errno.EACCES, errno.EPERM):
        return f"Can't write to the models folder {doing}: {reason}."
    return f"Disk error {doing}: {reason}."
