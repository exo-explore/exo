import contextlib
import json
from collections import OrderedDict
from collections.abc import Iterator
from datetime import datetime, timezone
from io import BufferedRandom, BufferedReader
from pathlib import Path

import msgspec
import zstandard
from loguru import logger
from pydantic import TypeAdapter

from exo.shared.types.events import Event

_EVENT_ADAPTER: TypeAdapter[Event] = TypeAdapter(Event)

_HEADER_SIZE = 4  # uint32 big-endian
_OFFSET_CACHE_SIZE = 128
_MAX_ARCHIVES = 5


def _serialize_event(event: Event) -> bytes:
    return msgspec.msgpack.encode(event.model_dump(mode="json"))


def _deserialize_event(raw: bytes) -> Event:
    # Decode msgpack into a Python dict, then re-encode as JSON for Pydantic.
    # Pydantic's validate_json() uses JSON-mode coercion (e.g. string -> enum)
    # even under strict=True, whereas validate_python() does not. Going through
    # JSON is the only way to get correct round-trip deserialization without
    # disabling strict mode or adding casts everywhere.
    as_json = json.dumps(msgspec.msgpack.decode(raw, type=dict))
    return _EVENT_ADAPTER.validate_json(as_json)


def _unpack_header(header: bytes) -> int:
    return int.from_bytes(header, byteorder="big")


def _skip_record(f: BufferedReader) -> bool:
    """Skip one length-prefixed record. Returns False on EOF."""
    header = f.read(_HEADER_SIZE)
    if len(header) < _HEADER_SIZE:
        return False
    f.seek(_unpack_header(header), 1)
    return True


def _read_record(f: BufferedReader) -> Event | None:
    """Read one length-prefixed record. Returns None on EOF."""
    header = f.read(_HEADER_SIZE)
    if len(header) < _HEADER_SIZE:
        return None
    length = _unpack_header(header)
    payload = f.read(length)
    if len(payload) < length:
        return None
    return _deserialize_event(payload)


class DiskEventLog:
    """Append-only event log backed by a file on disk.

    On-disk format: sequence of length-prefixed msgpack records.
    Each record is [4-byte big-endian uint32 length][msgpack payload].

    Uses a bounded LRU cache of event index → byte offset for efficient
    random access without storing an offset per event.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)
        self._active_path = directory / "events.bin"
        self._offset_cache: OrderedDict[int, int] = OrderedDict()
        self._count: int = 0

        # Rotate stale active file from a previous session/crash
        if self._active_path.exists():
            self._rotate(self._active_path, self._directory)

        self._file: BufferedRandom = open(self._active_path, "w+b")  # noqa: SIM115

    def _cache_offset(self, idx: int, offset: int) -> None:
        self._offset_cache[idx] = offset
        self._offset_cache.move_to_end(idx)
        if len(self._offset_cache) > _OFFSET_CACHE_SIZE:
            self._offset_cache.popitem(last=False)

    def _seek_to(self, f: BufferedReader, target_idx: int) -> None:
        """Seek f to the byte offset of event target_idx, using cache or scanning forward."""
        if target_idx in self._offset_cache:
            self._offset_cache.move_to_end(target_idx)
            f.seek(self._offset_cache[target_idx])
            return

        # Find the highest cached index before target_idx
        scan_from_idx = 0
        scan_from_offset = 0
        for cached_idx in self._offset_cache:
            if cached_idx < target_idx:
                scan_from_idx = cached_idx
                scan_from_offset = self._offset_cache[cached_idx]

        # Scan forward, skipping records
        f.seek(scan_from_offset)
        for _ in range(scan_from_idx, target_idx):
            _skip_record(f)

        self._cache_offset(target_idx, f.tell())

    def append(self, event: Event) -> None:
        packed = _serialize_event(event)
        self._file.write(len(packed).to_bytes(_HEADER_SIZE, byteorder="big"))
        self._file.write(packed)
        self._count += 1

    def read_range(self, start: int, end: int) -> Iterator[Event]:
        """Yield events from index start (inclusive) to end (exclusive)."""
        end = min(end, self._count)
        if start < 0 or end < 0 or start >= end:
            return

        self._file.flush()
        with open(self._active_path, "rb") as f:
            self._seek_to(f, start)
            for _ in range(end - start):
                event = _read_record(f)
                if event is None:
                    break
                yield event

            # Cache where we ended up so the next sequential read is a hit
            if end < self._count:
                self._cache_offset(end, f.tell())

    def read_all(self) -> Iterator[Event]:
        """Yield all events from the log one at a time."""
        if self._count == 0:
            return
        self._file.flush()
        with open(self._active_path, "rb") as f:
            for _ in range(self._count):
                event = _read_record(f)
                if event is None:
                    break
                yield event

    def __len__(self) -> int:
        return self._count

    def close(self) -> None:
        """Close the file and rotate active file to compressed archive."""
        if self._file.closed:
            return
        self._file.close()
        if self._active_path.exists() and self._count > 0:
            self._rotate(self._active_path, self._directory)
        elif self._active_path.exists():
            self._active_path.unlink()

    @staticmethod
    def _rotate(source: Path, directory: Path) -> None:
        """Compress source into a timestamped archive.

        Keeps at most ``_MAX_ARCHIVES`` compressed copies.  Oldest beyond
        the limit are deleted.
        """
        try:
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_%f")
            dest = directory / f"events.{stamp}.bin.zst"
            compressor = zstandard.ZstdCompressor()
            with open(source, "rb") as f_in, open(dest, "wb") as f_out:
                compressor.copy_stream(f_in, f_out)
            source.unlink()
            logger.info(f"Rotated event log: {source} -> {dest}")

            # Prune oldest archives beyond the limit
            archives = sorted(directory.glob("events.*.bin.zst"))
            for old in archives[:-_MAX_ARCHIVES]:
                old.unlink()
        except Exception as e:
            logger.opt(exception=e).warning(f"Failed to rotate event log {source}")
            # Clean up the source even if compression fails
            with contextlib.suppress(OSError):
                source.unlink()
