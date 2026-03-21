"""Remote KV cache tier backed by iPad HTTP blob store.

Enabled via the EXO_IPAD_CACHE_URL environment variable.
When not set, RemoteCacheTier is never constructed and has zero runtime cost.

Design:
  - store_async(): fire-and-forget push to iPad over LAN.  Zero hot-path penalty.
  - fetch(): check disk index for prefix match, then GET blob from iPad.
  - Index stored in ~/.exo/remote_cache_index.json — allows prefix matching
    without a network roundtrip.
  - Only standard KVCache layers are serialized.  RotatingKVCache /
    QuantizedKVCache / ArraysCache layers are replaced with fresh KVCache() on
    deserialization (graceful degradation — partial re-prefill only for those).
"""

from __future__ import annotations

import io
import json
import logging
import os
import struct
import threading
import time
import urllib.request
from http.client import HTTPResponse
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache

from exo.shared.types.mlx import KVCacheType, Model

logger = logging.getLogger(__name__)

_MAGIC: bytes = b"EXOC"
_CACHE_URLS_FILE: Path = Path.home() / ".exo" / "remote_cache_urls"
_LAYER_TYPE_SKIP: int = 0
_LAYER_TYPE_KV: int = 1
_DEFAULT_INDEX_PATH: Path = Path.home() / ".exo" / "remote_cache_index.json"


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _unpack1(fmt: str, buf: io.BytesIO, size: int) -> int:
    """Unpack a single integer from *buf* and return it as a typed int."""
    return cast(int, struct.unpack(fmt, buf.read(size))[0])


def _write_array(buf: io.BytesIO, arr: mx.array) -> None:
    """Serialise an mx.array into *buf* using big-endian structs + raw bytes."""
    np_arr = np.array(arr)
    dtype_bytes = np_arr.dtype.name.encode()
    buf.write(struct.pack(">B", len(dtype_bytes)))
    buf.write(dtype_bytes)
    shape: tuple[int, ...] = cast(tuple[int, ...], np_arr.shape)
    buf.write(struct.pack(">B", len(shape)))
    for dim in shape:
        buf.write(struct.pack(">I", dim))
    raw = np_arr.tobytes()
    buf.write(struct.pack(">I", len(raw)))
    buf.write(raw)


def _read_array(buf: io.BytesIO) -> mx.array:
    """Deserialise an mx.array previously written by _write_array."""
    dtype_len = _unpack1(">B", buf, 1)
    dtype_name = buf.read(dtype_len).decode()
    ndim = _unpack1(">B", buf, 1)
    shape = tuple(_unpack1(">I", buf, 4) for _ in range(ndim))
    data_len = _unpack1(">I", buf, 4)
    raw = buf.read(data_len)
    np_arr = np.frombuffer(raw, dtype=np.dtype(dtype_name)).reshape(shape)
    return mx.array(np_arr)


def _serialize(prompt_tokens: mx.array, cache: KVCacheType) -> bytes:
    """Serialise prompt tokens + KV cache to bytes for remote storage."""
    buf = io.BytesIO()
    buf.write(_MAGIC)
    buf.write(struct.pack(">I", len(cache)))
    for layer in cache:
        if isinstance(layer, KVCache):
            keys: mx.array | None = getattr(layer, "keys", None)
            values: mx.array | None = getattr(layer, "values", None)
            if keys is not None and values is not None:
                buf.write(struct.pack(">B", _LAYER_TYPE_KV))
                _write_array(buf, keys)
                _write_array(buf, values)
                offset: int = getattr(layer, "offset", 0)
                buf.write(struct.pack(">I", offset))
                continue
        buf.write(struct.pack(">B", _LAYER_TYPE_SKIP))
    _write_array(buf, prompt_tokens)
    return buf.getvalue()


def _deserialize(data: bytes) -> tuple[mx.array, KVCacheType]:
    """Deserialise bytes produced by _serialize."""
    buf = io.BytesIO(data)
    magic = buf.read(4)
    if magic != _MAGIC:
        raise ValueError(f"Invalid magic: {magic!r}")
    layer_count = _unpack1(">I", buf, 4)
    reconstructed: list[KVCache] = []
    for _ in range(layer_count):
        layer_type = _unpack1(">B", buf, 1)
        layer: KVCache = KVCache()
        if layer_type == _LAYER_TYPE_KV:
            keys = _read_array(buf)
            values = _read_array(buf)
            offset = _unpack1(">I", buf, 4)
            object.__setattr__(layer, "keys", keys) if hasattr(type(layer), "__slots__") else setattr(layer, "keys", keys)
            object.__setattr__(layer, "values", values) if hasattr(type(layer), "__slots__") else setattr(layer, "values", values)
            object.__setattr__(layer, "offset", offset) if hasattr(type(layer), "__slots__") else setattr(layer, "offset", offset)
        reconstructed.append(layer)
    prompt_tokens = _read_array(buf)
    return prompt_tokens, cast(KVCacheType, reconstructed)


# ---------------------------------------------------------------------------
# Prefix-length helper (inlined to avoid circular import with cache.py)
# ---------------------------------------------------------------------------


def _prefix_length(prompt: mx.array, cached: mx.array) -> int:
    n = min(int(prompt.shape[0]), int(cached.shape[0]))
    if n == 0:
        return 0
    equal = mx.equal(prompt[:n], cached[:n]).astype(mx.int32)
    prefix_mask = mx.cumprod(equal)
    return int(mx.sum(prefix_mask).item())


# ---------------------------------------------------------------------------
# RemoteCacheTier
# ---------------------------------------------------------------------------


@runtime_checkable
class RemoteCacheTierProtocol(Protocol):
    def store_async(self, entry_id: str, prompt_tokens: mx.array, cache: KVCacheType) -> None: ...
    def fetch(self, prompt_tokens: mx.array, _model: Model) -> tuple[KVCacheType, int] | None: ...
    def remove(self, entry_id: str) -> None: ...


class TieredRemoteCache:
    """Fan-out cache: pushes to ALL tiers on store, fetches from first hit (highest priority first)."""

    def __init__(self, tiers: list[RemoteCacheTier]) -> None:
        self._tiers = tiers  # ordered highest priority first

    def store_async(self, entry_id: str, prompt_tokens: mx.array, cache: KVCacheType) -> None:
        for tier in self._tiers:
            tier.store_async(entry_id, prompt_tokens, cache)

    def fetch(self, prompt_tokens: mx.array, _model: Model) -> tuple[KVCacheType, int] | None:
        for tier in self._tiers:
            result = tier.fetch(prompt_tokens, _model)
            if result is not None:
                return result
        return None

    def remove(self, entry_id: str) -> None:
        for tier in self._tiers:
            tier.remove(entry_id)


class LiveTieredRemoteCache:
    """TieredRemoteCache whose tier list hot-reloads from ~/.exo/remote_cache_urls.

    Allows iOS devices to join/leave mid-session without restarting the runner.
    Background thread polls the file every 30s; atomically swaps tier list on change.
    """

    _POLL_INTERVAL_S: float = 30.0

    def __init__(self, initial_urls: list[str]) -> None:
        self._lock = threading.RLock()
        self._tiers: list[RemoteCacheTier] = [RemoteCacheTier(u) for u in initial_urls]
        self._last_mtime: float = 0.0
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="cache-url-watcher")
        self._thread.start()

    @classmethod
    def from_env(cls) -> "LiveTieredRemoteCache":
        """Build from EXO_REMOTE_CACHE_URLS / EXO_IPAD_CACHE_URL env vars,
        falling back to ~/.exo/remote_cache_urls (written by cache_discovery_daemon.sh).
        This prevents the 30-second blind spot on startup.
        """
        raw = os.environ.get("EXO_REMOTE_CACHE_URLS") or os.environ.get("EXO_IPAD_CACHE_URL") or ""
        urls = [u.strip() for u in raw.split(",") if u.strip()]
        if not urls and _CACHE_URLS_FILE.exists():
            try:
                file_urls = [u.strip() for u in _CACHE_URLS_FILE.read_text().split(",") if u.strip()]
                if file_urls:
                    logger.info("Remote cache: loaded %d tier(s) from %s", len(file_urls), _CACHE_URLS_FILE)
                    urls = file_urls
            except Exception as exc:
                logger.debug("Remote cache URL file read failed: %s", exc)
        return cls(urls)

    def _poll_loop(self) -> None:
        while True:
            time.sleep(self._POLL_INTERVAL_S)
            self._reload_if_changed()

    def _reload_if_changed(self) -> None:
        try:
            if not _CACHE_URLS_FILE.exists():
                return
            mtime = _CACHE_URLS_FILE.stat().st_mtime
            if mtime <= self._last_mtime:
                return
            content = _CACHE_URLS_FILE.read_text().strip()
            urls = [u.strip() for u in content.split(",") if u.strip()]
            new_tiers = [RemoteCacheTier(u) for u in urls]
            with self._lock:
                self._tiers = new_tiers
                self._last_mtime = mtime
            logger.info(f"Remote cache tiers reloaded: {urls}")
        except Exception as exc:
            logger.debug(f"Cache URL file reload failed: {exc}")

    def store_async(self, entry_id: str, prompt_tokens: mx.array, cache: KVCacheType) -> None:
        with self._lock:
            tiers = list(self._tiers)
        for tier in tiers:
            tier.store_async(entry_id, prompt_tokens, cache)

    def fetch(self, prompt_tokens: mx.array, _model: Model) -> tuple[KVCacheType, int] | None:
        with self._lock:
            tiers = list(self._tiers)
        for tier in tiers:
            result = tier.fetch(prompt_tokens, _model)
            if result is not None:
                return result
        return None

    def remove(self, entry_id: str) -> None:
        with self._lock:
            tiers = list(self._tiers)
        for tier in tiers:
            tier.remove(entry_id)

    def tier_status(self) -> list[dict[str, object]]:
        """Return health status of all current tiers (for monitoring/dashboard)."""
        with self._lock:
            tiers = list(self._tiers)
        return [
            {
                "url": tier.base_url,
                "degraded": tier.is_degraded(),
                "consecutive_failures": tier.consecutive_failures,
            }
            for tier in tiers
        ]


class RemoteCacheTier:
    """Client-side facade for the iPad blob store.

    Constructed once per runner session when *EXO_IPAD_CACHE_URL* is set.
    All network I/O is non-blocking (background threads / short timeouts).
    """

    _DEGRADED_THRESHOLD: int = 3
    _DEGRADED_BACKOFF_S: float = 300.0  # 5 minutes

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 5.0,
        index_path: Path = _DEFAULT_INDEX_PATH,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._index_path = index_path
        self._lock = threading.RLock()
        # (available, checked_at_monotonic)
        self._availability_cache: tuple[bool, float] | None = None
        self._availability_ttl_s: float = 30.0
        self._index: list[_IndexEntry] = []
        self._index_loaded: bool = False
        self._consecutive_failures: int = 0
        self._degraded_until: float = 0.0

    # ------------------------------------------------------------------
    # Public accessors (used by LiveTieredRemoteCache.tier_status)
    # ------------------------------------------------------------------

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def is_degraded(self) -> bool:
        return self._is_degraded()

    # ------------------------------------------------------------------
    # Failure tracking / degraded state
    # ------------------------------------------------------------------

    def _is_degraded(self) -> bool:
        with self._lock:
            return time.monotonic() < self._degraded_until

    def _record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._DEGRADED_THRESHOLD:
                self._degraded_until = time.monotonic() + self._DEGRADED_BACKOFF_S
                logger.warning(
                    f"RemoteCacheTier {self._base_url!r} marked degraded for "
                    f"{self._DEGRADED_BACKOFF_S:.0f}s after "
                    f"{self._consecutive_failures} consecutive failures"
                )

    def _record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._degraded_until = 0.0

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the iPad server is reachable.  Cached for 30 s."""
        now = time.monotonic()
        with self._lock:
            if self._availability_cache is not None:
                available, checked_at = self._availability_cache
                if now - checked_at < self._availability_ttl_s:
                    return available
        try:
            req = urllib.request.Request(f"{self._base_url}/stats", method="HEAD")
            with cast(HTTPResponse, urllib.request.urlopen(req, timeout=1.0)):
                pass
            result = True
        except Exception:
            result = False
        with self._lock:
            self._availability_cache = (result, now)
        return result

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _load_index(self) -> list[_IndexEntry]:
        with self._lock:
            if self._index_loaded:
                return self._index
            if self._index_path.exists():
                try:
                    parsed: dict[str, object] = cast(
                        dict[str, object], json.loads(self._index_path.read_text())
                    )
                    entries_raw = cast(list[object], parsed.get("entries", []))
                    parsed_entries: list[_IndexEntry] = []
                    for raw_entry in entries_raw:
                        if not isinstance(raw_entry, dict):
                            continue
                        entry_dict: dict[str, object] = cast(dict[str, object], raw_entry)
                        parsed_entries.append(
                            _IndexEntry(
                                id=str(entry_dict.get("id", "")),
                                token_count=int(entry_dict.get("token_count") or 0),  # type: ignore[arg-type]
                                stored_at=float(entry_dict.get("stored_at") or 0.0),  # type: ignore[arg-type]
                                prompt_tokens_hex=str(entry_dict.get("prompt_tokens_hex", "")),
                            )
                        )
                    self._index = parsed_entries
                except Exception:
                    self._index = []
            else:
                self._index = []
            self._index_loaded = True
            return self._index

    def _save_index(self) -> None:
        """Persist index to disk.  Must be called under self._lock."""
        try:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            serialised = [
                {
                    "id": e.id,
                    "token_count": e.token_count,
                    "stored_at": e.stored_at,
                    "prompt_tokens_hex": e.prompt_tokens_hex,
                }
                for e in self._index
            ]
            self._index_path.write_text(
                json.dumps({"entries": serialised}, indent=2)
            )
        except Exception as exc:
            logger.debug(f"Remote cache index save failed: {exc}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store_async(
        self,
        entry_id: str,
        prompt_tokens: mx.array,
        cache: KVCacheType,
    ) -> None:
        """Serialize cache and push to iPad in a background daemon thread.

        Never raises — failure is silently logged at DEBUG level.
        """

        def _do_store() -> None:
            try:
                data = _serialize(prompt_tokens, cache)
                req = urllib.request.Request(
                    f"{self._base_url}/blob/{entry_id}",
                    data=data,
                    method="PUT",
                )
                req.add_header("Content-Type", "application/octet-stream")
                with cast(HTTPResponse, urllib.request.urlopen(req, timeout=self._timeout_s)):
                    pass

                token_count = int(prompt_tokens.shape[0])
                token_hex = np.array(prompt_tokens).astype(np.int32).tobytes().hex()
                entry = _IndexEntry(
                    id=entry_id,
                    token_count=token_count,
                    stored_at=time.time(),
                    prompt_tokens_hex=token_hex,
                )
                with self._lock:
                    self._load_index()
                    self._index.append(entry)
                    self._save_index()

                self._record_success()
                logger.info(
                    f"Remote KV cache stored: {entry_id} "
                    f"({len(data):,} bytes, {token_count} tokens)"
                )
            except Exception as exc:
                logger.debug(f"Remote KV cache store failed (best-effort): {exc}")
                self._record_failure()

        thread = threading.Thread(target=_do_store, daemon=True)
        thread.start()

    def fetch(
        self,
        prompt_tokens: mx.array,
        _model: Model,
    ) -> tuple[KVCacheType, int] | None:
        """Find the best prefix match in the remote index and fetch the blob.

        Returns (cache, matched_token_count) or None on miss/timeout.
        """
        if self._is_degraded():
            logger.debug(f"Tier {self._base_url!r} is degraded — skipping fetch")
            return None

        entries = self._load_index()
        if not entries:
            return None

        best_entry: _IndexEntry | None = None
        best_length: int = 0
        min_useful_tokens: int = 32

        for entry in entries:
            if not entry.prompt_tokens_hex:
                continue
            try:
                token_bytes = bytes.fromhex(entry.prompt_tokens_hex)
                cached_tokens = mx.array(
                    np.frombuffer(token_bytes, dtype=np.int32)
                )
                length = _prefix_length(prompt_tokens, cached_tokens)
                if length > best_length:
                    best_length = length
                    best_entry = entry
            except Exception:
                continue

        if best_entry is None or best_length < min_useful_tokens:
            return None

        try:
            req = urllib.request.Request(
                f"{self._base_url}/blob/{best_entry.id}", method="GET"
            )
            with cast(HTTPResponse, urllib.request.urlopen(req, timeout=self._timeout_s)) as resp:
                data: bytes = resp.read()
            _prompt_tokens, cache = _deserialize(data)
            self._record_success()
            return cache, best_length
        except Exception as exc:
            logger.debug(f"Remote KV cache fetch failed: {exc}")
            self._record_failure()
            return None

    def remove(self, entry_id: str) -> None:
        """Delete a blob from the iPad and remove it from the local index."""
        try:
            req = urllib.request.Request(
                f"{self._base_url}/blob/{entry_id}", method="DELETE"
            )
            with cast(HTTPResponse, urllib.request.urlopen(req, timeout=self._timeout_s)):
                pass
        except Exception:
            pass
        with self._lock:
            self._load_index()
            self._index = [e for e in self._index if e.id != entry_id]
            self._save_index()


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


class _IndexEntry:
    """Typed representation of a remote cache index record."""

    __slots__ = ("id", "token_count", "stored_at", "prompt_tokens_hex")

    def __init__(
        self,
        id: str,  # noqa: A002
        token_count: int,
        stored_at: float,
        prompt_tokens_hex: str,
    ) -> None:
        self.id = id
        self.token_count = token_count
        self.stored_at = stored_at
        self.prompt_tokens_hex = prompt_tokens_hex
