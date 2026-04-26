"""Lightweight HTTP file server for peer-to-peer model transfer.

Serves model files from EXO_MODELS_DIRS so that peer nodes can download
model weights over the local network (e.g. Thunderbolt) instead of from
HuggingFace.

Listens on EXO_FILE_SERVER_PORT (default 52416). Concurrent serves are
capped at EXO_FILE_SERVER_MAX_CONCURRENCY (default 64); requests beyond
the cap get 503 with Retry-After: 1 so the receiver's retry loop picks
them up. When a sidecar ``<file>.sha256`` exists, its contents are
returned in an ``X-File-SHA256`` response header so the receiver can
verify the bytes after download.
"""

import asyncio
from pathlib import Path

from aiohttp import web
from loguru import logger

from exo.shared.constants import (
    EXO_FILE_SERVER_BIND_HOST,
    EXO_FILE_SERVER_MAX_CONCURRENCY,
    EXO_FILE_SERVER_PORT,
    EXO_MODELS_DIRS,
    EXO_MODELS_READ_ONLY_DIRS,
)

# Read buffer size when streaming a file out. Large enough to amortize
# per-chunk overhead, small enough that N concurrent serves don't crush
# RAM on a heavily-loaded node.
_STREAM_CHUNK_SIZE = 1 * 1024 * 1024  # 1 MiB

# Lazily-initialized so we share one semaphore across the lifetime of a
# single ``run_file_server()`` call but always create a fresh one in tests.
_serve_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _serve_semaphore
    if _serve_semaphore is None:
        _serve_semaphore = asyncio.Semaphore(EXO_FILE_SERVER_MAX_CONCURRENCY)
    return _serve_semaphore


def _all_model_dirs() -> list[str]:
    """Return all model directories (writable + read-only) as resolved strings."""
    seen: set[str] = set()
    dirs: list[str] = []
    for d in (*EXO_MODELS_DIRS, *EXO_MODELS_READ_ONLY_DIRS):
        resolved = str(d.resolve())
        if resolved not in seen:
            seen.add(resolved)
            dirs.append(resolved)
    return dirs


def _resolve_safe(model_dir: str, normalized: str, file_path: str) -> Path | None:
    """Resolve ``<model_dir>/<normalized>/<file_path>`` and return it only if
    the result is still inside the *specific* normalized model subdirectory.

    This rejects path-traversal attempts that escape upward (``..``) into a
    sibling model's directory, even though that sibling is technically inside
    ``model_dir``. Returns None on any traversal escape, missing file, or OS
    error.
    """
    model_root = Path(model_dir).resolve()
    expected_subdir = (model_root / normalized).resolve()
    candidate = (model_root / normalized / file_path).resolve()
    try:
        if not candidate.is_relative_to(expected_subdir):
            return None
        if not candidate.is_file():
            return None
    except (ValueError, OSError):
        return None
    return candidate


def _read_sidecar_sha256(file_path: Path) -> str | None:
    """Return the contents of ``<file>.sha256`` (a SHA256 hex digest) if the
    sidecar exists alongside ``file_path``. Used to populate the
    ``X-File-SHA256`` response header so peers can verify downloaded bytes
    against the same hash the source node verified at HF-download time.

    Sidecar absent → returns ``None`` (no header is sent; receiver downloads
    without verification, logging a debug message). Never computes the hash
    on demand — that would block the request handler for tens of seconds on
    a multi-GB safetensors file.
    """
    sidecar = file_path.with_suffix(file_path.suffix + ".sha256")
    try:
        if not sidecar.is_file():
            return None
        text = sidecar.read_text().strip()
    except OSError:
        return None
    # Defensive: a SHA256 hex digest is exactly 64 chars of [0-9a-f]. Reject
    # anything that doesn't match that shape so we never echo random bytes
    # back as a hash header.
    if len(text) != 64 or not all(c in "0123456789abcdef" for c in text.lower()):
        return None
    return text


def _parse_range_start(range_header: str | None) -> int | None:
    """Parse a ``Range: bytes=<n>-`` header and return the start offset, or
    ``None`` if the header is absent / malformed / unsupported.

    Only prefix-form (``bytes=N-``) is supported; suffix-form (``bytes=-N``)
    and multi-range are not. Malformed headers (non-numeric, negative,
    non-``bytes=`` unit) do not raise — callers treat ``None`` as "serve
    the whole file from offset 0".
    """
    if not range_header or not range_header.startswith("bytes="):
        return None
    spec = range_header[len("bytes="):]
    if "," in spec:  # multi-range — we don't support
        return None
    range_parts = spec.split("-", 1)
    if len(range_parts) != 2 or not range_parts[0]:
        return None
    try:
        start = int(range_parts[0])
    except ValueError:
        return None
    if start < 0:
        return None
    return start


async def _handle_model_file(request: web.Request) -> web.StreamResponse:
    """Serve a model file: GET /{org}/{model}/{file_path}

    ``model_id`` is exactly two URL segments (e.g.
    ``mlx-community/Qwen3.5-397B-A17B-nvfp4``); on disk it's normalized with
    ``--`` (``mlx-community--Qwen3.5-397B-A17B-nvfp4``). Supports HTTP Range
    requests in prefix form (``bytes=N-``) for resumable downloads.

    Concurrency cap: at most ``EXO_FILE_SERVER_MAX_CONCURRENCY`` of these run
    at once; further requests get 503 with ``Retry-After: 1`` and the
    receiver's outer retry loop re-issues the curl, auto-resuming via
    ``-C -``.

    Security boundary: the resolved file path must live inside the requested
    model's specific normalized subdirectory. ``..`` traversal that escapes
    that subdirectory — even into a sibling model under the same ``model_dir``
    root — is rejected with 404. Error responses do not echo the request path
    back to the client.
    """
    semaphore = _get_semaphore()
    if semaphore.locked():
        # Already at the cap — fail fast rather than queueing requests
        # indefinitely (queued requests still hold an aiohttp connection
        # and amplify memory pressure).
        raise web.HTTPServiceUnavailable(
            text="Server busy", headers={"Retry-After": "1"}
        )

    async with semaphore:
        return await _serve(request)


async def _serve(request: web.Request) -> web.StreamResponse:
    full_match = request.match_info["path"]
    parts = full_match.split("/", 2)
    if len(parts) < 3:
        raise web.HTTPNotFound(text="Not found")

    model_id = f"{parts[0]}/{parts[1]}"
    file_path = parts[2]
    normalized = model_id.replace("/", "--")

    full_path: Path | None = None
    for model_dir in _all_model_dirs():
        full_path = _resolve_safe(model_dir, normalized, file_path)
        if full_path is not None:
            break

    if full_path is None:
        raise web.HTTPNotFound(text="Not found")

    file_size = full_path.stat().st_size

    start = _parse_range_start(request.headers.get("Range")) or 0

    if start >= file_size and start != 0:
        raise web.HTTPRequestRangeNotSatisfiable(
            headers={"Content-Range": f"bytes */{file_size}"}
        )

    remaining = file_size - start
    status = 206 if start > 0 else 200

    headers: dict[str, str] = {
        "Content-Length": str(remaining),
        "Content-Type": "application/octet-stream",
        "Accept-Ranges": "bytes",
    }
    if start > 0:
        headers["Content-Range"] = f"bytes {start}-{file_size - 1}/{file_size}"
    sha256 = _read_sidecar_sha256(full_path)
    if sha256:
        headers["X-File-SHA256"] = sha256

    response = web.StreamResponse(status=status, headers=headers)
    await response.prepare(request)

    with open(full_path, "rb") as f:
        if start > 0:
            f.seek(start)
        while True:
            chunk = f.read(_STREAM_CHUNK_SIZE)
            if not chunk:
                break
            await response.write(chunk)

    await response.write_eof()
    return response


async def run_file_server() -> None:
    """Start the model file server on EXO_FILE_SERVER_PORT."""
    app = web.Application()
    app.router.add_get("/{path:.*}", _handle_model_file)

    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, EXO_FILE_SERVER_BIND_HOST, EXO_FILE_SERVER_PORT)
    try:
        await site.start()
        logger.info(
            f"Model file server listening on "
            f"{EXO_FILE_SERVER_BIND_HOST}:{EXO_FILE_SERVER_PORT} "
            f"(max_concurrency={EXO_FILE_SERVER_MAX_CONCURRENCY})"
        )
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()
