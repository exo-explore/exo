"""Lightweight HTTP file server for peer-to-peer model transfer.

Serves model files from EXO_MODELS_DIRS so that peer nodes can download
model weights over the local network (e.g. Thunderbolt) instead of from
HuggingFace.

Listens on EXO_FILE_SERVER_PORT (default 52416).
"""

import asyncio

from aiohttp import web
from loguru import logger

from exo.shared.constants import (
    EXO_DEFAULT_MODELS_DIR,
    EXO_FILE_SERVER_PORT,
    EXO_MODELS_DIRS,
    EXO_MODELS_READ_ONLY_DIRS,
)


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


async def _handle_model_file(request: web.Request) -> web.StreamResponse:
    """Serve a model file: GET /{org}/{model}/{file_path}

    The model_id is always two segments (e.g. mlx-community/Qwen3.5-397B-A17B-nvfp4).
    On disk it's normalized with "--" (e.g. mlx-community--Qwen3.5-397B-A17B-nvfp4).
    Supports Range headers for resumable downloads.
    """
    full_match = request.match_info["path"]
    parts = full_match.split("/", 2)
    if len(parts) < 3:
        raise web.HTTPNotFound(text=f"Invalid path: {full_match}")

    model_id = f"{parts[0]}/{parts[1]}"
    file_path = parts[2]

    normalized = model_id.replace("/", "--")

    # Search all model directories for the file
    full_path = None
    for model_dir in _all_model_dirs():
        candidate = (EXO_DEFAULT_MODELS_DIR.parent / model_dir / normalized / file_path)
        # Use the directory directly since _all_model_dirs returns resolved paths
        from pathlib import Path

        candidate = Path(model_dir) / normalized / file_path
        try:
            candidate = candidate.resolve()
            if candidate.is_file() and candidate.is_relative_to(model_dir):
                full_path = candidate
                break
        except (ValueError, OSError):
            continue

    if full_path is None:
        raise web.HTTPNotFound(text=f"File not found: {model_id}/{file_path}")

    file_size = full_path.stat().st_size

    # Parse Range header for resume support
    range_header = request.headers.get("Range")
    start = 0
    if range_header and range_header.startswith("bytes="):
        range_spec = range_header[6:]
        range_parts = range_spec.split("-")
        if range_parts[0]:
            start = int(range_parts[0])

    if start >= file_size:
        raise web.HTTPRequestRangeNotSatisfiable(
            headers={"Content-Range": f"bytes */{file_size}"}
        )

    remaining = file_size - start
    status = 206 if start > 0 else 200

    response = web.StreamResponse(
        status=status,
        headers={
            "Content-Length": str(remaining),
            "Content-Type": "application/octet-stream",
            "Accept-Ranges": "bytes",
        },
    )

    if start > 0:
        response.headers["Content-Range"] = f"bytes {start}-{file_size - 1}/{file_size}"

    await response.prepare(request)

    chunk_size = 8 * 1024 * 1024  # 8MB chunks
    with open(full_path, "rb") as f:
        if start > 0:
            f.seek(start)
        while True:
            chunk = f.read(chunk_size)
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
    site = web.TCPSite(runner, "0.0.0.0", EXO_FILE_SERVER_PORT)
    try:
        await site.start()
        logger.info(f"Model file server listening on 0.0.0.0:{EXO_FILE_SERVER_PORT}")
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()
