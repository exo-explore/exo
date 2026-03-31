"""Lightweight HTTP file server for peer-to-peer model transfer.

Serves model files from EXO_MODELS_DIRS so that peer nodes can download
model weights over the local network (e.g. Thunderbolt) instead of from
HuggingFace.

Uses sendfile() for zero-copy transfers — data goes straight from disk
to socket without passing through Python.

Listens on EXO_FILE_SERVER_PORT (default 52416).
"""

import asyncio
from pathlib import Path

from aiohttp import web
from loguru import logger

from exo.shared.constants import (
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
    Uses FileResponse for zero-copy sendfile() with Range header support.
    """
    full_match = request.match_info["path"]
    parts = full_match.split("/", 2)
    if len(parts) < 3:
        raise web.HTTPNotFound(text=f"Invalid path: {full_match}")

    model_id = f"{parts[0]}/{parts[1]}"
    file_path = parts[2]

    normalized = model_id.replace("/", "--")

    # Search all model directories for the file
    for model_dir in _all_model_dirs():
        candidate = Path(model_dir) / normalized / file_path
        try:
            candidate = candidate.resolve()
            if candidate.is_file() and candidate.is_relative_to(model_dir):
                return web.FileResponse(candidate)
        except (ValueError, OSError):
            continue

    raise web.HTTPNotFound(text=f"File not found: {model_id}/{file_path}")


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
