"""Lightweight HTTP file server for peer-to-peer model downloads.

Each exo node runs a PeerFileServer that serves model files from the local
cache directory. When one node finishes downloading a model from HuggingFace,
other nodes on the same LAN can fetch it directly over HTTP instead of
re-downloading from the internet.

Supports serving in-progress downloads via .partial.meta files that track
how many bytes have been safely flushed to disk.
"""

import json
from pathlib import Path

import aiofiles
import aiofiles.os as aios
from aiohttp import web
from loguru import logger


class PeerFileServer:
    """HTTP server that exposes local model files for peer download."""

    def __init__(self, host: str, port: int, models_dir: Path) -> None:
        self.host = host
        self.port = port
        self.models_dir = models_dir
        self._app = web.Application()
        self._app.router.add_get("/status/{model_id}", self._handle_status)
        self._app.router.add_get(
            "/files/{model_id}/{file_path:.+}", self._handle_file
        )
        self._app.router.add_get("/health", self._handle_health)
        self._runner: web.AppRunner | None = None

    async def run(self) -> None:
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        logger.info(f"PeerFileServer listening on {self.host}:{self.port}")

    async def shutdown(self) -> None:
        if self._runner:
            await self._runner.cleanup()

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def _handle_status(self, request: web.Request) -> web.Response:
        """Return status of all files for a model (complete + in-progress)."""
        model_id = request.match_info["model_id"]
        model_dir = self.models_dir / model_id

        if not await aios.path.exists(model_dir):
            return web.json_response({"files": []})

        files = []
        for item in model_dir.iterdir():
            if item.is_dir() or item.name.endswith(".partial.meta"):
                continue

            if item.name.endswith(".partial"):
                # In-progress file - read meta for safe bytes
                meta = await _read_partial_meta(item)
                if meta:
                    files.append(
                        {
                            "path": item.name.removesuffix(".partial"),
                            "size": meta.get("total", 0),
                            "complete": False,
                            "safe_bytes": meta.get("safe_bytes", 0),
                        }
                    )
            else:
                # Complete file
                stat = await aios.stat(item)
                files.append(
                    {
                        "path": item.name,
                        "size": stat.st_size,
                        "complete": True,
                        "safe_bytes": stat.st_size,
                    }
                )

        return web.json_response({"files": files})

    async def _handle_file(self, request: web.Request) -> web.StreamResponse:
        """Serve a model file with Range request support.

        For complete files: standard HTTP file serving.
        For .partial files: serves only the safe byte range (flushed to disk).
        """
        model_id = request.match_info["model_id"]
        file_path = request.match_info["file_path"]

        model_dir = self.models_dir / model_id
        complete_path = model_dir / file_path
        partial_path = model_dir / f"{file_path}.partial"

        # Determine which file to serve and its safe size
        if await aios.path.exists(complete_path):
            serve_path = complete_path
            file_size = (await aios.stat(complete_path)).st_size
            safe_bytes = file_size
            is_complete = True
        elif await aios.path.exists(partial_path):
            meta = await _read_partial_meta(partial_path)
            if not meta or meta.get("safe_bytes", 0) == 0:
                return web.Response(status=404, text="File not available yet")
            serve_path = partial_path
            file_size = meta.get("total", 0)
            safe_bytes = meta["safe_bytes"]
            is_complete = False
        else:
            return web.Response(status=404, text="File not found")

        # Parse Range header
        range_header = request.headers.get("Range")
        start = 0
        if range_header:
            try:
                range_spec = range_header.replace("bytes=", "")
                start = int(range_spec.split("-")[0])
            except (ValueError, IndexError):
                return web.Response(status=416, text="Invalid range")

        if start >= safe_bytes:
            return web.Response(status=416, text="Range not satisfiable")

        end = safe_bytes  # Serve up to safe boundary only
        content_length = end - start

        response = web.StreamResponse(
            status=206 if start > 0 else 200,
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Length": str(content_length),
                "Accept-Ranges": "bytes",
                "Content-Range": f"bytes {start}-{end - 1}/{file_size}",
                "X-Exo-Safe-Bytes": str(safe_bytes),
                "X-Exo-Total-Size": str(file_size),
                "X-Exo-Complete": "true" if is_complete else "false",
            },
        )
        await response.prepare(request)

        chunk_size = 8 * 1024 * 1024  # 8MB chunks matching HF download
        async with aiofiles.open(serve_path, "rb") as f:
            await f.seek(start)
            remaining = content_length
            while remaining > 0:
                to_read = min(chunk_size, remaining)
                chunk = await f.read(to_read)
                if not chunk:
                    break
                await response.write(chunk)
                remaining -= len(chunk)

        await response.write_eof()
        return response


async def _read_partial_meta(partial_path: Path) -> dict | None:
    """Read the .partial.meta companion file for a .partial download."""
    meta_path = Path(f"{partial_path}.meta")
    if not await aios.path.exists(meta_path):
        return None
    try:
        async with aiofiles.open(meta_path, "r") as f:
            return json.loads(await f.read())
    except (json.JSONDecodeError, OSError):
        return None
