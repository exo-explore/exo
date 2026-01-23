from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator, cast

import aiofiles
import aiofiles.os as aios
import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from hypercorn.asyncio import serve  # pyright: ignore[reportUnknownVariableType]
from hypercorn.config import Config
from hypercorn.typing import ASGIFramework
from loguru import logger
from pydantic import BaseModel, ConfigDict, TypeAdapter

from exo.download.download_utils import calc_hash
from exo.shared.constants import EXO_MODELS_DIR
from exo.shared.types.common import ModelId
from exo.shared.types.worker.downloads import FileListEntry


class ModelStoreFileMetadata(BaseModel):
    etag: str
    size: int

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _normalize_model_id(model_id: str) -> str:
    return ModelId(model_id).normalize()


def _model_dir(model_id: str) -> Path:
    return EXO_MODELS_DIR / _normalize_model_id(model_id)


def _safe_relative_path(rel_path: str) -> Path:
    p = Path(rel_path)
    if p.is_absolute() or ".." in p.parts:
        raise HTTPException(status_code=400, detail="Invalid path")
    return p


def _metadata_root_dir(model_dir: Path) -> Path:
    return model_dir / ".exo" / "download_metadata"


def _metadata_path(model_dir: Path, rel_path: str) -> Path:
    safe_rel = _safe_relative_path(rel_path)
    return _metadata_root_dir(model_dir) / f"{safe_rel}.json"


async def _read_metadata(model_dir: Path, rel_path: str) -> ModelStoreFileMetadata | None:
    meta_path = _metadata_path(model_dir, rel_path)
    if await aios.path.exists(meta_path):
        async with aiofiles.open(meta_path, "r") as f:
            return ModelStoreFileMetadata.model_validate_json(await f.read())
    return None


async def _write_metadata(
    model_dir: Path, rel_path: str, metadata: ModelStoreFileMetadata
) -> None:
    meta_path = _metadata_path(model_dir, rel_path)
    await aios.makedirs(meta_path.parent, exist_ok=True)
    async with aiofiles.open(meta_path, "w") as f:
        await f.write(metadata.model_dump_json())


def _is_blocked_model_store_path(rel_path: str) -> bool:
    safe_rel = _safe_relative_path(rel_path)
    if safe_rel.parts and safe_rel.parts[0].startswith("."):
        return True
    return rel_path.endswith(".partial")


async def _get_or_create_etag(model_dir: Path, rel_path: str, size: int) -> str:
    existing = await _read_metadata(model_dir, rel_path)
    if existing is not None and existing.size == size and existing.etag:
        return existing.etag

    # Fallback when no metadata exists (e.g. models downloaded before this feature).
    # We use SHA-256 of file contents to keep client-side integrity checking intact.
    file_path = model_dir / _safe_relative_path(rel_path)
    etag = await calc_hash(file_path, hash_type="sha256")
    await _write_metadata(model_dir, rel_path, ModelStoreFileMetadata(etag=etag, size=size))
    return etag


def _parse_range_header(range_header: str, size: int) -> tuple[int, int]:
    # We only support a single bytes-range.
    if not range_header.startswith("bytes="):
        raise HTTPException(status_code=416, detail="Invalid Range header")
    raw = range_header.removeprefix("bytes=").strip()
    if "," in raw:
        raise HTTPException(status_code=416, detail="Multiple ranges not supported")

    start_s, end_s = (raw.split("-", 1) + [""])[:2]
    if not start_s:
        raise HTTPException(status_code=416, detail="Suffix ranges not supported")

    try:
        start = int(start_s)
    except ValueError as exc:
        raise HTTPException(status_code=416, detail="Invalid Range header") from exc

    end: int
    if end_s:
        try:
            end = int(end_s)
        except ValueError as exc:
            raise HTTPException(status_code=416, detail="Invalid Range header") from exc
    else:
        end = size - 1

    if start < 0 or start >= size:
        raise HTTPException(status_code=416, detail="Range start out of bounds")
    if end < start:
        raise HTTPException(status_code=416, detail="Invalid Range header")
    end = min(end, size - 1)
    return start, end


async def _stream_file_range(
    path: Path,
    *,
    start: int,
    end_inclusive: int,
    chunk_size: int = 8 * 1024 * 1024,
) -> AsyncIterator[bytes]:
    remaining = end_inclusive - start + 1
    async with aiofiles.open(path, "rb") as f:
        await f.seek(start)
        while remaining > 0:
            chunk = await f.read(min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


async def _walk_model_dir_for_files(model_dir: Path) -> list[FileListEntry]:
    def _walk_sync() -> list[FileListEntry]:
        out: list[FileListEntry] = []
        for p in model_dir.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(model_dir).as_posix()
            if _is_blocked_model_store_path(rel):
                continue
            out.append(FileListEntry(type="file", path=rel, size=p.stat().st_size))
        return out

    return await asyncio.to_thread(_walk_sync)


class ModelStoreServer:
    """
    A lightweight HTTP server that exposes a HuggingFace-like subset of endpoints
    backed by a local on-disk model cache.

    It is intended for intra-cluster use so that only one node (the master) needs
    to download from Hugging Face, while other nodes fetch model files over the
    local network.
    """

    def __init__(self, *, port: int) -> None:
        self.port = port
        self.app = FastAPI()
        self._setup_routes()

    def _setup_routes(self) -> None:
        self.app.get("/api/models/{model_id:path}/tree/{revision}")(self.tree)
        self.app.get("/api/models/{model_id:path}/tree/{revision}/{subpath:path}")(
            self.tree
        )
        self.app.head("/{model_id:path}/resolve/{revision}/{file_path:path}")(
            self.head_resolve
        )
        self.app.get("/{model_id:path}/resolve/{revision}/{file_path:path}")(
            self.get_resolve
        )

    async def tree(
        self, model_id: str, revision: str, subpath: str = ""
    ) -> list[FileListEntry]:
        normalized_model_id = _normalize_model_id(model_id)
        cache_file = (
            EXO_MODELS_DIR
            / "caches"
            / normalized_model_id
            / f"{normalized_model_id}--{revision}--file_list.json"
        )
        if await aios.path.exists(cache_file):
            async with aiofiles.open(cache_file, "r") as f:
                files = TypeAdapter(list[FileListEntry]).validate_json(await f.read())
        else:
            model_dir = _model_dir(model_id)
            if not await aios.path.exists(model_dir):
                raise HTTPException(status_code=404, detail="Model not found")
            files = await _walk_model_dir_for_files(model_dir)

        if subpath:
            prefix = subpath.rstrip("/") + "/"
            files = [f for f in files if f.path.startswith(prefix)]

        return files

    async def head_resolve(self, model_id: str, revision: str, file_path: str) -> Response:
        _ = revision  # revision is currently informational only
        if _is_blocked_model_store_path(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        model_dir = _model_dir(model_id)
        local_path = model_dir / _safe_relative_path(file_path)
        if not await aios.path.exists(local_path):
            raise HTTPException(status_code=404, detail="File not found")

        size = (await aios.stat(local_path)).st_size
        etag = await _get_or_create_etag(model_dir, file_path, size)

        return Response(
            status_code=200,
            headers={
                "Content-Length": str(size),
                "ETag": etag,
                "Accept-Ranges": "bytes",
            },
        )

    async def get_resolve(
        self, request: Request, model_id: str, revision: str, file_path: str
    ) -> StreamingResponse:
        _ = revision  # revision is currently informational only
        if _is_blocked_model_store_path(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        model_dir = _model_dir(model_id)
        local_path = model_dir / _safe_relative_path(file_path)
        if not await aios.path.exists(local_path):
            raise HTTPException(status_code=404, detail="File not found")

        size = (await aios.stat(local_path)).st_size
        etag = await _get_or_create_etag(model_dir, file_path, size)

        range_header = request.headers.get("range")
        if range_header:
            start, end = _parse_range_header(range_header, size)
            content_length = end - start + 1
            headers = {
                "Content-Length": str(content_length),
                "Content-Range": f"bytes {start}-{end}/{size}",
                "ETag": etag,
                "Accept-Ranges": "bytes",
            }
            return StreamingResponse(
                _stream_file_range(local_path, start=start, end_inclusive=end),
                status_code=206,
                headers=headers,
                media_type="application/octet-stream",
            )

        headers = {
            "Content-Length": str(size),
            "ETag": etag,
            "Accept-Ranges": "bytes",
        }
        return StreamingResponse(
            _stream_file_range(local_path, start=0, end_inclusive=size - 1),
            status_code=200,
            headers=headers,
            media_type="application/octet-stream",
        )

    async def run(self) -> None:
        cfg = Config()
        cfg.bind = f"0.0.0.0:{self.port}"
        cfg.accesslog = None
        cfg.errorlog = "-"

        logger.info(f"Starting model store server on :{self.port}")
        await serve(
            cast(ASGIFramework, self.app),
            cfg,
            shutdown_trigger=lambda: anyio.sleep_forever(),
        )

