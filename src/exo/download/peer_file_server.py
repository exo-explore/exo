"""Lightweight HTTP file server for peer-to-peer model downloads.

Each exo node runs a PeerFileServer that serves model files from its local
caches. When one node finishes downloading a model from HuggingFace, other
nodes on the same LAN can fetch it directly over HTTP instead of
re-downloading from the internet.

Supports serving in-progress downloads via .partial.meta files that track
how many bytes have been safely flushed to disk.

The server is given the *full* set of directories the local node may store
models in (the writable ``EXO_MODELS_DIRS`` plus any read-only mounts under
``EXO_MODELS_READ_ONLY_DIRS``) so that peers can fetch any locally-resident
model regardless of which directory the downloader picked. Restricting the
server to a single hard-coded directory would silently disable the peer
download path whenever ``select_download_dir_for_shard`` placed the model
in a non-default directory (custom path, low-disk fallback, or a read-only
mount).
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias, cast

import aiofiles
import aiofiles.os as aios
import anyio
from aiohttp import web
from loguru import logger

PartialMeta: TypeAlias = dict[str, int | str]


class PeerFileServer:
    """HTTP server that exposes local model files for peer download."""

    def __init__(self, host: str, port: int, models_dirs: Sequence[Path]) -> None:
        if not models_dirs:
            raise ValueError("PeerFileServer requires at least one models directory")
        self.host = host
        self.port = port
        # Preserve caller order so callers can prefer writable dirs over
        # read-only dirs without us re-sorting them.
        self.models_dirs: tuple[Path, ...] = tuple(models_dirs)
        self._app = web.Application()
        self._app.router.add_get("/status/{model_id}", self._handle_status)
        self._app.router.add_get("/files/{model_id}/{file_path:.+}", self._handle_file)
        self._app.router.add_get("/health", self._handle_health)
        self._runner: web.AppRunner | None = None

    async def run(self) -> None:
        """Start the peer file server and keep the task alive until cancelled.

        Codex P2 (PR #16 round-(N+10), peer_file_server.py:56): pre-fix
        ``run()`` returned immediately after ``site.start()``, so the
        task spawned by ``Node.run()`` (``tg.start_soon(self.peer_file_server.run)``)
        completed on the first event-loop tick and the parent task
        group considered the server "done". When the node was
        cancelled, there was no live coroutine for the task group to
        cancel, so the aiohttp listener kept its TCP socket open
        until process exit. That manifested as
        ``OSError: [Errno 48] address already in use`` whenever a
        node was stopped/restarted in the same process (commonly in
        tests, embedded runs, or systemd-style restart loops).

        The fix keeps the coroutine alive via ``anyio.sleep_forever``
        and runs ``self._runner.cleanup()`` in a shielded ``finally``
        block on cancellation, so the listener is reliably released
        before the task group considers the server torn down.
        """
        runner = web.AppRunner(self._app)
        self._runner = runner
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        logger.info(f"PeerFileServer listening on {self.host}:{self.port}")
        try:
            await anyio.sleep_forever()
        finally:
            # Shield cleanup from the cancellation that woke us so
            # ``aiohttp`` can drain in-flight responses and release
            # the listening socket before this task is considered
            # complete. Without the shield the cleanup itself is
            # cancelled immediately, which leaves the socket bound
            # and reproduces the original ``EADDRINUSE`` symptom.
            with anyio.CancelScope(shield=True):
                # Re-read self._runner so an external ``shutdown()``
                # call (e.g. from a separate code path) doesn't drive
                # cleanup twice. ``cast`` because the type-checker has
                # narrowed ``self._runner`` to ``AppRunner`` from the
                # assignment above; an external mutation could still
                # have set it to ``None``.
                live_runner = cast(web.AppRunner | None, self._runner)
                if live_runner is not None:
                    self._runner = None
                    await live_runner.cleanup()
                logger.info(f"PeerFileServer on {self.host}:{self.port} stopped")

    async def shutdown(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            self._runner = None

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def _handle_status(self, request: web.Request) -> web.Response:
        """Return status of all files for a model (complete + in-progress).

        Codex P2 (PR #16 round-(N+9), peer_file_server.py:201): when
        a model's contents are split across multiple configured
        roots (e.g. an earlier writable cache holds a partial copy
        and a later read-only mount holds the full canonical copy),
        report the union across every root that contains the model.
        For files that appear in more than one root we keep the
        most-complete entry (complete > larger partial) so peers see
        the true 'most progressed' version of the file. The earlier
        single-root behaviour caused the peer downloader to
        miss-report missing files and silently fall back to
        HuggingFace even when this node had a complete copy
        elsewhere on disk.
        """
        model_id = request.match_info["model_id"]
        model_dirs = await self._locate_all_model_dirs(model_id)
        if not model_dirs:
            return web.json_response({"files": []})

        # path -> entry; complete files dominate partials; larger
        # partials dominate smaller ones when no complete is found.
        merged: dict[str, dict[str, object]] = {}

        def merge(entry: dict[str, object]) -> None:
            path = cast(str, entry["path"])
            existing = merged.get(path)
            if existing is None:
                merged[path] = entry
                return
            existing_complete = bool(existing["complete"])
            new_complete = bool(entry["complete"])
            new_partial_is_more_complete = (
                not new_complete
                and not existing_complete
                and cast(int, entry["safe_bytes"]) > cast(int, existing["safe_bytes"])
            )
            if (new_complete and not existing_complete) or (
                new_partial_is_more_complete
            ):
                merged[path] = entry
            # complete-vs-complete: keep the first (sizes equal by
            # construction, callers only need one entry).

        for model_dir in model_dirs:
            for item in model_dir.rglob("*"):
                relative_path = item.relative_to(model_dir).as_posix()
                if item.is_dir() or relative_path.endswith(".partial.meta"):
                    continue
                if _resolve_child(model_dir, relative_path) is None:
                    continue

                if relative_path.endswith(".partial"):
                    meta = await _read_partial_meta(item)
                    if meta:
                        total = _meta_int(meta, "total")
                        safe_bytes = _meta_int(meta, "safe_bytes")
                        merge(
                            {
                                "path": relative_path.removesuffix(".partial"),
                                "size": total,
                                "complete": False,
                                "safe_bytes": safe_bytes,
                            }
                        )
                else:
                    stat = await aios.stat(item)
                    merge(
                        {
                            "path": relative_path,
                            "size": stat.st_size,
                            "complete": True,
                            "safe_bytes": stat.st_size,
                        }
                    )

        return web.json_response({"files": list(merged.values())})

    async def _handle_file(self, request: web.Request) -> web.StreamResponse:
        """Serve a model file with Range request support.

        For complete files: standard HTTP file serving.
        For .partial files: serves only the safe byte range (flushed to disk).

        Codex P2 (PR #16 round-(N+9), peer_file_server.py:201): when
        a model's contents are split across multiple roots, prefer
        the root holding a *complete* copy of the requested file
        over the first root that merely contains the model
        directory. Fall back to a partial copy only if no root has
        the file complete. Pre-fix the server returned 404 for
        files that lived in a later root, forcing peers to fall
        back to HuggingFace despite a complete local copy.
        """
        model_id = request.match_info["model_id"]
        file_path = request.match_info["file_path"]

        model_dirs = await self._locate_all_model_dirs(model_id)
        if not model_dirs:
            return web.Response(status=404, text="Model not found")

        complete_hit: Path | None = None
        best_partial: tuple[Path, PartialMeta] | None = None

        for model_dir in model_dirs:
            complete_candidate = _resolve_child(model_dir, file_path)
            partial_candidate = _resolve_child(model_dir, f"{file_path}.partial")
            if complete_candidate is None or partial_candidate is None:
                continue
            if complete_hit is None and await aios.path.exists(complete_candidate):
                complete_hit = complete_candidate
                # Complete copy in the first matching root wins; we
                # don't need to scan the rest for this file.
                break
            if await aios.path.exists(partial_candidate):
                meta = await _read_partial_meta(partial_candidate)
                if (
                    meta
                    and _meta_int(meta, "safe_bytes") > 0
                    and (
                        best_partial is None
                        or _meta_int(meta, "safe_bytes")
                        > _meta_int(best_partial[1], "safe_bytes")
                    )
                ):
                    best_partial = (partial_candidate, meta)

        if complete_hit is not None:
            serve_path = complete_hit
            file_size = (await aios.stat(complete_hit)).st_size
            safe_bytes = file_size
            is_complete = True
        elif best_partial is not None:
            partial_path, meta = best_partial
            serve_path = partial_path
            file_size = _meta_int(meta, "total")
            safe_bytes = _meta_int(meta, "safe_bytes")
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

    async def _locate_model_dir(self, model_id: str) -> Path | None:
        """Return the first configured directory that contains ``model_id``.

        Each candidate root is path-traversal-checked independently before we
        probe the filesystem. We prefer the first directory in ``models_dirs``
        that has a matching subdirectory; this preserves caller-specified
        priority (e.g. writable before read-only) without re-sorting.

        Note: callers that need to merge contents across multiple
        roots should use :meth:`_locate_all_model_dirs` instead. That
        helper exists to address Codex P2 (PR #16 round-(N+9),
        peer_file_server.py:201) where an earlier incomplete root
        masked a later complete copy.
        """
        for root in self.models_dirs:
            candidate = _resolve_child(root, model_id)
            if candidate is None:
                continue
            if await aios.path.exists(candidate):
                return candidate
        return None

    async def _locate_all_model_dirs(self, model_id: str) -> list[Path]:
        """Return every configured directory that contains ``model_id``.

        Roots are returned in the same priority order as
        ``self.models_dirs`` (writable before read-only) so callers
        can short-circuit to the first complete copy. Each candidate
        root is path-traversal-checked independently before we probe
        the filesystem.

        Codex P2 (PR #16 round-(N+9), peer_file_server.py:201):
        ``_locate_model_dir`` returned the first root that *contained*
        the model directory regardless of completeness. When an
        earlier writable root held a partial download and a later
        read-only mount held a complete copy, ``/status`` and
        ``/files`` only saw the partial tree -- peers thought the
        node had no canonical copy and fell back to HuggingFace.
        Callers that merge across roots use this helper to scan
        every match.
        """
        matches: list[Path] = []
        for root in self.models_dirs:
            candidate = _resolve_child(root, model_id)
            if candidate is None:
                continue
            if await aios.path.exists(candidate):
                matches.append(candidate)
        return matches


def _resolve_child(root: Path, relative_path: str) -> Path | None:
    """Resolve relative_path under root, rejecting path traversal."""
    resolved_root = root.resolve(strict=False)
    resolved_path = (resolved_root / relative_path).resolve(strict=False)
    if resolved_root in resolved_path.parents:
        return resolved_path
    return None


def _meta_int(meta: PartialMeta, key: str) -> int:
    value = meta.get(key, 0)
    return value if isinstance(value, int) else 0


async def _read_partial_meta(partial_path: Path) -> PartialMeta | None:
    """Read the .partial.meta companion file for a .partial download."""
    meta_path = Path(f"{partial_path}.meta")
    if not await aios.path.exists(meta_path):
        return None
    try:
        async with aiofiles.open(meta_path, "r") as f:
            data = cast(object, json.loads(await f.read()))
            if not isinstance(data, dict):
                return None
            raw_meta = cast(dict[object, object], data)
            return {
                str(key): value
                for key, value in raw_meta.items()
                if isinstance(value, (int, str))
            }
    except (json.JSONDecodeError, OSError):
        return None
