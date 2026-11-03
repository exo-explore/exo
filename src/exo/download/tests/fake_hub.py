"""A fake Hugging Face Hub for download tests.

It speaks just enough of the Hub's HTTP API for exo's downloader, over a real TCP
socket so that resets, stalls and short bodies happen exactly as they would on a
real network:

- ``GET /api/models/{repo}/tree/{revision}[/{path}]`` (``recursive``, ``cursor``
  pagination with a ``Link`` header);
- ``GET /api/models/{repo}/revision/{revision}`` (resolves a branch to a commit);
- ``HEAD|GET /{repo}/resolve/{revision}/{path}``: small files are served directly
  with their git blob sha1 as ETag; LFS files answer 302 to a CDN-like path with
  ``X-Linked-Size``/``X-Linked-Etag`` (sha256) and ``X-Repo-Commit``;
- ``HEAD|GET /cdn/{sha256}``: the LFS bytes, with Range support (206 and 416).

Faults are injected per request with :class:`Fault`. Every request is recorded so
tests can count how many bytes of each file were sent (and so re-fetched).
"""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import hashlib
import json
import socket
import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, cast, final
from urllib.parse import parse_qs, quote, unquote, urlsplit

LFS_SUFFIXES = (".safetensors", ".gguf", ".bin", ".pth", ".pt", ".npz")

RequestKind = Literal["tree", "revision", "resolve", "cdn", "other"]
Where = Literal["tree", "revision", "head", "resolve", "body", "any"]


def content_for(seed: str, size: int) -> bytes:
    """Deterministic pseudo-random bytes."""
    return hashlib.shake_256(seed.encode()).digest(size) if size > 0 else b""


def git_blob_sha1(content: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()


def sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@dataclass(frozen=True)
class Blob:
    content: bytes
    lfs: bool

    @property
    def size(self) -> int:
        return len(self.content)

    @property
    def git_sha1(self) -> str:
        return git_blob_sha1(self.content)

    @property
    def sha256(self) -> str:
        return sha256_hex(self.content)

    @property
    def etag(self) -> str:
        return self.sha256 if self.lfs else self.git_sha1


@dataclass(frozen=True)
class Commit:
    sha: str
    files: Mapping[str, Blob]


@final
class FakeRepo:
    """A model repository with a linear history. ``main`` is the newest commit."""

    def __init__(
        self,
        repo_id: str,
        files: Mapping[str, bytes],
        *,
        gated: bool = False,
    ) -> None:
        self.repo_id = repo_id
        self.gated = gated
        self.commits: list[Commit] = []
        self.commit(files)

    @property
    def main(self) -> Commit:
        return self.commits[-1]

    def commit(self, changes: Mapping[str, bytes | None]) -> str:
        """Make a new commit on main. ``None`` deletes a file. Returns the commit sha."""
        files: dict[str, Blob] = dict(self.commits[-1].files) if self.commits else {}
        for path, content in changes.items():
            if content is None:
                files.pop(path, None)
            else:
                files[path] = Blob(content, lfs=path.endswith(LFS_SUFFIXES))
        digest = hashlib.sha1(
            (self.commits[-1].sha if self.commits else self.repo_id).encode()
        )
        for path in sorted(files):
            digest.update(f"{path}:{files[path].git_sha1}".encode())
        sha = digest.hexdigest()
        self.commits.append(Commit(sha=sha, files=files))
        return sha

    def at(self, revision: str) -> Commit | None:
        if revision == "main":
            return self.main
        for commit in self.commits:
            if commit.sha == revision:
                return commit
        return None


def model_repo_files(
    repo_id: str,
    *,
    shards: int = 2,
    shard_size: int = 256 * 1024,
    extra: Mapping[str, bytes] | None = None,
) -> dict[str, bytes]:
    """The files of a small safetensors model repo (weights are random bytes)."""
    shard_names = [
        f"model-{i + 1:05d}-of-{shards:05d}.safetensors" for i in range(shards)
    ]
    index = {
        "metadata": {"total_size": shards * shard_size},
        "weight_map": {
            f"model.layers.{i}.weight": name for i, name in enumerate(shard_names)
        },
    }
    files: dict[str, bytes] = {
        ".gitattributes": b"*.safetensors filter=lfs diff=lfs merge=lfs -text\n",
        "README.md": f"# {repo_id}\n".encode(),
        "config.json": json.dumps(
            {"model_type": "llama", "num_hidden_layers": shards, "hidden_size": 64}
        ).encode(),
        "tokenizer.json": content_for(f"{repo_id}/tokenizer.json", 40_000),
        "tokenizer_config.json": b'{"model_max_length": 4096}',
        "model.safetensors.index.json": json.dumps(index).encode(),
    }
    for name in shard_names:
        files[name] = content_for(f"{repo_id}/{name}", shard_size)
    files.update(extra or {})
    return files


# --- faults ---------------------------------------------------------------------


@dataclass(frozen=True)
class Reset:
    """Send the headers and ``after_bytes`` of the body, then reset the connection.

    The reset comes ``delay`` seconds after the last byte, so the client has read
    what was sent (a reset discards data the client has not read yet).
    """

    after_bytes: int = 0
    delay: float = 0.05


@dataclass(frozen=True)
class Stall:
    """Send the headers and ``after_bytes`` of the body, then send nothing.

    With ``seconds=None`` the connection is never closed by the server.
    Otherwise the response carries on normally after ``seconds``.
    """

    after_bytes: int = 0
    seconds: float | None = None


@dataclass(frozen=True)
class ShortBody:
    """Promise the full Content-Length, send ``after_bytes``, then close cleanly."""

    after_bytes: int


@dataclass(frozen=True)
class Throttle:
    bytes_per_second: int


@dataclass(frozen=True)
class FirstByteDelay:
    seconds: float


@dataclass(frozen=True)
class Respond:
    """Answer with this status instead (5xx, 429, 401, 403, 404, ...)."""

    status: int
    headers: Mapping[str, str] = field(default_factory=dict[str, str])
    body: bytes = b""


@dataclass(frozen=True)
class IgnoreRange:
    """Answer a Range request with 200 and the whole body."""


@dataclass(frozen=True)
class RangeNotSatisfiable:
    """Answer a Range request with 416."""


@dataclass(frozen=True)
class Corrupt:
    """Flip the byte at ``offset`` of the body (offset into the whole file)."""

    offset: int = 0


@dataclass(frozen=True)
class Drop:
    """Reset the connection before sending anything."""


FaultAction = (
    Reset
    | Stall
    | ShortBody
    | Throttle
    | FirstByteDelay
    | Respond
    | IgnoreRange
    | RangeNotSatisfiable
    | Corrupt
    | Drop
)


def rate_limited(reset_seconds: float) -> Respond:
    """HF's 429, with the reset time in the ``RateLimit`` header."""
    return Respond(
        429,
        {
            "RateLimit": f'"api";r=0;t={reset_seconds:g}',
            "RateLimit-Policy": '"fixed window";"api";q=500;w=300',
        },
        b'{"error":"rate limited"}',
    )


def server_error(status: int = 503) -> Respond:
    return Respond(status, {}, b'{"error":"server error"}')


@dataclass
class Fault:
    """A fault applied to matching requests.

    ``where`` picks the requests: ``head`` (HEAD on resolve), ``resolve`` (any
    resolve request), ``body`` (requests that send file bytes: GET on resolve for
    small files, GET on the CDN for LFS files), ``tree``, ``revision`` or ``any``.
    ``path`` is a glob on the file path inside the repo, ``repo`` one on the repo
    id. ``times=None`` applies the
    fault forever; ``ranged`` restricts it to requests with (True) or without
    (False) a Range header.
    """

    action: FaultAction
    where: Where = "body"
    path: str = "*"
    times: int | None = 1
    ranged: bool | None = None
    repo: str = "*"
    hits: int = 0

    def matches(self, request: Request) -> bool:
        if self.times is not None and self.hits >= self.times:
            return False
        if self.ranged is not None and (request.range_start is not None) != self.ranged:
            return False
        if not fnmatch.fnmatch(request.file_path or "", self.path):
            return False
        if not fnmatch.fnmatch(request.repo_id or "", self.repo):
            return False
        match self.where:
            case "any":
                return True
            case "tree":
                return request.kind == "tree"
            case "revision":
                return request.kind == "revision"
            case "head":
                return request.kind == "resolve" and request.method == "HEAD"
            case "resolve":
                return request.kind == "resolve"
            case "body":
                return request.method == "GET" and request.serves_body


# --- requests -------------------------------------------------------------------


@dataclass
class Request:
    method: str
    target: str
    headers: dict[str, str]
    kind: RequestKind = "other"
    repo_id: str | None = None
    revision: str | None = None
    file_path: str | None = None
    range_start: int | None = None
    range_end: int | None = None
    serves_body: bool = False
    status: int = 0
    body_bytes: int = 0
    fault: FaultAction | None = None


@dataclass
class _Reply:
    status: int
    headers: dict[str, str]
    body: bytes = b""
    # where in the whole file the body starts (for byte accounting and Corrupt)
    offset: int = 0


_REASONS = {
    200: "OK",
    206: "Partial Content",
    302: "Found",
    307: "Temporary Redirect",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    416: "Range Not Satisfiable",
    429: "Too Many Requests",
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout",
}


@final
class FakeHub:
    """The fake Hub server. Use as ``async with FakeHub() as hub:``."""

    def __init__(self, *, page_size: int = 1000) -> None:
        self.repos: dict[str, FakeRepo] = {}
        self.faults: list[Fault] = []
        self.requests: list[Request] = []
        self.page_size = page_size
        # Bytes per second for every response body, None for unlimited.
        self.bandwidth: int | None = None
        self.port = 0
        self._server: asyncio.Server | None = None
        self._writers: set[asyncio.StreamWriter] = set()
        self._handlers: set[asyncio.Task[None]] = set()
        self._link_up = asyncio.Event()  # cleared while the link is "dead"
        self._link_up.set()
        self._triggers: list[tuple[str, int, Callable[[], None]]] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def __aenter__(self) -> FakeHub:
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._on_connection, "127.0.0.1", self.port, reuse_address=True
        )
        self.port = cast(tuple[str, int], self._server.sockets[0].getsockname())[1]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            self._server = None
        self._abort_connections()
        for task in list(self._handlers):
            task.cancel()
        for task in list(self._handlers):
            with contextlib.suppress(BaseException):
                await task

    # --- control ----------------------------------------------------------------

    def add_repo(self, repo: FakeRepo) -> FakeRepo:
        self.repos[repo.repo_id] = repo
        return repo

    def inject(self, fault: Fault) -> Fault:
        self.faults.append(fault)
        return fault

    def clear_faults(self) -> None:
        self.faults.clear()

    def when_sent(
        self, path: str, after_bytes: int, callback: Callable[[], None]
    ) -> None:
        """Call ``callback`` once, when ``after_bytes`` of files matching ``path`` were sent."""
        self._triggers.append((path, after_bytes, callback))

    async def go_down(self, mode: Literal["refuse", "blackhole"] = "refuse") -> None:
        """Take the endpoint off the network.

        ``refuse``: stop listening and reset open connections (a crashed server).
        ``blackhole``: keep connections open but send nothing (a dead link).
        """
        if mode == "blackhole":
            self._link_up.clear()
            return
        if self._server is not None:
            self._server.close()
            self._server = None
        self._abort_connections()

    async def come_up(self) -> None:
        self._link_up.set()
        if self._server is None:
            await self.start()

    # --- accounting -------------------------------------------------------------

    def body_bytes(self, repo_id: str | None = None, path: str = "*") -> int:
        """File bytes sent for files matching ``path`` (in all responses)."""
        return sum(
            r.body_bytes
            for r in self.requests
            if r.serves_body
            and (repo_id is None or r.repo_id == repo_id)
            and fnmatch.fnmatch(r.file_path or "", path)
        )

    def count(
        self,
        kind: RequestKind | None = None,
        method: str | None = None,
        path: str = "*",
    ) -> int:
        return sum(
            1
            for r in self.requests
            if (kind is None or r.kind == kind)
            and (method is None or r.method == method)
            and fnmatch.fnmatch(r.file_path or "", path)
        )

    # --- connection handling ----------------------------------------------------

    def _abort_connections(self) -> None:
        for writer in list(self._writers):
            _reset(writer)
        self._writers.clear()

    async def _on_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        assert task is not None
        self._handlers.add(task)
        self._writers.add(writer)
        try:
            while True:
                await self._link_up.wait()
                request = await _read_request(reader)
                if request is None:
                    break
                if not await self._serve(request, writer):
                    break
        except (
            ConnectionError,
            asyncio.IncompleteReadError,
            asyncio.LimitOverrunError,
        ):
            pass
        finally:
            self._writers.discard(writer)
            self._handlers.discard(task)
            if not writer.is_closing():
                writer.close()

    async def _serve(self, request: Request, writer: asyncio.StreamWriter) -> bool:
        """Answer one request. Returns whether the connection can be reused."""
        reply = self._route(request)
        self.requests.append(request)
        fault = next((f for f in self.faults if f.matches(request)), None)
        action: FaultAction | None = None
        if fault is not None:
            fault.hits += 1
            action = fault.action
            request.fault = action
        match action:
            case Drop():
                _reset(writer)
                return False
            case Respond(status=status, headers=headers, body=body):
                reply = _Reply(
                    status, {**headers, "Content-Type": "application/json"}, body
                )
                request.serves_body = False
            case RangeNotSatisfiable() if request.range_start is not None:
                size = len(reply.body) + reply.offset
                reply = _Reply(416, {"Content-Range": f"bytes */{size}"})
                request.serves_body = False
            case IgnoreRange() if (
                request.range_start is not None and request.serves_body
            ):
                request.range_start = None
                request.range_end = None
                reply = self._route(request)
            case FirstByteDelay(seconds=seconds):
                await asyncio.sleep(seconds)
            case _:
                pass
        request.status = reply.status
        return await self._send(request, reply, action, writer)

    async def _send(
        self,
        request: Request,
        reply: _Reply,
        action: FaultAction | None,
        writer: asyncio.StreamWriter,
    ) -> bool:
        headers = {
            "Content-Length": str(len(reply.body)),
            "Connection": "keep-alive",
            **reply.headers,
        }
        head = f"HTTP/1.1 {reply.status} {_REASONS.get(reply.status, 'Status')}\r\n"
        head += "".join(f"{k}: {v}\r\n" for k, v in headers.items()) + "\r\n"
        writer.write(head.encode())
        await writer.drain()
        if request.method == "HEAD":
            return True
        body = reply.body
        if isinstance(action, Corrupt):
            index = action.offset - reply.offset
            if 0 <= index < len(body):
                body = body[:index] + bytes([body[index] ^ 0xFF]) + body[index + 1 :]
        limit = len(body)
        match action:
            case Reset(after_bytes=n) | Stall(after_bytes=n) | ShortBody(after_bytes=n):
                limit = min(n, len(body))
            case _:
                pass
        rate = action.bytes_per_second if isinstance(action, Throttle) else None
        await self._write_body(request, body[:limit], reply.offset, rate, writer)
        match action:
            case Reset(delay=delay):
                await asyncio.sleep(delay)
                _reset(writer)
                return False
            case ShortBody():
                writer.close()
                return False
            case Stall(seconds=None):
                await asyncio.sleep(10**9)
                return False
            case Stall(seconds=float() as seconds):
                await asyncio.sleep(seconds)
                await self._write_body(
                    request, body[limit:], reply.offset + limit, rate, writer
                )
            case _:
                pass
        return True

    async def _write_body(
        self,
        request: Request,
        body: bytes,
        offset: int,
        rate: int | None,
        writer: asyncio.StreamWriter,
    ) -> None:
        position = 0
        while position < len(body):
            await self._link_up.wait()
            limit = rate if rate is not None else self.bandwidth
            chunk_size = (
                64 * 1024 if limit is None else max(1, min(16 * 1024, limit // 20))
            )
            chunk = body[position : position + chunk_size]
            writer.write(chunk)
            await writer.drain()
            position += len(chunk)
            if request.serves_body:
                request.body_bytes += len(chunk)
                self._fire_triggers(request)
            if limit is not None:
                await asyncio.sleep(len(chunk) / limit)

    def _fire_triggers(self, request: Request) -> None:
        for trigger in list(self._triggers):
            path, after_bytes, callback = trigger
            if fnmatch.fnmatch(request.file_path or "", path) and (
                self.body_bytes(request.repo_id, path) >= after_bytes
            ):
                self._triggers.remove(trigger)
                callback()

    # --- routing ----------------------------------------------------------------

    def _route(self, request: Request) -> _Reply:
        parts = urlsplit(request.target)
        query = parse_qs(parts.query)
        segments = [unquote(s) for s in parts.path.split("/")[1:]]
        if segments[:2] == ["api", "models"] and len(segments) >= 6:
            request.repo_id = f"{segments[2]}/{segments[3]}"
            request.revision = segments[5]
            if segments[4] == "tree":
                request.kind = "tree"
                request.file_path = "/".join(segments[6:])
                return self._tree(request, query)
            if segments[4] == "revision":
                request.kind = "revision"
                return self._revision(request)
        if len(segments) >= 5 and segments[2] == "resolve":
            request.kind = "resolve"
            request.repo_id = f"{segments[0]}/{segments[1]}"
            request.revision = segments[3]
            request.file_path = "/".join(segments[4:])
            return self._resolve(request)
        if segments[:1] == ["cdn"] and len(segments) == 2:
            request.kind = "cdn"
            request.repo_id = (query.get("repo") or [""])[0]
            request.file_path = (query.get("path") or [""])[0]
            return self._cdn(request, segments[1])
        return _error(404, "Not found", "")

    def _repo_or_error(self, request: Request) -> FakeRepo | _Reply:
        repo = self.repos.get(request.repo_id or "")
        if repo is None:
            # HF answers anonymous requests for unknown repos with a bare 401
            return _error(401, "Invalid username or password.", "")
        return repo

    def _commit_or_error(self, request: Request) -> Commit | _Reply:
        repo = self._repo_or_error(request)
        if isinstance(repo, _Reply):
            return repo
        commit = repo.at(request.revision or "")
        if commit is None:
            return _error(404, "Revision not found", "RevisionNotFound")
        return commit

    def _tree(self, request: Request, query: dict[str, list[str]]) -> _Reply:
        commit = self._commit_or_error(request)
        if isinstance(commit, _Reply):
            return commit
        prefix = request.file_path.strip("/") + "/" if request.file_path else ""
        recursive = (query.get("recursive") or ["false"])[0].lower() in ("true", "1")
        entries: list[dict[str, object]] = []
        directories: set[str] = set()
        for path in sorted(commit.files):
            if not path.startswith(prefix):
                continue
            rest = path[len(prefix) :]
            if "/" in rest:
                parts = rest.split("/")
                depth = len(parts) - 1 if recursive else 1
                for i in range(1, depth + 1):
                    directories.add(prefix + "/".join(parts[:i]))
                if not recursive:
                    continue
            blob = commit.files[path]
            entry: dict[str, object] = {
                "type": "file",
                "oid": blob.git_sha1,
                "size": blob.size,
                "path": path,
            }
            if blob.lfs:
                entry["lfs"] = {
                    "oid": blob.sha256,
                    "size": blob.size,
                    "pointerSize": 134,
                }
            entries.append(entry)
        for directory in sorted(directories):
            entries.append(
                {
                    "type": "directory",
                    "oid": hashlib.sha1(directory.encode()).hexdigest(),
                    "size": 0,
                    "path": directory,
                }
            )
        cursor = int((query.get("cursor") or ["0"])[0])
        page = entries[cursor : cursor + self.page_size]
        headers = {"Content-Type": "application/json"}
        if cursor + self.page_size < len(entries):
            next_url = (
                f"{self.url}/api/models/{request.repo_id}/tree/{request.revision}"
                f"{'/' + request.file_path if request.file_path else ''}"
                f"?recursive={str(recursive).lower()}&cursor={cursor + self.page_size}"
            )
            headers["Link"] = f'<{next_url}>; rel="next"'
        return _Reply(200, headers, json.dumps(page).encode())

    def _revision(self, request: Request) -> _Reply:
        commit = self._commit_or_error(request)
        if isinstance(commit, _Reply):
            return commit
        body = {
            "id": request.repo_id,
            "sha": commit.sha,
            "siblings": [{"rfilename": p} for p in sorted(commit.files)],
        }
        return _Reply(
            200, {"Content-Type": "application/json"}, json.dumps(body).encode()
        )

    def _resolve(self, request: Request) -> _Reply:
        repo = self._repo_or_error(request)
        if isinstance(repo, _Reply):
            return repo
        if repo.gated:
            if "authorization" not in request.headers:
                return _error(401, "Access to this model is restricted", "GatedRepo")
            return _error(403, "You have not accepted the terms", "GatedRepo")
        commit = repo.at(request.revision or "")
        if commit is None:
            return _error(404, "Revision not found", "RevisionNotFound")
        blob = commit.files.get(request.file_path or "")
        if blob is None:
            return _error(404, "Entry not found", "EntryNotFound")
        headers = {"X-Repo-Commit": commit.sha, "Accept-Ranges": "bytes"}
        if blob.lfs:
            location = (
                f"{self.url}/cdn/{blob.sha256}?repo={quote(repo.repo_id)}"
                f"&path={quote(request.file_path or '')}"
            )
            headers |= {
                "Location": location,
                "X-Linked-Size": str(blob.size),
                "X-Linked-Etag": f'"{blob.sha256}"',
                "ETag": f'"{blob.git_sha1}"',
            }
            return _Reply(302, headers)
        headers["ETag"] = f'"{blob.git_sha1}"'
        request.serves_body = True
        return _ranged(request, blob, headers)

    def _cdn(self, request: Request, sha256: str) -> _Reply:
        blob = next(
            (
                b
                for repo in self.repos.values()
                for commit in repo.commits
                for b in commit.files.values()
                if b.lfs and b.sha256 == sha256
            ),
            None,
        )
        if blob is None:
            return _error(404, "Not found", "")
        request.serves_body = True
        return _ranged(request, blob, {"ETag": f'"{sha256}"', "Accept-Ranges": "bytes"})


def _ranged(request: Request, blob: Blob, headers: dict[str, str]) -> _Reply:
    if request.range_start is None:
        return _Reply(200, headers, blob.content)
    start = request.range_start
    end = (
        blob.size - 1
        if request.range_end is None
        else min(request.range_end, blob.size - 1)
    )
    if start >= blob.size or start > end:
        request.serves_body = False
        return _Reply(416, headers | {"Content-Range": f"bytes */{blob.size}"})
    headers = headers | {"Content-Range": f"bytes {start}-{end}/{blob.size}"}
    return _Reply(206, headers, blob.content[start : end + 1], offset=start)


def _error(status: int, message: str, code: str) -> _Reply:
    headers = {"Content-Type": "application/json"}
    if code:
        headers["X-Error-Code"] = code
    return _Reply(status, headers, json.dumps({"error": message}).encode())


def _reset(writer: asyncio.StreamWriter) -> None:
    """Close with RST instead of FIN."""
    sock = cast(socket.socket | None, writer.get_extra_info("socket"))
    if sock is not None:
        with contextlib.suppress(OSError):
            sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
            )
    writer.transport.abort()


async def _read_request(reader: asyncio.StreamReader) -> Request | None:
    line = await reader.readline()
    if not line:
        return None
    method, target, _ = line.decode("latin-1").split(" ", 2)
    headers: dict[str, str] = {}
    while True:
        header_line = (await reader.readline()).decode("latin-1").rstrip("\r\n")
        if not header_line:
            break
        name, _, value = header_line.partition(":")
        headers[name.strip().lower()] = value.strip()
    request = Request(method=method, target=target, headers=headers)
    range_header = headers.get("range")
    if range_header is not None and range_header.startswith("bytes="):
        first, _, last = range_header[len("bytes=") :].partition("-")
        request.range_start = int(first) if first else 0
        request.range_end = int(last) if last else None
    return request
