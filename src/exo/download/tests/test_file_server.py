# pyright: reportPrivateUsage = false, reportMissingTypeArgument = false, reportUnknownParameterType = false, reportUnknownMemberType = false, reportUnknownVariableType = false, reportAny = false

"""Tests for the P2P model file server.

We don't bind to a real port — instead we mount the file_server's URL
handler in an aiohttp `TestServer` so each test gets a fresh in-process
server with a local tmp_path standing in for the model directories.
"""

import asyncio
import socket
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from exo.download import file_server


def _write_model_file(model_dir: Path, model_id: str, file_path: str, content: bytes) -> Path:
    """Create a fake model file at ``<model_dir>/<normalized-id>/<file_path>``
    and return the absolute path."""
    normalized = model_id.replace("/", "--")
    target = model_dir / normalized / file_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


@pytest_asyncio.fixture
async def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[Any, Path]]:
    """Spin up an in-process server that uses ``tmp_path`` as its only
    model directory. Yields ``(test_client, model_dir)`` so each test
    can both write fixture files and make HTTP requests."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    monkeypatch.setattr(file_server, "EXO_MODELS_DIRS", (model_dir,))
    monkeypatch.setattr(file_server, "EXO_MODELS_READ_ONLY_DIRS", ())

    app = web.Application()
    app.router.add_get("/{path:.*}", file_server._handle_model_file)

    async with TestClient(TestServer(app)) as test_client:
        yield test_client, model_dir


# ---- happy paths ------------------------------------------------------------


async def test_serves_existing_file(client: tuple[Any, Path]) -> None:
    payload = b"hello world" * 1000
    _write_model_file(client[1], "test-org/m1", "model.safetensors", payload) 

    async with client[0].get("/test-org/m1/model.safetensors") as resp:
        assert resp.status == 200
        assert resp.headers["Content-Length"] == str(len(payload))
        assert resp.headers["Accept-Ranges"] == "bytes"
        assert await resp.read() == payload


async def test_nested_subdirectory_path(client: tuple[Any, Path]) -> None:
    """Files inside subdirectories of the model (e.g. tokenizer/, configs/)
    should be served — the path after model_id can have any depth."""
    payload = b"nested"
    _write_model_file(
        client[1], 
        "test-org/m1",
        "tokenizer/special_tokens_map.json",
        payload,
    )
    async with client[0].get(
        "/test-org/m1/tokenizer/special_tokens_map.json"
    ) as resp:
        assert resp.status == 200
        assert await resp.read() == payload


# ---- range requests ---------------------------------------------------------


async def test_range_request_returns_206_partial(client: tuple[Any, Path]) -> None:
    payload = b"0123456789" * 1000  # 10000 bytes
    _write_model_file(client[1], "test-org/m1", "weights.bin", payload) 

    async with client[0].get(
        "/test-org/m1/weights.bin", headers={"Range": "bytes=2000-"}
    ) as resp:
        assert resp.status == 206
        assert resp.headers["Content-Length"] == str(len(payload) - 2000)
        assert (
            resp.headers["Content-Range"]
            == f"bytes 2000-{len(payload) - 1}/{len(payload)}"
        )
        assert await resp.read() == payload[2000:]


async def test_range_request_at_zero_returns_200_not_206(client: tuple[Any, Path]) -> None:
    """A `Range: bytes=0-` header is technically a range request but covers
    the whole file. The server returns 200, not 206 — matches what
    curl/aiohttp produce when streaming a fresh download."""
    payload = b"AB" * 50
    _write_model_file(client[1], "test-org/m1", "f.bin", payload) 

    async with client[0].get(
        "/test-org/m1/f.bin", headers={"Range": "bytes=0-"}
    ) as resp:
        assert resp.status == 200
        assert "Content-Range" not in resp.headers
        assert await resp.read() == payload


async def test_range_past_end_returns_416(client: tuple[Any, Path]) -> None:
    payload = b"short"
    _write_model_file(client[1], "test-org/m1", "f.bin", payload) 

    async with client[0].get(
        "/test-org/m1/f.bin", headers={"Range": "bytes=1000-"}
    ) as resp:
        assert resp.status == 416
        assert resp.headers.get("Content-Range") == f"bytes */{len(payload)}"


# ---- error paths ------------------------------------------------------------


async def test_missing_file_returns_404(client: tuple[Any, Path]) -> None:
    async with client[0].get("/test-org/m1/never-existed.safetensors") as resp:
        assert resp.status == 404


async def test_missing_model_returns_404(client: tuple[Any, Path]) -> None:
    """The file_path exists in concept but the model dir doesn't."""
    async with client[0].get("/test-org/never-downloaded/file.bin") as resp:
        assert resp.status == 404


async def test_path_too_short_returns_404(client: tuple[Any, Path]) -> None:
    """The handler requires at least three segments: org/model/file. Two
    segments is not a valid model file URL."""
    async with client[0].get("/just-two/segments") as resp:
        assert resp.status == 404


async def test_internals_resolve_safe_rejects_lateral_traversal(
    tmp_path: Path,
) -> None:
    """``_resolve_safe`` must reject a ``..`` traversal that lands inside
    a *sibling* model dir under the same ``model_dir`` root.

    The previous implementation only checked ``is_relative_to(model_dir)``,
    which allowed a sideways escape from ``<root>/foo--m1/`` into
    ``<root>/foo--m1-ROGUE/`` because both are technically inside ``root``.
    The current implementation pins the check to the *specific* normalized
    subdirectory, so the lateral escape is rejected.

    We test the helper directly — aiohttp client URL parsers normalize
    ``..`` segments away client-side, which would let a buggy server
    silently pass this test. The protocol-level test below
    (``test_lateral_traversal_blocked_at_protocol_level``) closes that
    loop by sending a raw HTTP request.
    """
    model_dir = tmp_path / "models"
    (model_dir / "test-org--m1").mkdir(parents=True)
    (model_dir / "test-org--m1" / "real.bin").write_bytes(b"ok")
    (model_dir / "test-org--m1-ROGUE").mkdir()
    rogue = model_dir / "test-org--m1-ROGUE" / "secret.txt"
    rogue.write_bytes(b"do not leak")

    # The exact attack: ask for /<requested-model>/../<sibling>/<file>.
    assert (
        file_server._resolve_safe(
            str(model_dir), "test-org--m1", "../test-org--m1-ROGUE/secret.txt"
        )
        is None
    )
    # And a sanity check — the legit lookup still works.
    found = file_server._resolve_safe(str(model_dir), "test-org--m1", "real.bin")
    assert found is not None
    assert found.read_bytes() == b"ok"


async def test_internals_resolve_safe_rejects_escape_to_root(
    tmp_path: Path,
) -> None:
    """Going far enough up that we leave ``model_dir`` entirely is also
    rejected (the original ``is_relative_to(model_dir)`` check covered
    this — keep a test so a future refactor can't regress it)."""
    model_dir = tmp_path / "models"
    (model_dir / "org--m1").mkdir(parents=True)
    secret_outside = tmp_path / "outside_secret.txt"
    secret_outside.write_bytes(b"definitely don't")

    assert (
        file_server._resolve_safe(
            str(model_dir), "org--m1", "../../outside_secret.txt"
        )
        is None
    )


# ---- multi-directory resolution --------------------------------------------


async def test_falls_back_across_writable_and_read_only_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model that lives in a read-only mirror should still be servable."""
    writable = tmp_path / "writable"
    read_only = tmp_path / "readonly"
    writable.mkdir()
    read_only.mkdir()
    monkeypatch.setattr(file_server, "EXO_MODELS_DIRS", (writable,))
    monkeypatch.setattr(file_server, "EXO_MODELS_READ_ONLY_DIRS", (read_only,))

    payload = b"from read-only mirror"
    _write_model_file(read_only, "test-org/m1", "f.bin", payload)

    app = web.Application()
    app.router.add_get("/{path:.*}", file_server._handle_model_file)
    async with (
        TestClient(TestServer(app)) as test_client,
        test_client.get("/test-org/m1/f.bin") as resp,
    ):
        assert resp.status == 200
        assert await resp.read() == payload


async def test_deduplicates_overlapping_dir_lists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same path appearing in both writable and read-only must not be visited
    twice (otherwise the same file would be returned for the wrong reason)."""
    shared = tmp_path / "shared"
    shared.mkdir()
    monkeypatch.setattr(file_server, "EXO_MODELS_DIRS", (shared,))
    monkeypatch.setattr(file_server, "EXO_MODELS_READ_ONLY_DIRS", (shared,))

    dirs = file_server._all_model_dirs()
    assert len(dirs) == 1
    assert dirs[0] == str(shared.resolve())


# ---- Range header robustness -----------------------------------------------


@pytest.mark.parametrize(
    "header,expected",
    [
        (None, None),
        ("", None),
        ("bytes=", None),
        ("bytes=-", None),
        ("bytes=-100", None),  # suffix-form not supported (yet) — None means "ignore"
        ("bytes=abc-", None),  # non-numeric — must NOT raise
        ("bytes=10.5-", None),  # float — must NOT raise
        ("bytes=10-20,30-40", None),  # multi-range not supported
        ("invalid", None),  # missing "bytes=" unit
        ("items=10-", None),  # wrong unit
        ("bytes=0-", 0),
        ("bytes=10-", 10),
        ("bytes=999999999999999-", 999999999999999),
    ],
)
def test_parse_range_start(header: str | None, expected: int | None) -> None:
    """``_parse_range_start`` must never raise on malformed input — a 500
    here would be a trivial DoS for any peer who can hit the server. It
    returns ``None`` for unparseable / unsupported headers so the caller
    falls through to a normal full-file response."""
    assert file_server._parse_range_start(header) == expected


# ---- HTTP-protocol-level security tests ------------------------------------
# These bypass aiohttp's URL-normalizing client by sending raw HTTP requests
# over a socket. The path-traversal attack vector has historically been
# masked by client-side ``..`` collapsing, so we exercise it at the byte
# level to be sure the server's defense actually fires.


async def _raw_http_get(port: int, request_line: bytes, headers: bytes = b"") -> bytes:
    """Open a socket, send a literal HTTP/1.1 request, return the raw response."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("127.0.0.1", port))
    s.send(request_line + b"\r\n" + headers + b"\r\n")
    await asyncio.sleep(0.2)  # give the server a tick to write the response
    s.settimeout(1.0)
    data = b""
    try:
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
    except TimeoutError:
        pass
    s.close()
    return data


@pytest_asyncio.fixture
async def raw_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[int, Path]]:
    """Start a real file_server on an ephemeral port. Yields ``(port, model_dir)``."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    monkeypatch.setattr(file_server, "EXO_MODELS_DIRS", (model_dir,))
    monkeypatch.setattr(file_server, "EXO_MODELS_READ_ONLY_DIRS", ())

    app = web.Application()
    app.router.add_get("/{path:.*}", file_server._handle_model_file)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port: int = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield port, model_dir
    finally:
        await runner.cleanup()


async def test_lateral_traversal_blocked_at_protocol_level(
    raw_server: tuple[int, Path],
) -> None:
    """Send the actual ``..`` bytes over the wire — no client-side
    normalization. A buggy server would 200 with the rogue file's contents."""
    port, model_dir = raw_server
    (model_dir / "org--m1").mkdir()
    (model_dir / "org--m1" / "real.bin").write_bytes(b"REAL")
    (model_dir / "org--m1-ROGUE").mkdir()
    (model_dir / "org--m1-ROGUE" / "secret.txt").write_bytes(b"DO-NOT-LEAK")

    resp = await _raw_http_get(
        port,
        b"GET /org/m1/../org--m1-ROGUE/secret.txt HTTP/1.1\r\nHost: x",
    )
    assert resp.startswith(b"HTTP/1.1 404"), resp[:80]
    assert b"DO-NOT-LEAK" not in resp


async def test_escape_to_filesystem_root_blocked_at_protocol_level(
    raw_server: tuple[int, Path], tmp_path: Path
) -> None:
    """A traversal that escapes the model_dir entirely must also fail."""
    port, model_dir = raw_server
    (model_dir / "org--m1").mkdir()
    outside = tmp_path / "outside_secret.txt"
    outside.write_bytes(b"OUTSIDE-SECRET")

    resp = await _raw_http_get(
        port,
        b"GET /org/m1/../../outside_secret.txt HTTP/1.1\r\nHost: x",
    )
    assert resp.startswith(b"HTTP/1.1 404"), resp[:80]
    assert b"OUTSIDE-SECRET" not in resp


async def test_malformed_range_does_not_500(
    raw_server: tuple[int, Path],
) -> None:
    """A junk Range header must not crash the handler. Pre-fix this was a
    trivial DoS — ``Range: bytes=abc-`` raised ValueError out of
    ``int(...)`` and aiohttp returned 500."""
    port, model_dir = raw_server
    (model_dir / "org--m1").mkdir()
    (model_dir / "org--m1" / "real.bin").write_bytes(b"REAL")

    resp = await _raw_http_get(
        port,
        b"GET /org/m1/real.bin HTTP/1.1\r\nHost: x",
        b"Range: bytes=abc-\r\n",
    )
    # Either 200 (header ignored — current behavior) or 416 (rejected)
    # is fine; a 500 is not.
    status = resp.split(b" ", 2)[1]
    assert status in (b"200", b"416"), resp[:80]


async def test_error_response_does_not_echo_request_path(
    raw_server: tuple[int, Path],
) -> None:
    """The 404 body for a missing file must not reflect the requested path
    back. Even though Content-Type is text/plain (so this isn't an XSS by
    itself), reflecting attacker-controlled input is bad practice and would
    chain badly with any client that ever rendered the response as HTML."""
    port, _ = raw_server
    sentinel = b"<script>alert(1)</script>"
    resp = await _raw_http_get(
        port,
        b"GET /" + sentinel + b"/m1/file.bin HTTP/1.1\r\nHost: x",
    )
    assert resp.startswith(b"HTTP/1.1 404"), resp[:80]
    body = resp.split(b"\r\n\r\n", 1)[1] if b"\r\n\r\n" in resp else b""
    assert sentinel not in body, body


# ---- X-File-SHA256 header --------------------------------------------------


async def test_serves_x_file_sha256_when_sidecar_present(
    client: tuple[Any, Path],
) -> None:
    """A ``<file>.sha256`` sidecar makes the file_server emit
    ``X-File-SHA256`` so the receiver can verify the bytes after curl
    finishes. The sidecar is the source of truth — we never recompute the
    hash on the request path because that would block for tens of seconds
    on multi-GB safetensors."""
    payload = b"deadbeef" * 100
    digest = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
    target = client[1] / "test-org--m1" / "f.bin"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    target.with_suffix(".bin.sha256").write_text(digest + "\n")

    async with client[0].get("/test-org/m1/f.bin") as resp:
        assert resp.status == 200
        assert resp.headers.get("X-File-SHA256") == digest
        assert await resp.read() == payload


async def test_omits_x_file_sha256_when_sidecar_absent(
    client: tuple[Any, Path],
) -> None:
    """No sidecar → no header. The receiver treats this as "skip
    verification" and proceeds (logged at debug)."""
    _write_model_file(client[1], "test-org/m1", "f.bin", b"x")
    async with client[0].get("/test-org/m1/f.bin") as resp:
        assert resp.status == 200
        assert "X-File-SHA256" not in resp.headers


async def test_rejects_malformed_sidecar(
    client: tuple[Any, Path],
) -> None:
    """A sidecar with non-hex contents (or wrong length) must be ignored
    rather than echoed as if it were a real digest. Otherwise a corrupt
    sidecar on the source would propagate junk hashes to peers, who would
    then fail verification despite the bytes being fine."""
    target = client[1] / "test-org--m1" / "f.bin"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"x")
    # Length wrong + non-hex content
    target.with_suffix(".bin.sha256").write_text("not a digest at all")

    async with client[0].get("/test-org/m1/f.bin") as resp:
        assert resp.status == 200
        assert "X-File-SHA256" not in resp.headers


# ---- concurrency cap -------------------------------------------------------


async def test_excess_concurrent_requests_get_503(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the semaphore is at its cap, additional requests get 503 with
    Retry-After rather than queueing forever.

    We monkey-patch ``_serve`` to await an event we control, so we can
    deterministically pin one request inside the semaphore and observe
    the second getting rejected. (Reading from a real file races with
    aiohttp's internal buffering and isn't reliable as a pinning
    mechanism.)"""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    monkeypatch.setattr(file_server, "EXO_MODELS_DIRS", (model_dir,))
    monkeypatch.setattr(file_server, "EXO_MODELS_READ_ONLY_DIRS", ())
    monkeypatch.setattr(file_server, "_serve_semaphore", asyncio.Semaphore(1))

    release = asyncio.Event()

    async def held_serve(_: web.Request) -> web.Response:
        await release.wait()
        return web.Response(status=200, body=b"done")

    monkeypatch.setattr(file_server, "_serve", held_serve)

    app = web.Application()
    app.router.add_get("/{path:.*}", file_server._handle_model_file)

    async with TestClient(TestServer(app)) as test_client:
        # Start the first request — it'll block inside held_serve, holding
        # the semaphore.
        first_task = asyncio.create_task(test_client.get("/test-org/m1/f.bin"))
        # Yield to the event loop so the first request actually enters _serve.
        await asyncio.sleep(0.05)
        # Second request — semaphore is locked, must come back 503.
        async with test_client.get("/test-org/m1/f.bin") as second:
            assert second.status == 503
            assert second.headers.get("Retry-After") == "1"
        # Release the held first request and confirm it completes normally.
        release.set()
        first_resp = await first_task
        assert first_resp.status == 200
        await first_resp.release()


async def test_semaphore_releases_on_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If a serve raises (e.g. for a missing file → 404), the semaphore
    permit must still be released so subsequent requests aren't blocked
    forever. ``async with semaphore`` guarantees this; we lock the cap at 1
    and verify that 100 sequential 404s don't deadlock."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    monkeypatch.setattr(file_server, "EXO_MODELS_DIRS", (model_dir,))
    monkeypatch.setattr(file_server, "EXO_MODELS_READ_ONLY_DIRS", ())
    monkeypatch.setattr(file_server, "_serve_semaphore", asyncio.Semaphore(1))

    app = web.Application()
    app.router.add_get("/{path:.*}", file_server._handle_model_file)

    async with TestClient(TestServer(app)) as test_client:
        for _ in range(100):
            async with test_client.get("/test-org/m1/missing.bin") as resp:
                assert resp.status == 404
