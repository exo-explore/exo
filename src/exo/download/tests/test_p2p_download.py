# pyright: reportPrivateUsage = false, reportAny = false

"""End-to-end test: file_server (producer) ↔ _download_file_from_peer (consumer).

Spins up a real ``file_server`` on an ephemeral port and drives the curl-based
``_download_file_from_peer`` against it. Skipped if curl is not on $PATH.
"""

import asyncio
import hashlib
import shutil
import socket
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from aiohttp import web

from exo.download import file_server
from exo.download.download_utils import (
    _download_file_from_peer,
    _parse_x_file_sha256,
    _sha256_sidecar_path,
)
from exo.shared.models.model_cards import ModelId

pytestmark = pytest.mark.skipif(
    shutil.which("curl") is None,
    reason="curl is not installed; P2P download tests need the curl binary",
)


def _free_port() -> int:
    """Bind a socket to port 0, read back the OS-assigned port, release it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest_asyncio.fixture
async def p2p_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[str, Path]]:
    """Start a file_server on a free port serving a tmp model directory.

    Yields ``(repo_url, server_model_dir)`` so the test can write source
    files into ``server_model_dir`` and pass ``repo_url`` to the downloader.
    """
    server_model_dir = tmp_path / "server_models"
    server_model_dir.mkdir()
    monkeypatch.setattr(file_server, "EXO_MODELS_DIRS", (server_model_dir,))
    monkeypatch.setattr(file_server, "EXO_MODELS_READ_ONLY_DIRS", ())

    port = _free_port()
    app = web.Application()
    app.router.add_get("/{path:.*}", file_server._handle_model_file)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    try:
        yield f"http://127.0.0.1:{port}", server_model_dir
    finally:
        await runner.cleanup()


def _seed_source_file(
    server_model_dir: Path, model_id: str, path: str, content: bytes
) -> None:
    normalized = model_id.replace("/", "--")
    target = server_model_dir / normalized / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)


def _seed_source_file_with_sidecar(
    server_model_dir: Path,
    model_id: str,
    path: str,
    content: bytes,
    sidecar_override: str | None = None,
) -> None:
    """Seed a file plus its ``<file>.sha256`` sidecar so the file_server will
    advertise ``X-File-SHA256`` for it. Pass ``sidecar_override`` to write
    a deliberately-wrong hash for mismatch tests."""
    _seed_source_file(server_model_dir, model_id, path, content)
    normalized = model_id.replace("/", "--")
    target = server_model_dir / normalized / path
    digest = sidecar_override or hashlib.sha256(content).hexdigest()
    _sha256_sidecar_path(target).write_text(digest + "\n")


# ---- happy path ------------------------------------------------------------


async def test_round_trip_serves_and_downloads(
    p2p_server: tuple[str, Path], tmp_path: Path
) -> None:
    repo_url, server_model_dir = p2p_server
    model_id = ModelId("test-org/round-trip")
    payload = b"x" * (256 * 1024)  # 256 KiB

    _seed_source_file(server_model_dir, str(model_id), "weights.bin", payload)

    client_target_dir = tmp_path / "client_models"
    client_target_dir.mkdir()

    final = await _download_file_from_peer(
        repo_url=repo_url,
        model_id=model_id,
        path="weights.bin",
        target_dir=client_target_dir,
    )

    assert final == client_target_dir / "weights.bin"
    assert final.read_bytes() == payload
    # The .partial file was renamed away on success.
    assert not (client_target_dir / "weights.bin.partial").exists()


async def test_progress_callback_invoked(
    p2p_server: tuple[str, Path], tmp_path: Path
) -> None:
    """``_download_file_from_peer`` invokes ``on_progress(curr, total, done)``
    at least once on success (with done=True for the final call)."""
    repo_url, server_model_dir = p2p_server
    model_id = ModelId("test-org/progress")
    payload = b"y" * 4096

    _seed_source_file(server_model_dir, str(model_id), "f.bin", payload)

    client_target_dir = tmp_path / "client_models"
    client_target_dir.mkdir()

    calls: list[tuple[int, int, bool]] = []

    def _record(curr: int, total: int, done: bool) -> None:
        calls.append((curr, total, done))

    await _download_file_from_peer(
        repo_url=repo_url,
        model_id=model_id,
        path="f.bin",
        target_dir=client_target_dir,
        on_progress=_record,
        total_bytes=len(payload),
    )

    assert any(c[2] for c in calls), "expected at least one done=True progress call"
    final = next(c for c in reversed(calls) if c[2])
    assert final[0] == len(payload)


async def test_resume_from_partial(
    p2p_server: tuple[str, Path], tmp_path: Path
) -> None:
    """If a ``.partial`` file already has bytes from a previous attempt,
    curl's ``-C -`` resumes from where it left off rather than restarting."""
    repo_url, server_model_dir = p2p_server
    model_id = ModelId("test-org/resume")
    payload = b"R" * 8192

    _seed_source_file(server_model_dir, str(model_id), "f.bin", payload)

    client_target_dir = tmp_path / "client_models"
    client_target_dir.mkdir()

    # Pre-seed the .partial file with the first half — simulating a
    # previously interrupted download.
    partial = client_target_dir / "f.bin.partial"
    partial.write_bytes(payload[: len(payload) // 2])

    final = await _download_file_from_peer(
        repo_url=repo_url,
        model_id=model_id,
        path="f.bin",
        target_dir=client_target_dir,
    )

    assert final.read_bytes() == payload


# ---- error paths -----------------------------------------------------------


async def test_curl_raises_runtime_error_on_404(
    p2p_server: tuple[str, Path], tmp_path: Path
) -> None:
    repo_url, _ = p2p_server
    model_id = ModelId("test-org/missing")

    client_target_dir = tmp_path / "client_models"
    client_target_dir.mkdir()

    with pytest.raises(RuntimeError, match="curl failed"):
        await _download_file_from_peer(
            repo_url=repo_url,
            model_id=model_id,
            path="never-existed.bin",
            target_dir=client_target_dir,
        )


async def test_curl_raises_runtime_error_on_unreachable_host(
    tmp_path: Path,
) -> None:
    """If the file server is not running (or the URL is wrong), the
    subprocess exits non-zero and we surface that as a RuntimeError."""
    model_id = ModelId("test-org/unreach")
    client_target_dir = tmp_path / "client_models"
    client_target_dir.mkdir()

    # Bind a port to grab a number, then close it — the address should be
    # unbound by the time curl tries to dial.
    port = _free_port()

    with pytest.raises(RuntimeError, match="curl failed"):
        await asyncio.wait_for(
            _download_file_from_peer(
                repo_url=f"http://127.0.0.1:{port}",
                model_id=model_id,
                path="any.bin",
                target_dir=client_target_dir,
            ),
            timeout=10.0,
        )


# ---- hash verification ----------------------------------------------------


async def test_hash_header_emitted_when_sidecar_exists(
    p2p_server: tuple[str, Path], tmp_path: Path
) -> None:
    """When the source has a ``.sha256`` sidecar, the receiver verifies the
    downloaded bytes against it — and persists its own sidecar so it can in
    turn serve to other peers with the same hash header."""
    repo_url, server_model_dir = p2p_server
    model_id = ModelId("test-org/with-hash")
    payload = b"verify-me" * 4096
    _seed_source_file_with_sidecar(
        server_model_dir, str(model_id), "f.bin", payload
    )

    client_target_dir = tmp_path / "client_models"
    client_target_dir.mkdir()
    final = await _download_file_from_peer(
        repo_url=repo_url,
        model_id=model_id,
        path="f.bin",
        target_dir=client_target_dir,
    )
    assert final.read_bytes() == payload
    # Receiver re-emitted the hash sidecar.
    receiver_sidecar = _sha256_sidecar_path(final)
    assert receiver_sidecar.exists()
    assert receiver_sidecar.read_text().strip() == hashlib.sha256(payload).hexdigest()


async def test_hash_mismatch_aborts_and_raises(
    p2p_server: tuple[str, Path], tmp_path: Path
) -> None:
    """If the peer's reported X-File-SHA256 doesn't match the bytes we
    received, the download must fail (so the outer retry loop can re-fetch),
    the partial file must be removed, and no final file may be left in
    target_dir for the caller to mistakenly trust."""
    repo_url, server_model_dir = p2p_server
    model_id = ModelId("test-org/bad-hash")
    payload = b"these-are-the-real-bytes" * 1000
    wrong_hash = "0" * 64  # plausible-looking SHA256 hex, won't match payload
    _seed_source_file_with_sidecar(
        server_model_dir,
        str(model_id),
        "f.bin",
        payload,
        sidecar_override=wrong_hash,
    )

    client_target_dir = tmp_path / "client_models"
    client_target_dir.mkdir()

    with pytest.raises(RuntimeError, match="P2P hash mismatch"):
        await _download_file_from_peer(
            repo_url=repo_url,
            model_id=model_id,
            path="f.bin",
            target_dir=client_target_dir,
        )
    # The final file must NOT exist — we don't rename .partial → final on
    # mismatch, and we delete the .partial.
    assert not (client_target_dir / "f.bin").exists()
    assert not (client_target_dir / "f.bin.partial").exists()


async def test_no_hash_header_skips_verification(
    p2p_server: tuple[str, Path], tmp_path: Path
) -> None:
    """If the peer doesn't have a sidecar (e.g. the source's own download
    happened before we shipped sidecar-writing), the file_server omits the
    header entirely. The receiver logs a debug note and proceeds — this is
    backward-compatible with peers running an older build."""
    repo_url, server_model_dir = p2p_server
    model_id = ModelId("test-org/no-hash")
    payload = b"no-sidecar-here" * 2000
    # NB: NOT using _seed_source_file_with_sidecar — no sidecar gets written.
    _seed_source_file(server_model_dir, str(model_id), "f.bin", payload)

    client_target_dir = tmp_path / "client_models"
    client_target_dir.mkdir()
    final = await _download_file_from_peer(
        repo_url=repo_url,
        model_id=model_id,
        path="f.bin",
        target_dir=client_target_dir,
    )
    assert final.read_bytes() == payload
    # No sidecar on receiver either, since the peer didn't advertise one.
    assert not _sha256_sidecar_path(final).exists()


# ---- _parse_x_file_sha256 unit tests --------------------------------------


def test_parse_x_file_sha256_picks_last_match() -> None:
    """If a curl ``-D`` dump contains more than one set of response headers
    (e.g. a redirect chain), we want the *final* response's hash, not an
    intermediate one. Walk the dump in order and let the last value win."""
    digest_a = "a" * 64
    digest_b = "b" * 64
    headers_text = (
        "HTTP/1.1 301 Moved\r\n"
        f"X-File-SHA256: {digest_a}\r\n"
        "\r\n"
        "HTTP/1.1 200 OK\r\n"
        f"X-File-SHA256: {digest_b}\r\n"
        "\r\n"
    )
    assert _parse_x_file_sha256(headers_text) == digest_b


def test_parse_x_file_sha256_rejects_non_hex_value() -> None:
    """Defense in depth: the server-side reader also rejects non-hex values,
    but if a malformed header somehow makes it onto the wire we must not
    pass it on as a "valid" expected hash."""
    headers_text = "HTTP/1.1 200 OK\r\nX-File-SHA256: definitely-not-a-hex-digest\r\n\r\n"
    assert _parse_x_file_sha256(headers_text) is None


def test_parse_x_file_sha256_returns_none_when_absent() -> None:
    headers_text = "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n\r\n"
    assert _parse_x_file_sha256(headers_text) is None


def test_parse_x_file_sha256_case_insensitive_header_name() -> None:
    """HTTP header names are case-insensitive per RFC 7230. We accept any
    casing in the curl dump."""
    digest = "f" * 64
    headers_text = f"HTTP/1.1 200 OK\r\nx-FILE-sha256: {digest}\r\n\r\n"
    assert _parse_x_file_sha256(headers_text) == digest
