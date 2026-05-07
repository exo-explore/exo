"""Tests for peer-to-peer model downloading."""

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import aiofiles
import aiofiles.os as aios
import pytest

from exo.download.peer_download import download_file_from_peer, get_peer_file_status
from exo.download.peer_file_server import PeerFileServer


@pytest.fixture
async def temp_models_dir(tmp_path: Path) -> AsyncIterator[Path]:
    """Set up a temporary models directory for testing."""
    models_dir = tmp_path / "models"
    await aios.makedirs(models_dir, exist_ok=True)
    yield models_dir


@pytest.fixture
async def peer_server(temp_models_dir: Path) -> AsyncIterator[PeerFileServer]:
    """Start a PeerFileServer on a random port for testing."""
    server = PeerFileServer(host="127.0.0.1", port=0, models_dir=temp_models_dir)
    # Use port 0 to let OS assign a free port
    from aiohttp import web

    server._runner = web.AppRunner(server._app)
    await server._runner.setup()
    site = web.TCPSite(server._runner, "127.0.0.1", 0)
    await site.start()
    # Get the actual port assigned
    server.port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    yield server
    await server.shutdown()


class TestPeerFileServer:
    """Tests for the HTTP file server that serves model files to peers."""

    async def test_health_check(self, peer_server: PeerFileServer) -> None:
        """Health endpoint should return ok."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://127.0.0.1:{peer_server.port}/health"
            ) as r:
                assert r.status == 200
                data = await r.json()
                assert data["status"] == "ok"

    async def test_status_empty_model(self, peer_server: PeerFileServer) -> None:
        """Status for non-existent model should return empty file list."""
        files = await get_peer_file_status(
            "127.0.0.1", peer_server.port, "nonexistent--model"
        )
        assert files is not None
        assert len(files) == 0

    async def test_status_with_complete_file(
        self, peer_server: PeerFileServer, temp_models_dir: Path
    ) -> None:
        """Status should report complete files correctly."""
        model_dir = temp_models_dir / "test--model"
        await aios.makedirs(model_dir, exist_ok=True)

        # Create a complete test file
        async with aiofiles.open(model_dir / "config.json", "wb") as f:
            await f.write(b'{"test": true}')

        files = await get_peer_file_status(
            "127.0.0.1", peer_server.port, "test--model"
        )
        assert files is not None
        assert len(files) == 1
        assert files[0].path == "config.json"
        assert files[0].complete is True
        assert files[0].safe_bytes == 14

    async def test_status_with_partial_file(
        self, peer_server: PeerFileServer, temp_models_dir: Path
    ) -> None:
        """Status should report partial files with safe byte count."""
        model_dir = temp_models_dir / "test--model"
        await aios.makedirs(model_dir, exist_ok=True)

        # Create a partial file with metadata
        partial_data = b"x" * 1024
        async with aiofiles.open(model_dir / "weights.safetensors.partial", "wb") as f:
            await f.write(partial_data)

        meta = {"safe_bytes": 1024, "total": 4096, "etag": "abc123"}
        async with aiofiles.open(
            model_dir / "weights.safetensors.partial.meta", "w"
        ) as f:
            await f.write(json.dumps(meta))

        files = await get_peer_file_status(
            "127.0.0.1", peer_server.port, "test--model"
        )
        assert files is not None
        assert len(files) == 1
        assert files[0].path == "weights.safetensors"
        assert files[0].complete is False
        assert files[0].safe_bytes == 1024
        assert files[0].size == 4096

    async def test_serve_complete_file(
        self, peer_server: PeerFileServer, temp_models_dir: Path
    ) -> None:
        """Should serve a complete file with correct headers."""
        model_dir = temp_models_dir / "test--model"
        await aios.makedirs(model_dir, exist_ok=True)

        content = b"hello world test content"
        async with aiofiles.open(model_dir / "config.json", "wb") as f:
            await f.write(content)

        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://127.0.0.1:{peer_server.port}/files/test--model/config.json"
            ) as r:
                assert r.status == 200
                assert r.headers["X-Exo-Complete"] == "true"
                body = await r.read()
                assert body == content

    async def test_serve_with_range_request(
        self, peer_server: PeerFileServer, temp_models_dir: Path
    ) -> None:
        """Should support Range requests for resume."""
        model_dir = temp_models_dir / "test--model"
        await aios.makedirs(model_dir, exist_ok=True)

        content = b"0123456789abcdef"
        async with aiofiles.open(model_dir / "weights.bin", "wb") as f:
            await f.write(content)

        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://127.0.0.1:{peer_server.port}/files/test--model/weights.bin",
                headers={"Range": "bytes=8-"},
            ) as r:
                assert r.status == 206
                body = await r.read()
                assert body == b"89abcdef"

    async def test_file_not_found(self, peer_server: PeerFileServer) -> None:
        """Should return 404 for missing files."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://127.0.0.1:{peer_server.port}/files/test--model/missing.bin"
            ) as r:
                assert r.status == 404


class TestPeerDownloadClient:
    """Tests for downloading files from a peer server."""

    async def test_download_complete_file(
        self, peer_server: PeerFileServer, temp_models_dir: Path, tmp_path: Path
    ) -> None:
        """Should download a complete file from peer."""
        # Set up source file on the peer server
        model_dir = temp_models_dir / "test--model"
        await aios.makedirs(model_dir, exist_ok=True)

        content = b"model weights data " * 100
        async with aiofiles.open(model_dir / "weights.bin", "wb") as f:
            await f.write(content)

        # Download to a different directory
        download_dir = tmp_path / "downloads" / "test--model"
        await aios.makedirs(download_dir, exist_ok=True)

        progress_calls: list[tuple[int, int, bool]] = []

        result = await download_file_from_peer(
            "127.0.0.1",
            peer_server.port,
            "test--model",
            "weights.bin",
            download_dir,
            len(content),
            on_progress=lambda c, t, r: progress_calls.append((c, t, r)),
        )

        assert result is not None
        assert result == download_dir / "weights.bin"
        async with aiofiles.open(result, "rb") as f:
            downloaded = await f.read()
        assert downloaded == content
        # Should have progress calls including final
        assert len(progress_calls) > 0
        assert progress_calls[-1][2] is True  # is_renamed

    async def test_download_returns_none_on_missing(
        self, peer_server: PeerFileServer, tmp_path: Path
    ) -> None:
        """Should return None when file doesn't exist on peer."""
        download_dir = tmp_path / "downloads" / "test--model"
        await aios.makedirs(download_dir, exist_ok=True)

        result = await download_file_from_peer(
            "127.0.0.1",
            peer_server.port,
            "test--model",
            "nonexistent.bin",
            download_dir,
            1000,
        )
        assert result is None

    async def test_download_returns_none_on_unreachable_peer(
        self, tmp_path: Path
    ) -> None:
        """Should return None when peer is unreachable."""
        download_dir = tmp_path / "downloads" / "test--model"
        await aios.makedirs(download_dir, exist_ok=True)

        result = await download_file_from_peer(
            "127.0.0.1",
            19999,  # Nobody listening
            "test--model",
            "weights.bin",
            download_dir,
            1000,
        )
        assert result is None

    async def test_skip_already_complete(
        self, peer_server: PeerFileServer, temp_models_dir: Path, tmp_path: Path
    ) -> None:
        """Should skip download if file already exists locally with correct size."""
        model_dir = temp_models_dir / "test--model"
        await aios.makedirs(model_dir, exist_ok=True)

        content = b"existing content"
        # File already exists in target
        download_dir = tmp_path / "downloads" / "test--model"
        await aios.makedirs(download_dir, exist_ok=True)
        async with aiofiles.open(download_dir / "config.json", "wb") as f:
            await f.write(content)

        result = await download_file_from_peer(
            "127.0.0.1",
            peer_server.port,
            "test--model",
            "config.json",
            download_dir,
            len(content),
        )

        assert result is not None
        assert result == download_dir / "config.json"
