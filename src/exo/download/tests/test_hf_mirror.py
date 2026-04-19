"""Tests for hf-mirror compatibility and the 416 / stale-partial recovery path.

Observed header shapes (huggingface.co & hf-mirror.com, probed 2026-04-19):
  - Small non-LFS file HEAD /resolve/...: 307, Location relative (/api/resolve-cache/...),
    x-linked-etag present, x-linked-size ABSENT. file_meta must recurse via Location.
  - LFS/xet file HEAD /resolve/...: 302, Location absolute (cas-bridge.xethub.hf.co),
    x-linked-size AND x-linked-etag present. file_meta must trust headers, not recurse.

These tests exercise those paths via mocks only; no network, no large file I/O.
"""

import hashlib
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiofiles
import aiofiles.os as aios
import pytest

from exo.download.download_utils import (
    HuggingFaceAuthenticationError,
    XetNotReachableError,
    _download_file,  # pyright: ignore[reportPrivateUsage]
    download_file_with_retry,
    file_meta,
)
from exo.shared.types.common import ModelId


@pytest.fixture
def model_id() -> ModelId:
    return ModelId("mlx-community/Qwen2.5-0.5B-Instruct-4bit")


def _mk_head_response(
    status: int,
    headers: dict[str, str] | None = None,
    url: str = "https://huggingface.co/x",
) -> MagicMock:
    """Build an aiohttp-like response mock for HEAD requests."""
    response = MagicMock()
    response.status = status
    response.headers = headers or {}
    response.url = url
    return response


def _wire_session_head(session_factory: MagicMock, responses: list[MagicMock]) -> None:
    """Configure a mock session factory to return the given HEAD responses in sequence."""
    call_state: dict[str, int] = {"i": 0}

    def head_side_effect(*_: object, **__: object) -> MagicMock:
        idx = call_state["i"]
        call_state["i"] += 1
        r = responses[idx] if idx < len(responses) else responses[-1]
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=r)
        cm.__aexit__ = AsyncMock(return_value=None)
        return cm

    session = MagicMock()
    session.head.side_effect = head_side_effect  # pyright: ignore[reportAny]
    session_factory.return_value.__aenter__ = AsyncMock(return_value=session)  # pyright: ignore[reportAny]
    session_factory.return_value.__aexit__ = AsyncMock(return_value=None)  # pyright: ignore[reportAny]


class TestFileMetaRedirects:
    async def test_307_relative_location_recurses_and_reads_200(
        self, model_id: ModelId
    ) -> None:
        """hf-mirror small-file shape: 307 → relative /api/resolve-cache, then 200 with
        content-length. file_meta must follow the Location and return the 200's values."""
        redirect = _mk_head_response(
            status=307,
            headers={
                "location": "/api/resolve-cache/models/x/abc/config.json?etag=%22abc%22",
                "x-linked-etag": '"deadbeef"',
                # NOTE: no x-linked-size on 307 for small non-LFS files
                "content-length": "284",  # redirect body length — NOT the file size
            },
        )
        follow_200 = _mk_head_response(
            status=200,
            headers={
                "content-length": "783",
                "etag": '"deadbeef"',
            },
        )
        with patch(
            "exo.download.download_utils.create_http_session"
        ) as mock_session_factory:
            _wire_session_head(mock_session_factory, [redirect, follow_200])
            length, etag = await file_meta(model_id, "main", "config.json")
        assert length == 783
        assert etag == "deadbeef"

    async def test_302_lfs_header_trust_no_recursion(self, model_id: ModelId) -> None:
        """LFS/xet shape: 302 → absolute cas-bridge URL with x-linked-size and x-linked-etag.
        file_meta must trust the headers without following the redirect."""
        resp = _mk_head_response(
            status=302,
            headers={
                "location": "https://cas-bridge.xethub.hf.co/xet-bridge-us/...",
                "x-linked-size": "278064920",
                "x-linked-etag": '"ddffab9cbc7bf6dde941c6724841eeca8981fcfa81ca20ff8efff1396326d153"',
                "content-length": "1369",
            },
        )
        with patch(
            "exo.download.download_utils.create_http_session"
        ) as mock_session_factory:
            _wire_session_head(mock_session_factory, [resp])
            length, etag = await file_meta(model_id, "main", "model.safetensors")
        assert length == 278064920
        assert (
            etag == "ddffab9cbc7bf6dde941c6724841eeca8981fcfa81ca20ff8efff1396326d153"
        )

    async def test_401_raises_auth_error(self, model_id: ModelId) -> None:
        resp = _mk_head_response(status=401, headers={})
        with patch(
            "exo.download.download_utils.create_http_session"
        ) as mock_session_factory:
            _wire_session_head(mock_session_factory, [resp])
            with pytest.raises(HuggingFaceAuthenticationError):
                await file_meta(model_id, "main", "config.json")


class TestResumeHandling:
    """Covers exo-explore/exo#1914: stale partial larger than current remote → HTTP 416."""

    async def test_resume_byte_pos_greater_than_length_clears_partial(
        self, model_id: ModelId, tmp_path: Path
    ) -> None:
        """If the .partial exceeds the remote size, it must be dropped before the GET
        so we don't send a Range past EOF."""
        target_dir = tmp_path / "d"
        await aios.makedirs(target_dir, exist_ok=True)
        partial = target_dir / "config.json.partial"
        async with aiofiles.open(partial, "wb") as f:
            await f.write(b"A" * 1000)  # larger than the "remote" length (500) below

        body = b"B" * 500
        remote_hash = hashlib.sha1(b"blob 500\0" + body).hexdigest()

        # Mock GET response: streams `body` in one chunk.
        mock_get = MagicMock()
        mock_get.status = 200
        mock_get.url = "https://huggingface.co/x/resolve/main/config.json"
        mock_get.content.read = AsyncMock(side_effect=[body, b""])  # pyright: ignore[reportAny]

        with (
            patch(
                "exo.download.download_utils.file_meta",
                new_callable=AsyncMock,
                return_value=(500, remote_hash),
            ),
            patch(
                "exo.download.download_utils.create_http_session"
            ) as mock_session_factory,
        ):
            session = MagicMock()
            session.get.return_value.__aenter__ = AsyncMock(return_value=mock_get)  # pyright: ignore[reportAny]
            session.get.return_value.__aexit__ = AsyncMock(return_value=None)  # pyright: ignore[reportAny]
            mock_session_factory.return_value.__aenter__ = AsyncMock(  # pyright: ignore[reportAny]
                return_value=session
            )
            mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=None)  # pyright: ignore[reportAny]

            await _download_file(model_id, "main", "config.json", target_dir)

            # The Range header must NOT have been sent — partial was stale.
            _, kwargs = session.get.call_args  # pyright: ignore[reportAny]
            sent_headers_raw = kwargs.get("headers", {})  # pyright: ignore[reportAny]
            sent_headers: dict[str, str] = dict(sent_headers_raw)  # pyright: ignore[reportAny]
            assert "Range" not in sent_headers, (
                f"Expected no Range header after stale partial, got {sent_headers}"
            )

        # Final file must exist with the correct bytes (partial was rewritten from scratch).
        final_path = target_dir / "config.json"
        assert await aios.path.exists(final_path)
        async with aiofiles.open(final_path, "rb") as f:
            assert await f.read() == body

    async def test_explicit_416_from_server_deletes_partial(
        self, model_id: ModelId, tmp_path: Path
    ) -> None:
        """If the server returns 416 anyway (e.g. endpoint served a smaller revision
        between the HEAD and the GET), we must delete the partial and raise so the
        retry layer starts fresh on the next attempt."""
        target_dir = tmp_path / "d"
        await aios.makedirs(target_dir, exist_ok=True)
        partial = target_dir / "config.json.partial"
        async with aiofiles.open(partial, "wb") as f:
            await f.write(b"A" * 200)

        mock_get = MagicMock()
        mock_get.status = 416
        mock_get.url = "https://hf-mirror.com/x/resolve/main/config.json"
        mock_get.content.read = AsyncMock(return_value=b"")  # pyright: ignore[reportAny]

        with (
            patch(
                "exo.download.download_utils.file_meta",
                new_callable=AsyncMock,
                return_value=(500, "deadbeef"),  # length > partial, so Range IS sent
            ),
            patch(
                "exo.download.download_utils.create_http_session"
            ) as mock_session_factory,
        ):
            session = MagicMock()
            session.get.return_value.__aenter__ = AsyncMock(return_value=mock_get)  # pyright: ignore[reportAny]
            session.get.return_value.__aexit__ = AsyncMock(return_value=None)  # pyright: ignore[reportAny]
            mock_session_factory.return_value.__aenter__ = AsyncMock(  # pyright: ignore[reportAny]
                return_value=session
            )
            mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=None)  # pyright: ignore[reportAny]

            with pytest.raises(Exception, match="416"):
                await _download_file(model_id, "main", "config.json", target_dir)

        assert not await aios.path.exists(partial), (
            "Partial must be removed after 416 so the retry starts fresh"
        )


class TestEndpointFallback:
    async def test_mirror_used_when_primary_fails(
        self, model_id: ModelId, tmp_path: Path
    ) -> None:
        """When the primary endpoint is unreachable, download_file_with_retry must try
        the mirror on a subsequent attempt and succeed."""
        target_dir = tmp_path / "d"
        await aios.makedirs(target_dir, exist_ok=True)

        body = b"ok"
        remote_hash = hashlib.sha1(b"blob 2\0" + body).hexdigest()

        endpoints_seen: list[str] = []

        async def fake_download(*args: Any, **kwargs: Any) -> Path:  # pyright: ignore[reportAny]
            endpoint = kwargs.get("endpoint")
            assert isinstance(endpoint, str)
            endpoints_seen.append(endpoint)
            if "huggingface.co" in endpoint:
                import aiohttp

                raise aiohttp.ClientConnectorError(
                    connection_key=MagicMock(), os_error=OSError("blocked")
                )
            # hf-mirror path succeeds: write the file and return
            target_path = target_dir / "config.json"
            async with aiofiles.open(target_path, "wb") as f:
                _ = await f.write(body)
            return target_path

        with (
            patch.dict(
                "os.environ",
                {
                    "HF_ENDPOINT": "https://huggingface.co",
                    "HF_MIRROR_ENDPOINT": "https://hf-mirror.com",
                },
            ),
            patch(
                "exo.download.download_utils._download_file",
                side_effect=fake_download,
            ),
            patch(
                "exo.download.download_utils.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            _ = remote_hash  # silence unused warning; hash validated inside _download_file normally
            result = await download_file_with_retry(
                model_id, "main", "config.json", target_dir
            )

        assert result == target_dir / "config.json"
        assert any("huggingface.co" in e for e in endpoints_seen), endpoints_seen
        assert any("hf-mirror.com" in e for e in endpoints_seen), endpoints_seen

    async def test_authn_error_does_not_fallback(
        self, model_id: ModelId, tmp_path: Path
    ) -> None:
        """401/403 from the primary should not trigger the mirror — the mirror will
        reject identically (gated model) and retries are just noise."""
        target_dir = tmp_path / "d"
        await aios.makedirs(target_dir, exist_ok=True)

        endpoints_seen: list[str] = []

        async def fake_download(*args: Any, **kwargs: Any) -> Path:  # pyright: ignore[reportAny]
            endpoint = kwargs.get("endpoint")
            assert isinstance(endpoint, str)
            endpoints_seen.append(endpoint)
            raise HuggingFaceAuthenticationError("401 from " + endpoint)

        with (
            patch.dict(
                "os.environ",
                {
                    "HF_ENDPOINT": "https://huggingface.co",
                    "HF_MIRROR_ENDPOINT": "https://hf-mirror.com",
                },
            ),
            patch(
                "exo.download.download_utils._download_file",
                side_effect=fake_download,
            ),
            pytest.raises(HuggingFaceAuthenticationError),
        ):
            await download_file_with_retry(
                model_id, "main", "model.safetensors", target_dir
            )

        assert endpoints_seen == ["https://huggingface.co"], (
            f"Mirror must not be tried on 401/403; endpoints tried: {endpoints_seen}"
        )


class TestXetDetection:
    async def test_xet_cas_unreachable_raises_actionable_error(
        self, model_id: ModelId, tmp_path: Path
    ) -> None:
        """A GET that lands on cas-bridge.xethub.hf.co and then errors must surface
        XetNotReachableError with guidance (not a cryptic connection error)."""
        target_dir = tmp_path / "d"
        await aios.makedirs(target_dir, exist_ok=True)

        import aiohttp

        async def raise_after_first(*_: Any, **__: Any) -> bytes:  # pyright: ignore[reportAny]
            raise aiohttp.ServerDisconnectedError("connection reset")

        mock_get = MagicMock()
        mock_get.status = 200  # aiohttp follows the 302 transparently by default
        mock_get.url = "https://cas-bridge.xethub.hf.co/xet-bridge-us/deadbeef"
        mock_get.content.read = AsyncMock(side_effect=raise_after_first)  # pyright: ignore[reportAny]

        with (
            patch(
                "exo.download.download_utils.file_meta",
                new_callable=AsyncMock,
                return_value=(278064920, "deadbeef"),
            ),
            patch(
                "exo.download.download_utils.create_http_session"
            ) as mock_session_factory,
        ):
            session = MagicMock()
            session.get.return_value.__aenter__ = AsyncMock(return_value=mock_get)  # pyright: ignore[reportAny]
            session.get.return_value.__aexit__ = AsyncMock(return_value=None)  # pyright: ignore[reportAny]
            mock_session_factory.return_value.__aenter__ = AsyncMock(  # pyright: ignore[reportAny]
                return_value=session
            )
            mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=None)  # pyright: ignore[reportAny]

            with pytest.raises(XetNotReachableError) as exc_info:
                await _download_file(model_id, "main", "model.safetensors", target_dir)

        msg = str(exc_info.value)
        assert "xet" in msg.lower()
        assert "cas-bridge" in msg or "HTTPS_PROXY" in msg
