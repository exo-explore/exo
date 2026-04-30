"""Tests for HuggingFace 429 rate-limit handling in download_utils."""

from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiofiles.os as aios
import pytest

from exo.download.download_utils import (
    HuggingFaceRateLimitError,
    _download_file,  # pyright: ignore[reportPrivateUsage]
    _fetch_file_list,  # pyright: ignore[reportPrivateUsage]
    _parse_retry_after,  # pyright: ignore[reportPrivateUsage]
    download_file_with_retry,
    fetch_file_list_with_retry,
    file_meta,
)
from exo.shared.types.common import ModelId

# captured from a real HF 429 on 2026-04-30 (header is lowercased by Cloudfront)
REAL_HF_429_HEADERS_2026_04_30 = {
    "ratelimit": '"api";r=0;t=52',
    "ratelimit-policy": '"fixed window";"api";q=500;w=300',
}


class TestParseRetryAfter:
    def test_parses_documented_format(self) -> None:
        assert _parse_retry_after({"RateLimit": '"api";r=0;t=243'}) == 243.0

    def test_parses_real_hf_response(self) -> None:
        assert _parse_retry_after(REAL_HF_429_HEADERS_2026_04_30) == 52.0

    def test_parses_resolvers_bucket(self) -> None:
        assert _parse_retry_after({"ratelimit": '"resolvers";r=0;t=120'}) == 120.0

    def test_parses_pages_bucket(self) -> None:
        assert _parse_retry_after({"ratelimit": '"pages";r=0;t=10'}) == 10.0

    def test_returns_none_when_header_missing(self) -> None:
        assert _parse_retry_after({}) is None

    def test_returns_none_when_only_retry_after_present(self) -> None:
        assert _parse_retry_after({"Retry-After": "60"}) is None

    def test_returns_none_when_format_unrecognised(self) -> None:
        assert _parse_retry_after({"ratelimit": "garbage"}) is None

    def test_handles_extra_whitespace(self) -> None:
        assert _parse_retry_after({"ratelimit": '"api"; r=0; t=42'}) == 42.0


class TestFetchFileListRetry:
    async def test_uses_retry_after_from_error(self) -> None:
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        async def fake_fetch(*args: object, **kwargs: object) -> list[object]:
            if not sleeps:
                raise HuggingFaceRateLimitError("rate limited", retry_after=2.0)
            return []

        with (
            patch(
                "exo.download.download_utils._fetch_file_list", side_effect=fake_fetch
            ),
            patch("exo.download.download_utils.asyncio.sleep", side_effect=fake_sleep),
        ):
            result = await fetch_file_list_with_retry(ModelId("test/model"))

        assert result == []
        assert len(sleeps) == 1
        assert 2.0 <= sleeps[0] < 3.0  # retry_after + jitter[0,1)

    async def test_falls_back_to_exp_backoff_when_no_retry_after(self) -> None:
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        async def fake_fetch(*args: object, **kwargs: object) -> list[object]:
            if not sleeps:
                raise HuggingFaceRateLimitError("rate limited", retry_after=None)
            return []

        with (
            patch(
                "exo.download.download_utils._fetch_file_list", side_effect=fake_fetch
            ),
            patch("exo.download.download_utils.asyncio.sleep", side_effect=fake_sleep),
        ):
            await fetch_file_list_with_retry(ModelId("test/model"))

        assert len(sleeps) == 1
        assert 1.0 <= sleeps[0] < 2.0  # 2**0 + jitter[0,1)

    async def test_caps_sleep_at_max_window(self) -> None:
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        async def fake_fetch(*args: object, **kwargs: object) -> list[object]:
            if not sleeps:
                raise HuggingFaceRateLimitError("rate limited", retry_after=10_000.0)
            return []

        with (
            patch(
                "exo.download.download_utils._fetch_file_list", side_effect=fake_fetch
            ),
            patch("exo.download.download_utils.asyncio.sleep", side_effect=fake_sleep),
        ):
            await fetch_file_list_with_retry(ModelId("test/model"))

        assert len(sleeps) == 1
        assert 300.0 <= sleeps[0] < 301.0  # cap + jitter[0,1)

    async def test_retries_up_to_five_times(self) -> None:
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        async def fake_fetch(*args: object, **kwargs: object) -> list[object]:
            raise HuggingFaceRateLimitError("rate limited", retry_after=1.0)

        with (
            patch(
                "exo.download.download_utils._fetch_file_list", side_effect=fake_fetch
            ),
            patch("exo.download.download_utils.asyncio.sleep", side_effect=fake_sleep),
            pytest.raises(HuggingFaceRateLimitError),
        ):
            await fetch_file_list_with_retry(ModelId("test/model"))

        assert len(sleeps) == 4  # 5 attempts -> 4 sleeps before giving up


class TestDownloadFileRetry:
    @pytest.fixture
    async def target_dir(self, tmp_path: Path) -> AsyncIterator[Path]:
        target = tmp_path / "downloads"
        await aios.makedirs(target, exist_ok=True)
        yield target

    async def test_uses_retry_after_from_error(self, target_dir: Path) -> None:
        sleeps: list[float] = []
        results: list[Path] = [target_dir / "file.bin"]

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        async def fake_download(*args: object, **kwargs: object) -> Path:
            if not sleeps:
                raise HuggingFaceRateLimitError("rate limited", retry_after=5.0)
            return results[0]

        with (
            patch(
                "exo.download.download_utils._download_file",
                side_effect=fake_download,
            ),
            patch("exo.download.download_utils.asyncio.sleep", side_effect=fake_sleep),
        ):
            result = await download_file_with_retry(
                ModelId("test/model"), "main", "file.bin", target_dir
            )

        assert result == results[0]
        assert len(sleeps) == 1
        assert 5.0 <= sleeps[0] < 6.0

    async def test_caps_sleep_at_max_window(self, target_dir: Path) -> None:
        sleeps: list[float] = []
        results: list[Path] = [target_dir / "file.bin"]

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        async def fake_download(*args: object, **kwargs: object) -> Path:
            if not sleeps:
                raise HuggingFaceRateLimitError("rate limited", retry_after=99_999.0)
            return results[0]

        with (
            patch(
                "exo.download.download_utils._download_file",
                side_effect=fake_download,
            ),
            patch("exo.download.download_utils.asyncio.sleep", side_effect=fake_sleep),
        ):
            await download_file_with_retry(
                ModelId("test/model"), "main", "file.bin", target_dir
            )

        assert len(sleeps) == 1
        assert 300.0 <= sleeps[0] < 301.0

    async def test_retries_up_to_five_times(self, target_dir: Path) -> None:
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        with (
            patch(
                "exo.download.download_utils._download_file",
                new_callable=AsyncMock,
                side_effect=HuggingFaceRateLimitError("rate limited", retry_after=1.0),
            ),
            patch("exo.download.download_utils.asyncio.sleep", side_effect=fake_sleep),
            pytest.raises(HuggingFaceRateLimitError),
        ):
            await download_file_with_retry(
                ModelId("test/model"), "main", "file.bin", target_dir
            )

        assert len(sleeps) == 4


def _make_mock_session_returning(
    response_attrs: dict[str, object], method: str = "get"
) -> MagicMock:
    """Build a MagicMock that mimics ``create_http_session`` returning a
    response whose ``status`` / ``headers`` are set from ``response_attrs``.

    Mocks the chain ``create_http_session().__aenter__() -> session``, and
    ``session.<method>().__aenter__() -> response``.
    """
    mock_response = MagicMock()
    for k, v in response_attrs.items():
        setattr(mock_response, k, v)

    mock_session = MagicMock()
    method_mock = getattr(mock_session, method)  # pyright: ignore[reportAny]
    method_mock.return_value.__aenter__ = AsyncMock(  # pyright: ignore[reportAny]
        return_value=mock_response
    )
    method_mock.return_value.__aexit__ = AsyncMock(  # pyright: ignore[reportAny]
        return_value=None
    )

    mock_factory = MagicMock()
    mock_factory.return_value.__aenter__ = AsyncMock(  # pyright: ignore[reportAny]
        return_value=mock_session
    )
    mock_factory.return_value.__aexit__ = AsyncMock(  # pyright: ignore[reportAny]
        return_value=None
    )
    return mock_factory


REAL_HF_429_HEADER_DICT = {"ratelimit": '"api";r=0;t=52'}


class TestRateLimitAtHttpCallSites:
    """Verify each HF call site translates an HTTP 429 into a
    ``HuggingFaceRateLimitError`` carrying the parsed ``retry_after``.

    These tests would catch regressions where (a) the 429 branch is
    deleted, (b) ``_parse_retry_after`` stops being called, or
    (c) the wrong header object is passed to it.
    """

    async def test_fetch_file_list_maps_429_to_rate_limit_error(self) -> None:
        mock_factory = _make_mock_session_returning(
            {"status": 429, "headers": REAL_HF_429_HEADER_DICT}
        )
        with (
            patch("exo.download.download_utils.create_http_session", mock_factory),
            pytest.raises(HuggingFaceRateLimitError) as exc_info,
        ):
            await _fetch_file_list(ModelId("test/model"), "main")
        assert exc_info.value.retry_after == 52.0

    async def test_file_meta_maps_429_to_rate_limit_error(self) -> None:
        mock_factory = _make_mock_session_returning(
            {"status": 429, "headers": REAL_HF_429_HEADER_DICT}, method="head"
        )
        with (
            patch("exo.download.download_utils.create_http_session", mock_factory),
            pytest.raises(HuggingFaceRateLimitError) as exc_info,
        ):
            await file_meta(ModelId("test/model"), "main", "weights.safetensors")
        assert exc_info.value.retry_after == 52.0

    async def test_file_meta_maps_429_after_307_redirect(self) -> None:
        """When the initial HEAD 307s and the redirected HEAD then 429s,
        the 429 must still surface as ``HuggingFaceRateLimitError``."""
        # First HEAD -> 307 with a Location header pointing somewhere new.
        first_response = MagicMock()
        first_response.status = 307
        first_response.headers = {"location": "/redirected/url"}

        # Second HEAD (the recursive call) -> 429 with the real-HF header.
        second_response = MagicMock()
        second_response.status = 429
        second_response.headers = REAL_HF_429_HEADER_DICT

        responses = iter([first_response, second_response])

        mock_session = MagicMock()
        mock_session.head.return_value.__aenter__ = AsyncMock(  # pyright: ignore[reportAny]
            side_effect=lambda: next(responses)
        )
        mock_session.head.return_value.__aexit__ = AsyncMock(  # pyright: ignore[reportAny]
            return_value=None
        )

        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(  # pyright: ignore[reportAny]
            return_value=mock_session
        )
        mock_factory.return_value.__aexit__ = AsyncMock(  # pyright: ignore[reportAny]
            return_value=None
        )

        with (
            patch("exo.download.download_utils.create_http_session", mock_factory),
            pytest.raises(HuggingFaceRateLimitError) as exc_info,
        ):
            await file_meta(ModelId("test/model"), "main", "weights.safetensors")
        assert exc_info.value.retry_after == 52.0

    async def test_download_file_maps_429_to_rate_limit_error(
        self, tmp_path: Path
    ) -> None:
        target_dir = tmp_path / "downloads"
        await aios.makedirs(target_dir, exist_ok=True)
        # No local file -> _download_file goes straight to file_meta then GET.
        # We need both calls to succeed enough to reach the GET branch:
        #   - file_meta returns a non-429 (size, etag) so we proceed.
        #   - the GET then 429s.
        with (
            patch(
                "exo.download.download_utils.file_meta",
                new_callable=AsyncMock,
                return_value=(100, "abc123"),
            ),
            patch(
                "exo.download.download_utils.create_http_session",
                _make_mock_session_returning(
                    {"status": 429, "headers": REAL_HF_429_HEADER_DICT}
                ),
            ),
            pytest.raises(HuggingFaceRateLimitError) as exc_info,
        ):
            await _download_file(
                ModelId("test/model"), "main", "weights.safetensors", target_dir
            )
        assert exc_info.value.retry_after == 52.0
