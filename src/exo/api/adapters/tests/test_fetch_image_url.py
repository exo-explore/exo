# pyright: reportAny=false
"""Tests for fetch_image_url SSRF protection.

Verifies that scheme, metadata-host, and literal private/loopback/link-local IP
checks fire before any network access, and that valid public URLs are allowed.
"""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exo.api.adapters.chat_completions import fetch_image_url


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_session(response_data: bytes = b"img") -> MagicMock:
    """Return a mock aiohttp session whose .get() never actually sends a request."""
    resp = MagicMock()
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    resp.raise_for_status = MagicMock()
    resp.read = AsyncMock(return_value=response_data)

    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.get = MagicMock(return_value=resp)
    return session


# ---------------------------------------------------------------------------
# Rejection cases — scheme
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_file_scheme_rejected() -> None:
    with patch("exo.api.adapters.chat_completions.create_http_session") as mock_cs:
        with pytest.raises(ValueError, match="scheme"):
            await fetch_image_url("file:///etc/passwd")
        mock_cs.assert_not_called()


@pytest.mark.asyncio
async def test_ftp_scheme_rejected() -> None:
    with patch("exo.api.adapters.chat_completions.create_http_session") as mock_cs:
        with pytest.raises(ValueError, match="scheme"):
            await fetch_image_url("ftp://example.com/image.jpg")
        mock_cs.assert_not_called()


# ---------------------------------------------------------------------------
# Rejection cases — metadata host blocklist
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_aws_metadata_rejected() -> None:
    with patch("exo.api.adapters.chat_completions.create_http_session") as mock_cs:
        with pytest.raises(ValueError, match="metadata"):
            await fetch_image_url("http://169.254.169.254/latest/meta-data/")
        mock_cs.assert_not_called()


@pytest.mark.asyncio
async def test_gcp_metadata_rejected() -> None:
    with patch("exo.api.adapters.chat_completions.create_http_session") as mock_cs:
        with pytest.raises(ValueError, match="metadata"):
            await fetch_image_url("http://metadata.google.internal/computeMetadata/v1/")
        mock_cs.assert_not_called()


@pytest.mark.asyncio
async def test_azure_metadata_rejected() -> None:
    with patch("exo.api.adapters.chat_completions.create_http_session") as mock_cs:
        with pytest.raises(ValueError, match="metadata"):
            await fetch_image_url("http://169.254.170.2/metadata/instance")
        mock_cs.assert_not_called()


# ---------------------------------------------------------------------------
# Rejection cases — literal private/loopback/link-local IPs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_private_ip_rfc1918_rejected() -> None:
    with patch("exo.api.adapters.chat_completions.create_http_session") as mock_cs:
        with pytest.raises(ValueError, match="Non-public IP"):
            await fetch_image_url("http://192.168.1.1/image.jpg")
        mock_cs.assert_not_called()


@pytest.mark.asyncio
async def test_private_ip_10_rejected() -> None:
    with patch("exo.api.adapters.chat_completions.create_http_session") as mock_cs:
        with pytest.raises(ValueError, match="Non-public IP"):
            await fetch_image_url("http://10.0.0.1/image.jpg")
        mock_cs.assert_not_called()


@pytest.mark.asyncio
async def test_loopback_rejected() -> None:
    with patch("exo.api.adapters.chat_completions.create_http_session") as mock_cs:
        with pytest.raises(ValueError, match="Non-public IP"):
            await fetch_image_url("http://127.0.0.1/internal")
        mock_cs.assert_not_called()


@pytest.mark.asyncio
async def test_link_local_non_metadata_rejected() -> None:
    """Link-local IPs not in metadata blocklist are still rejected by the IP check."""
    with patch("exo.api.adapters.chat_completions.create_http_session") as mock_cs:
        with pytest.raises(ValueError, match="Non-public IP"):
            await fetch_image_url("http://169.254.1.1/any")
        mock_cs.assert_not_called()


# ---------------------------------------------------------------------------
# Allowed cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_valid_https_url_succeeds() -> None:
    image_data = b"\x89PNG\r\n"
    session = _mock_session(image_data)
    with patch("exo.api.adapters.chat_completions.create_http_session", return_value=session):
        result = await fetch_image_url("https://example.com/image.png")
    assert result == base64.b64encode(image_data).decode("ascii")
    session.get.assert_called_once()


@pytest.mark.asyncio
async def test_public_ip_literal_allowed() -> None:
    """8.8.8.8 is a genuine public IP (Google DNS); should pass the IP check."""
    image_data = b"data"
    session = _mock_session(image_data)
    with patch("exo.api.adapters.chat_completions.create_http_session", return_value=session):
        result = await fetch_image_url("https://8.8.8.8/image.jpg")
    assert result == base64.b64encode(image_data).decode("ascii")
    session.get.assert_called_once()


@pytest.mark.asyncio
async def test_hostname_not_literal_ip_allowed_through() -> None:
    """A plain hostname is not a literal IP; ip_address() raises ValueError and check is skipped."""
    image_data = b"pixels"
    session = _mock_session(image_data)
    with patch("exo.api.adapters.chat_completions.create_http_session", return_value=session):
        result = await fetch_image_url("https://cdn.example.com/photo.jpg")
    assert result == base64.b64encode(image_data).decode("ascii")
    session.get.assert_called_once()
