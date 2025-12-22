"""Tests for HTTP session configuration."""

import pytest

from exo.worker.download.download_utils import create_http_session


@pytest.mark.asyncio
async def test_create_http_session_trusts_env():
    """Verify that create_http_session creates a session with trust_env=True.

    This ensures the session respects http_proxy, https_proxy, and no_proxy
    environment variables for network requests.
    """
    session = create_http_session()
    try:
        assert session.trust_env is True, (
            "ClientSession should have trust_env=True to respect proxy env variables"
        )
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_create_http_session_short_timeout_trusts_env():
    """Verify that create_http_session with short timeout also trusts env."""
    session = create_http_session(timeout_profile="short")
    try:
        assert session.trust_env is True
    finally:
        await session.close()
