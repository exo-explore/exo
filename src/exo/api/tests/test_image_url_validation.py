import asyncio
import socket
import sys
from collections.abc import Sequence
from typing import cast
from unittest.mock import patch

import pytest
from fastapi import Request

from exo.api.adapters.chat_completions import (
    ImageUrlRejectedError,
    fetch_image_url,
    validate_image_url,
)
from exo.api.adapters.claude import handle_image_block
from exo.api.types.api import ErrorResponse
from exo.api.types.claude_api import ClaudeImageBlock, ClaudeImageSource

SocketAddressInfo = tuple[int, int, int, str, tuple[str, int]]


async def _resolve_never(host: str) -> Sequence[str]:
    raise AssertionError(f"resolver should not be called for {host!r}")


def _resolver(mapping: dict[str, list[str]]):
    async def resolve(host: str) -> Sequence[str]:
        return mapping.get(host, [])

    return resolve


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/cat.png",
        "data:image/png;base64,AAAA",
        "http:///no-host",
    ],
)
async def test_rejects_non_http_or_hostless_urls(url: str) -> None:
    with pytest.raises(ValueError):
        await validate_image_url(url, resolve=_resolve_never)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:52415/state",
        "http://[::1]/x",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/img.png",
        "http://192.168.1.21/img.png",
        "http://172.16.3.4/img.png",
        "http://0.0.0.0/x",
        "http://224.0.0.1/x",
        "http://[fe80::1]/x",
        "http://[fd00::1]/x",
        "http://[::ffff:127.0.0.1]/x",
    ],
)
async def test_rejects_literal_non_public_addresses(url: str) -> None:
    with pytest.raises(ValueError):
        await validate_image_url(url, resolve=_resolve_never)


async def test_rejects_hostname_resolving_to_private_address() -> None:
    resolve = _resolver({"internal.example": ["10.1.2.3"]})
    with pytest.raises(ValueError):
        await validate_image_url("https://internal.example/x.png", resolve=resolve)


async def test_rejects_hostname_with_mixed_public_and_private_records() -> None:
    resolve = _resolver({"rebind.example": ["93.184.216.34", "127.0.0.1"]})
    with pytest.raises(ValueError):
        await validate_image_url("https://rebind.example/x.png", resolve=resolve)


async def test_rejects_unresolvable_hostname() -> None:
    with pytest.raises(ImageUrlRejectedError):
        await validate_image_url("https://nowhere.example/x.png", resolve=_resolver({}))


async def test_resolver_failure_is_a_rejection() -> None:
    async def resolve(host: str) -> Sequence[str]:
        raise socket.gaierror(f"Name or service not known: {host}")

    with pytest.raises(ImageUrlRejectedError):
        await validate_image_url("https://nowhere.example/x.png", resolve=resolve)


@pytest.mark.parametrize("url", ["http://[::1", "http://cdn.example:bad/x.png"])
async def test_malformed_url_is_a_rejection(url: str) -> None:
    with pytest.raises(ImageUrlRejectedError):
        await validate_image_url(
            url, resolve=_resolver({"cdn.example": ["93.184.216.34"]})
        )


async def test_dns_resolution_has_a_deadline() -> None:
    deadline = asyncio.timeout

    def immediate_deadline(_delay: float | None) -> asyncio.Timeout:
        return deadline(0)

    async def resolve(_host: str) -> Sequence[str]:
        await asyncio.sleep(60)
        return ["93.184.216.34"]

    with (
        patch(
            "exo.api.adapters.chat_completions.asyncio.timeout",
            side_effect=immediate_deadline,
        ),
        pytest.raises(ImageUrlRejectedError),
    ):
        await asyncio.wait_for(
            validate_image_url("https://cdn.example/x.png", resolve=resolve), 0.1
        )


async def test_accepts_public_hostname_and_literal() -> None:
    resolve = _resolver(
        {"cdn.example": ["93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"]}
    )
    await validate_image_url("https://cdn.example/cat.png", resolve=resolve)
    await validate_image_url("https://93.184.216.34/cat.png", resolve=_resolve_never)


async def test_fetch_image_url_refuses_before_any_network_call() -> None:
    # A literal loopback address never reaches the resolver or the socket.
    with pytest.raises(ValueError):
        await fetch_image_url("http://127.0.0.1:1/never-opened")


async def test_claude_image_block_does_not_drop_a_rejected_url() -> None:
    block = ClaudeImageBlock(
        source=ClaudeImageSource(type="url", url="http://127.0.0.1:1/x.png")
    )
    with pytest.raises(ImageUrlRejectedError):
        await handle_image_block(block)


@pytest.mark.parametrize("proxy", ["", "http://93.184.216.100:8080"])
async def test_fetch_uses_validated_addresses_when_dns_changes(proxy: str) -> None:
    loop = asyncio.get_running_loop()
    public = [
        (
            socket.AF_INET,
            socket.SOCK_STREAM,
            socket.IPPROTO_TCP,
            "",
            ("93.184.216.34", 80),
        )
    ]
    rebound = [
        (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", 80))
    ]
    attempted_addresses: list[SocketAddressInfo] = []

    class ConnectionCapturedError(Exception):
        pass

    async def capture_connection(
        *, addr_infos: Sequence[SocketAddressInfo], **_kwargs: object
    ) -> socket.socket:
        attempted_addresses.extend(addr_infos)
        raise ConnectionCapturedError

    with (
        patch.object(loop, "getaddrinfo", side_effect=[public, rebound]),
        patch("aiohappyeyeballs.start_connection", side_effect=capture_connection),
        patch.dict("os.environ", {"HTTP_PROXY": proxy, "HTTPS_PROXY": proxy}),
        pytest.raises(ConnectionCapturedError),
    ):
        await fetch_image_url("http://rebind.example/image.png")

    assert [address[4] for address in attempted_addresses] == [("93.184.216.34", 80)]


@pytest.mark.parametrize("status", [200, 302])
async def test_public_image_fetch_preserves_host_and_refuses_redirects(
    status: int,
) -> None:
    requests: list[bytes] = []

    async def respond(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        requests.append(await reader.readuntil(b"\r\n\r\n"))
        writer.write(
            f"HTTP/1.1 {status} Test\r\nContent-Length: 5\r\n"
            "Location: http://127.0.0.1/private\r\nConnection: close\r\n\r\nimage".encode()
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(respond, "127.0.0.1", 0)
    assert server.sockets
    local_address = cast(tuple[str, int], server.sockets[0].getsockname())
    loop = asyncio.get_running_loop()

    async def connect_to_fixture(
        *, addr_infos: Sequence[SocketAddressInfo], **_kwargs: object
    ) -> socket.socket:
        # Keep aiohttp's real resolver/request/response flow; only substitute
        # the external TCP peer with a deterministic local HTTP fixture.
        assert [info[4] for info in addr_infos] == [("93.184.216.34", 80)]
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connection.setblocking(False)
        try:
            await loop.sock_connect(connection, local_address)
        except BaseException:
            connection.close()
            raise
        return connection

    public = [
        (
            socket.AF_INET,
            socket.SOCK_STREAM,
            socket.IPPROTO_TCP,
            "",
            ("93.184.216.34", 80),
        )
    ]
    async with server:
        with (
            patch.object(loop, "getaddrinfo", return_value=public),
            patch("aiohappyeyeballs.start_connection", side_effect=connect_to_fixture),
            patch.dict("os.environ", {"HTTP_PROXY": "", "HTTPS_PROXY": ""}),
        ):
            if status == 200:
                assert (
                    await fetch_image_url("http://cdn.example/image.png") == "aW1hZ2U="
                )
            else:
                with pytest.raises(ImageUrlRejectedError):
                    await fetch_image_url("http://cdn.example/image.png")

    assert len(requests) == 1
    assert b"Host: cdn.example\r\n" in requests[0]


@pytest.mark.skipif(sys.platform == "win32", reason="API imports POSIX resource module")
async def test_rejected_image_url_is_a_400() -> None:
    from exo.api.main import API

    api = object.__new__(API)
    request = Request({"type": "http", "method": "POST", "path": "/", "headers": []})

    response = await api.image_url_rejected_handler(
        request, ImageUrlRejectedError("Refusing to fetch image from 127.0.0.1")
    )

    assert response.status_code == 400
    body = ErrorResponse.model_validate_json(bytes(response.body))
    assert body.error.code == 400
    assert "127.0.0.1" in body.error.message
