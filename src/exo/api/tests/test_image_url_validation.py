from collections.abc import Sequence

import pytest

from exo.api.adapters.chat_completions import fetch_image_url, validate_image_url


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
    with pytest.raises(ValueError):
        await validate_image_url("https://nowhere.example/x.png", resolve=_resolver({}))


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
