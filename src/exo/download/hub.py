"""Hugging Face Hub over HTTP.

Resolves a branch to a commit, lists a commit's files, and opens a file at an
offset. Every failed request comes out as one of the errors below, each with a
short message a user can act on. No retries and no disk access here.
"""

from __future__ import annotations

import asyncio
import os
import ssl
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import final
from urllib.parse import quote, urlsplit

import aiohttp
import certifi
from loguru import logger
from pydantic import BaseModel, TypeAdapter, ValidationError

from exo.download.huggingface_utils import get_auth_headers, get_hf_endpoint
from exo.shared.types.common import ModelId
from exo.utils.pydantic_ext import FrozenModel


class RepoFile(FrozenModel):
    """A file of a repo at a commit, as the Hub lists it."""

    path: str
    size: int
    # sha256 of the content for LFS files; None for files stored in git
    sha256: str | None = None
    # git blob sha1, for files stored in git
    git_sha1: str = ""


class HubError(Exception):
    """A request to the Hub failed. ``message`` is short and meant for users."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class TransientError(HubError):
    """Worth retrying: the network or the Hub may recover."""


class RateLimitedError(TransientError):
    def __init__(self, message: str, retry_after: float | None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class PermanentError(HubError):
    """Retrying soon won't help: the user has to do something first."""


class NotFoundError(PermanentError):
    """The repo, revision or file isn't on the Hub."""


class RevisionGoneError(NotFoundError):
    """The commit a download was pinned to no longer exists upstream."""

    def __init__(self, message: str, repo: ModelId) -> None:
        super().__init__(message)
        self.repo = repo


def new_session(
    *,
    connect_timeout: float,
    idle_timeout: float,
    total_timeout: float | None = None,
    auto_decompress: bool = False,
) -> aiohttp.ClientSession:
    """A session honouring SSL_CERT_FILE and the usual proxy variables.

    ``idle_timeout`` is the longest wait for the next bytes; there is no limit on
    a whole request unless ``total_timeout`` is given.
    """
    ssl_context = ssl.create_default_context(
        cafile=os.getenv("SSL_CERT_FILE") or certifi.where()
    )
    return aiohttp.ClientSession(
        auto_decompress=auto_decompress,
        connector=aiohttp.TCPConnector(ssl=ssl_context),
        # HTTP(S)_PROXY in either case, and NO_PROXY
        trust_env=True,
        timeout=aiohttp.ClientTimeout(
            total=total_timeout,
            sock_connect=connect_timeout,
            sock_read=idle_timeout,
        ),
    )


def parse_retry_after(headers: Mapping[str, str]) -> float | None:
    """Seconds to wait from HF's ``RateLimit: "api";r=0;t=52`` header, else ``Retry-After``."""
    for part in (headers.get("RateLimit") or "").split(";"):
        key, _, value = part.strip().partition("=")
        if key == "t":
            try:
                return float(value)
            except ValueError:
                return None
    try:
        return float(headers.get("Retry-After") or "")
    except ValueError:
        return None


def describe_network_error(error: BaseException) -> str:
    """A few words for a network failure, without URLs or reprs."""
    host = urlsplit(get_hf_endpoint()).hostname or "the Hub"
    match error:
        case TimeoutError():
            return f"{host} stopped responding"
        case aiohttp.ClientConnectorError():
            return f"can't connect to {host}"
        case (
            aiohttp.ClientPayloadError()
            | aiohttp.ServerDisconnectedError()
            | aiohttp.ClientOSError()
            | ConnectionError()
        ):
            return f"the connection to {host} was cut"
        case _:
            return f"network error talking to {host}"


@asynccontextmanager
async def network_errors() -> AsyncIterator[None]:
    """Turn network exceptions into :class:`TransientError`."""
    try:
        yield
    except (aiohttp.ClientError, TimeoutError, ConnectionError) as error:
        logger.debug(f"Hub request failed: {error!r}")
        raise TransientError(describe_network_error(error)) from error


def _raise_for_status(
    response: aiohttp.ClientResponse, repo: ModelId, path: str
) -> None:
    status = response.status
    if status < 400 or status == 416:
        return
    code = response.headers.get("X-Error-Code", "")
    if status == 429:
        raise RateLimitedError(
            "Hugging Face is rate limiting this machine",
            parse_retry_after(response.headers),
        )
    if status >= 500:
        raise TransientError(f"Hugging Face server error {status}")
    if response.url.host != urlsplit(get_hf_endpoint()).hostname:
        # the storage server a file redirects to: an expired link, most likely
        raise TransientError(f"the file server refused {path} ({status})")
    if code == "RevisionNotFound":
        raise RevisionGoneError(
            f"The revision exo was downloading is gone from {repo}.", repo
        )
    if code == "GatedRepo" and status == 401:
        raise PermanentError(
            f"{repo} is gated. Accept its terms at huggingface.co/{repo}, then set HF_TOKEN."
        )
    if status == 403:
        raise PermanentError(
            f"No access to {repo}. Accept its terms at huggingface.co/{repo} "
            "with the account of your HF_TOKEN."
        )
    # Hugging Face answers anonymous requests for unknown repos with a bare 401
    if code == "RepoNotFound" or status == 401 or (status == 404 and not path):
        raise NotFoundError(
            f"{repo} was not found on Hugging Face. If it is private, set HF_TOKEN."
        )
    if status == 404:
        raise NotFoundError(f"{path} is listed in {repo} but was not found (404).")
    raise PermanentError(f"Hugging Face refused to send {path or repo} ({status}).")


def is_safe_path(path: str) -> bool:
    """A relative path that stays inside the folder it is joined to."""
    parts = path.split("/")
    return (
        bool(path)
        and not path.startswith("/")
        and "\\" not in path
        and "\0" not in path
        and all(part not in ("", ".", "..") for part in parts)
    )


class _Lfs(BaseModel):
    oid: str


class _TreeEntry(BaseModel):
    type: str
    path: str
    size: int = 0
    oid: str = ""
    lfs: _Lfs | None = None


class _Revision(BaseModel):
    sha: str


_tree_entries: TypeAdapter[list[_TreeEntry]] = TypeAdapter(list[_TreeEntry])


@final
class Hub:
    """A connection to the Hub; use ``async with Hub.connect(...) as hub``."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        headers: dict[str, str],
        metadata_timeout: float,
    ) -> None:
        self._session = session
        self._headers = headers
        self._metadata_timeout = metadata_timeout
        self._endpoint = get_hf_endpoint().rstrip("/")

    @classmethod
    @asynccontextmanager
    async def connect(
        cls, *, connect_timeout: float, idle_timeout: float, metadata_timeout: float
    ) -> AsyncIterator[Hub]:
        headers = {**(await get_auth_headers()), "Accept-Encoding": "identity"}
        async with new_session(
            connect_timeout=connect_timeout, idle_timeout=idle_timeout
        ) as session:
            yield cls(session, headers, metadata_timeout)

    async def _get_json(self, url: str, repo: ModelId) -> tuple[bytes, str | None]:
        """The body of a metadata request and the URL of its next page, if any."""
        async with (
            network_errors(),
            asyncio.timeout(self._metadata_timeout),
            self._session.get(url, headers=self._headers) as response,
        ):
            _raise_for_status(response, repo, "")
            body = await response.read()
            next_link = response.links.get("next")
            next_url = next_link.get("url") if next_link is not None else None
            return body, None if next_url is None else str(next_url)

    async def resolve_commit(self, repo: ModelId, revision: str = "main") -> str:
        body, _ = await self._get_json(
            f"{self._endpoint}/api/models/{repo}/revision/{quote(revision, safe='')}",
            repo,
        )
        try:
            return _Revision.model_validate_json(body).sha
        except ValidationError as error:
            raise TransientError("Hugging Face sent an unreadable answer") from error

    async def list_files(self, repo: ModelId, commit: str) -> list[RepoFile]:
        """Every file of ``repo`` at ``commit``, following the listing's pages."""
        files: list[RepoFile] = []
        url: str | None = (
            f"{self._endpoint}/api/models/{repo}/tree/{commit}?recursive=true"
        )
        while url is not None:
            body, url = await self._get_json(url, repo)
            try:
                entries = _tree_entries.validate_json(body)
            except ValidationError as error:
                raise TransientError(
                    "Hugging Face sent an unreadable file list"
                ) from error
            for entry in entries:
                if entry.type != "file":
                    continue
                if not is_safe_path(entry.path):
                    raise PermanentError(
                        f"{repo} lists a file exo won't write: {entry.path!r}."
                    )
                files.append(
                    RepoFile(
                        path=entry.path,
                        size=entry.size,
                        sha256=entry.lfs.oid if entry.lfs is not None else None,
                        git_sha1=entry.oid,
                    )
                )
        return files

    @asynccontextmanager
    async def open(
        self, repo: ModelId, revision: str, path: str, offset: int = 0
    ) -> AsyncIterator[aiohttp.ClientResponse]:
        """GET a file from ``offset``. A 416 is returned, not raised."""
        headers = dict(self._headers)
        if offset > 0:
            headers["Range"] = f"bytes={offset}-"
        url = f"{self._endpoint}/{repo}/resolve/{revision}/{quote(path)}"
        async with (
            network_errors(),
            self._session.get(url, headers=headers) as response,
        ):
            _raise_for_status(response, repo, path)
            yield response


async def chunks(response: aiohttp.ClientResponse, size: int) -> AsyncGenerator[bytes]:
    """The response body in chunks of up to ``size`` bytes, network errors translated."""
    async with network_errors():
        async for chunk in response.content.iter_chunked(size):
            yield chunk
