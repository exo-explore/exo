import aiohttp
import requests
from _typeshed import Incomplete
from collections.abc import Mapping
from pathlib import Path

class HTTPConnection:
    reuse_client: Incomplete
    def __init__(self, *, reuse_client: bool = True) -> None: ...
    def get_sync_client(self) -> requests.Session: ...
    async def get_async_client(self) -> aiohttp.ClientSession: ...
    def get_response(
        self,
        url: str,
        *,
        stream: bool = False,
        timeout: float | None = None,
        extra_headers: Mapping[str, str] | None = None,
        allow_redirects: bool = True,
    ): ...
    async def get_async_response(
        self,
        url: str,
        *,
        timeout: float | None = None,
        extra_headers: Mapping[str, str] | None = None,
        allow_redirects: bool = True,
    ): ...
    def get_bytes(
        self, url: str, *, timeout: float | None = None, allow_redirects: bool = True
    ) -> bytes: ...
    async def async_get_bytes(
        self, url: str, *, timeout: float | None = None, allow_redirects: bool = True
    ) -> bytes: ...
    def get_text(self, url: str, *, timeout: float | None = None) -> str: ...
    async def async_get_text(
        self, url: str, *, timeout: float | None = None
    ) -> str: ...
    def get_json(self, url: str, *, timeout: float | None = None) -> str: ...
    async def async_get_json(
        self, url: str, *, timeout: float | None = None
    ) -> str: ...
    def download_file(
        self,
        url: str,
        save_path: Path,
        *,
        timeout: float | None = None,
        chunk_size: int = 128,
    ) -> Path: ...
    async def async_download_file(
        self,
        url: str,
        save_path: Path,
        *,
        timeout: float | None = None,
        chunk_size: int = 128,
    ) -> Path: ...

global_http_connection: Incomplete
