from _typeshed import Incomplete
from collections.abc import Callable as Callable
from ssl import SSLContext
from vllm.logger import init_logger as init_logger
from watchfiles import Change as Change

logger: Incomplete

class SSLCertRefresher:
    ssl: Incomplete
    key_path: Incomplete
    cert_path: Incomplete
    ca_path: Incomplete
    watch_ssl_cert_task: Incomplete
    watch_ssl_ca_task: Incomplete
    def __init__(
        self,
        ssl_context: SSLContext,
        key_path: str | None = None,
        cert_path: str | None = None,
        ca_path: str | None = None,
    ) -> None: ...
    def stop(self) -> None: ...
