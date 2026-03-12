from _typeshed import Incomplete
from collections.abc import Awaitable
from starlette.types import (
    ASGIApp as ASGIApp,
    Message as Message,
    Receive as Receive,
    Scope as Scope,
    Send as Send,
)

class WebSocketMetricsMiddleware:
    app: Incomplete
    def __init__(self, app: ASGIApp) -> None: ...
    def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> Awaitable[None]: ...
