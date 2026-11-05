"""A client's request streams end when its HTTP exchange does, however it ends.

A response's body closes its request's stream when it ends, and closing it cancels the request
on every rank. But a client that leaves before its response starts (while the request is still
being handed to the ranks, say) never starts the body, and nothing else would close the stream:
the request would run to its end for no one.
"""

from contextvars import ContextVar
from typing import final

from starlette.types import ASGIApp, Receive, Scope, Send

from exo.api.serving_client import ServeStream

# The streams opened while handling the current HTTP request
_opened: ContextVar[list[ServeStream] | None] = ContextVar(
    "exo_opened_streams", default=None
)


def close_with_exchange(stream: ServeStream) -> None:
    """Close this stream when the HTTP exchange that opened it ends"""
    if (opened := _opened.get()) is not None:
        opened.append(stream)


@final
class ClosesStreams:
    """ASGI middleware that closes the streams each HTTP request opened once it is answered
    (a no-op for a stream that has ended), or once its client has gone"""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        opened: list[ServeStream] = []
        token = _opened.set(opened)
        try:
            await self.app(scope, receive, send)
        finally:
            _opened.reset(token)
            for stream in opened:
                stream.close()
