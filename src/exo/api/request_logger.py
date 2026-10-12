from typing import cast

from loguru import logger
from starlette.types import ASGIApp, Receive, Scope, Send


class RequestLogger:
    """Logs each HTTP request the API receives.

    A plain ASGI middleware rather than FastAPI's `@app.middleware("http")`, which wraps every
    route in Starlette's BaseHTTPMiddleware: that raises "No response returned." (logged as an
    ASGI error with a full traceback) whenever a client disconnects before a non-streaming
    response has started, which is every time a client gives up on a non-streaming request.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            logger.debug(
                f"API request: {cast(str, scope['method'])} {cast(str, scope['path'])}"
            )
        await self.app(scope, receive, send)
