from _typeshed import Incomplete
from collections.abc import Awaitable
from starlette.types import (
    ASGIApp as ASGIApp,
    Receive as Receive,
    Scope as Scope,
    Send as Send,
)

def get_scaling_elastic_ep(): ...
def set_scaling_elastic_ep(value) -> None: ...

class ScalingMiddleware:
    app: Incomplete
    def __init__(self, app: ASGIApp) -> None: ...
    def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> Awaitable[None]: ...
