import errno
import socket
from typing import cast

import anyio
from hypercorn.asyncio import serve  # pyright: ignore[reportUnknownVariableType]
from hypercorn.config import Config
from hypercorn.typing import ASGIFramework
from hypercorn.utils import LifespanTimeoutError, ShutdownError
from loguru import logger
from starlette.types import ASGIApp

from exo.shared.logging import InterceptLogger

API_BIND_HOST = "0.0.0.0"


class ApiPortInUseError(Exception):
    """The API's port is taken, most often by an exo that is already running."""

    def __init__(self, port: int) -> None:
        super().__init__(
            f"Port {port} is already in use - is exo already running? Use --api-port to choose another."
        )
        self.port = port


def listen_on_api_port(port: int) -> socket.socket:
    """A socket listening on the API's port, taken before exo starts anything else.

    A port that's in use then fails straight away with ApiPortInUseError, and the dashboard URL
    takes connections from the moment exo announces it.
    """
    listening_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Like hypercorn: a port left in TIME_WAIT by the last exo can be taken again straight away
    listening_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listening_socket.bind((API_BIND_HOST, port))
    except OSError as error:
        listening_socket.close()
        if error.errno == errno.EADDRINUSE:
            raise ApiPortInUseError(port) from error
        raise
    listening_socket.listen()
    return listening_socket


async def serve_on_socket(
    app: ASGIApp, listening_socket: socket.socket, shutdown: anyio.Event
) -> None:
    """Serve the app on the socket the node bound when it started, until `shutdown` is set.
    Hypercorn takes ownership of the socket."""
    config = Config()
    config.bind = [f"fd://{listening_socket.detach()}"]
    # nb: shared.logging needs updating if any of this changes
    config.accesslog = None
    config.errorlog = "-"
    config.logger_class = InterceptLogger
    # prevents hangs when mid-request and connection refuses to close
    config.graceful_timeout = 2  # seconds
    config.shutdown_timeout = 3  # seconds
    with anyio.CancelScope(shield=True):
        try:
            await serve(
                cast(ASGIFramework, app), config, shutdown_trigger=shutdown.wait
            )
            if not shutdown.is_set():
                raise ShutdownError(
                    "Server exited without shutdown trigger - exiting abnormally"
                )
        except LifespanTimeoutError as error:
            logger.warning(
                "Graceful server shutdown timed out, some connections forcebly closed"
            )
            logger.opt(exception=error).debug("")
