import errno
import socket

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
