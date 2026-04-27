import socket
import socketserver
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, BinaryIO, cast

import msgspec
from loguru import logger

from exo.worker.disaggregated.protocol import (
    Header,
    read_frame,
    write_error,
    write_frame,
    write_header,
)


class PrefillJob(msgspec.Struct):
    request_id: str = ""
    model_id: str = ""
    token_ids: list[int] = []
    start_pos: int = 0


_request_encoder = msgspec.msgpack.Encoder()
_request_decoder: msgspec.msgpack.Decoder[PrefillJob] = msgspec.msgpack.Decoder(
    PrefillJob
)


def write_request(stream: BinaryIO, job: PrefillJob) -> None:
    write_frame(stream, _request_encoder.encode(job))


def read_request(stream: BinaryIO) -> PrefillJob:
    payload = read_frame(stream)
    if not payload:
        raise ConnectionError("No request received")
    return _request_decoder.decode(payload)


ResolveHandler = Callable[[PrefillJob, BinaryIO], bool]


def _send_error(wfile: BinaryIO, code: int, message: str) -> None:
    try:
        write_header(wfile, Header(num_layers=0, dtype="float32"))
        write_error(wfile, code=code, message=message)
    except Exception:
        pass


class _PrefillTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    resolve: ResolveHandler


class _PrefillHandler(socketserver.StreamRequestHandler):
    def setup(self) -> None:
        super().setup()
        sock = cast(socket.socket, self.request)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)

    def handle(self) -> None:
        server = cast(_PrefillTCPServer, self.server)
        wfile: BinaryIO = cast(BinaryIO, cast(object, self.wfile))
        rfile: BinaryIO = cast(BinaryIO, cast(object, self.rfile))
        try:
            job = read_request(rfile)
        except ConnectionError:
            return
        except (msgspec.DecodeError, ValueError) as exc:
            _send_error(wfile, 400, f"Bad request: {exc}")
            return
        try:
            picked_up = server.resolve(job, wfile)
        except Exception as exc:  # noqa: BLE001
            logger.opt(exception=True).warning(
                f"Prefill resolve error for request_id={job.request_id}"
            )
            _send_error(wfile, 500, str(exc))
            return
        if not picked_up:
            _send_error(
                wfile, 503, f"Prefill not picked up for request_id={job.request_id!r}"
            )


@dataclass
class PrefillServer:
    resolve: ResolveHandler
    host: str = "0.0.0.0"
    port: int = 0
    _server: _PrefillTCPServer | None = field(default=None, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def start(self) -> int:
        if self._server is not None:
            sock = cast(socket.socket, cast(Any, self._server).socket)
            return int(cast(tuple[str, int], sock.getsockname())[1])

        self._server = _PrefillTCPServer((self.host, self.port), _PrefillHandler)
        self._server.resolve = self.resolve
        sock = cast(socket.socket, cast(Any, self._server).socket)
        bound_port = int(cast(tuple[str, int], sock.getsockname())[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="prefill-server"
        )
        self._thread.start()
        logger.info(f"Prefill server listening on {self.host}:{bound_port}")
        return bound_port

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
