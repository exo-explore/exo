import socket
import socketserver
import threading
from collections.abc import Callable
from typing import BinaryIO, cast

import msgspec
from loguru import logger

from exo.worker.disaggregated.protocol import (
    Header,
    read_frame,
    write_error,
    write_frame,
    write_header,
)


class PrefillRequest(msgspec.Struct):
    request_id: str = ""
    model_id: str = ""
    token_ids: list[int] = []
    start_pos: int = 0


_request_encoder = msgspec.msgpack.Encoder()
_request_decoder: msgspec.msgpack.Decoder[PrefillRequest] = msgspec.msgpack.Decoder(
    PrefillRequest
)


def write_request(stream: BinaryIO, job: PrefillRequest) -> None:
    write_frame(stream, _request_encoder.encode(job))


def read_request(stream: BinaryIO) -> PrefillRequest:
    payload = read_frame(stream)
    if not payload:
        raise ConnectionError("No request received")
    return _request_decoder.decode(payload)


ResolveHandler = Callable[[PrefillRequest, BinaryIO], bool]


def _send_error(wfile: BinaryIO, code: int, message: str) -> None:
    try:
        write_header(wfile, Header(num_layers=0, dtype="float32"))
        write_error(wfile, code=code, message=message)
    except Exception:
        pass


class _PrefillHandler(socketserver.StreamRequestHandler):
    def setup(self) -> None:
        super().setup()
        sock = cast(socket.socket, self.request)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)

    def handle(self) -> None:
        server = cast(PrefillServer, self.server)
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
        except Exception as e:
            logger.opt(exception=e).warning(
                f"Prefill resolve error for request_id={job.request_id}"
            )
            _send_error(wfile, 500, str(e))
            return
        if not picked_up:
            _send_error(
                wfile, 503, f"Prefill not picked up for request_id={job.request_id!r}"
            )


class PrefillServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    resolve: ResolveHandler

    def __init__(self, resolve: ResolveHandler, host: str, port: int) -> None:
        super().__init__((host, port), _PrefillHandler)
        self.resolve = resolve
        self._thread = threading.Thread(
            target=self.serve_forever, name="prefill-server"
        )
        self._thread.start()
        logger.info(f"Prefill server listening on {host}:{port}")

    def stop(self) -> None:
        self.shutdown()
        self.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
