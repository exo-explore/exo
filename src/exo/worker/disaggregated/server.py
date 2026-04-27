import json
import socket
import socketserver
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, BinaryIO, cast

from loguru import logger

from exo.worker.disaggregated.protocol import (
    Header,
    write_error,
    write_header,
)


@dataclass
class PrefillJob:
    request_id: str
    model_id: str
    token_ids: list[int]
    start_pos: int


ResolveHandler = Callable[[PrefillJob], bytes | None]


def _parse_request_line(line: bytes) -> PrefillJob:
    parsed = cast(object, json.loads(line.decode("utf-8")))
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
    d = cast(dict[str, object], parsed)

    def _str(key: str, default: str = "") -> str:
        v = d.get(key, default)
        return v if isinstance(v, str) else default

    def _int(key: str, default: int = 0) -> int:
        v = d.get(key, default)
        return v if isinstance(v, int) else default

    raw_tokens = d.get("token_ids")
    if not isinstance(raw_tokens, list):
        raise ValueError("Missing token_ids in request")
    token_ids: list[int] = []
    for t in cast(list[object], raw_tokens):
        if not isinstance(t, int):
            raise ValueError("token_ids must be list[int]")
        token_ids.append(t)

    return PrefillJob(
        request_id=_str("request_id"),
        model_id=_str("model"),
        token_ids=token_ids,
        start_pos=_int("start_pos"),
    )


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
        line: bytes = self.rfile.readline()
        if not line:
            return
        try:
            job = _parse_request_line(line)
        except (ValueError, json.JSONDecodeError) as exc:
            _send_error(wfile, 400, f"Bad request: {exc}")
            return
        try:
            payload = server.resolve(job)
        except Exception as exc:  # noqa: BLE001
            logger.opt(exception=True).warning(
                f"Prefill resolve error for request_id={job.request_id}"
            )
            _send_error(wfile, 500, str(exc))
            return
        if payload is None:
            _send_error(
                wfile, 503, f"No payload ready for request_id={job.request_id!r}"
            )
            return
        try:
            wfile.write(payload)
            wfile.flush()
        except Exception:
            logger.opt(exception=True).warning(
                f"Failed to write payload for request_id={job.request_id}"
            )


@dataclass
class PrefillServer:
    resolve: ResolveHandler
    host: str = "0.0.0.0"
    port: int = 0
    _server: _PrefillTCPServer | None = field(default=None, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _bound_port: int = field(default=0, init=False, repr=False)

    @property
    def bound_port(self) -> int:
        return self._bound_port

    def start(self) -> int:
        if self._server is not None:
            return self._bound_port

        self._server = _PrefillTCPServer((self.host, self.port), _PrefillHandler)
        self._server.resolve = self.resolve
        sock = cast(socket.socket, cast(Any, self._server).socket)
        addr = cast(tuple[str, int], sock.getsockname())
        self._bound_port = int(addr[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="prefill-server", daemon=True
        )
        self._thread.start()
        logger.info(f"Prefill server listening on {self.host}:{self._bound_port}")
        return self._bound_port

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._bound_port = 0
