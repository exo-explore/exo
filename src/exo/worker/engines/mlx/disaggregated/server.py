import json
import socket
import socketserver
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, BinaryIO, cast

from loguru import logger

from exo.worker.engines.mlx.disaggregated.protocol import (
    make_header,
    write_error,
    write_header,
)


@dataclass
class PrefillJob:
    request_id: str
    model_id: str
    token_ids: list[int]
    start_pos: int


class PrefillPayloadLookup:
    _lock: threading.Lock
    _payloads: dict[str, bytes]

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._payloads = {}

    def register(self, request_id: str, payload: bytes) -> None:
        with self._lock:
            self._payloads[request_id] = payload

    def pop(self, request_id: str) -> bytes | None:
        with self._lock:
            return self._payloads.pop(request_id, None)

    def drop(self, request_id: str) -> None:
        with self._lock:
            self._payloads.pop(request_id, None)


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


ResolveHandler = Callable[[PrefillJob], bytes | None]


class PrefillServer:
    _thread: threading.Thread | None
    _server: socketserver.TCPServer | None
    _resolve: ResolveHandler
    _host: str
    _requested_port: int
    _bound_port: int

    def __init__(
        self,
        resolve: ResolveHandler,
        host: str = "0.0.0.0",
        port: int = 0,
    ) -> None:
        self._thread = None
        self._server = None
        self._resolve = resolve
        self._host = host
        self._requested_port = port
        self._bound_port = 0

    @property
    def port(self) -> int:
        return self._bound_port

    def start(self) -> int:
        if self._server is not None:
            return self._bound_port

        resolve = self._resolve

        def _send_error(wfile: BinaryIO, code: int, message: str) -> None:
            # Client reads a header first, so errors must be wrapped.
            try:
                write_header(wfile, make_header(num_layers=0, dtype="float32"))
                write_error(wfile, code=code, message=message)
            except Exception:
                pass

        class _Handler(socketserver.StreamRequestHandler):
            def setup(self) -> None:
                super().setup()
                req_sock: socket.socket = cast(socket.socket, self.request)
                req_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                req_sock.setsockopt(
                    socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024
                )

            def handle(self) -> None:
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
                    payload = resolve(job)
                except Exception as exc:  # noqa: BLE001
                    logger.opt(exception=True).warning(
                        f"Prefill resolve error for request_id={job.request_id}"
                    )
                    _send_error(wfile, 500, str(exc))
                    return
                if payload is None:
                    _send_error(
                        wfile,
                        503,
                        f"No payload ready for request_id={job.request_id!r}",
                    )
                    return
                try:
                    wfile.write(payload)
                    wfile.flush()
                except Exception:
                    logger.opt(exception=True).warning(
                        f"Failed to write payload for request_id={job.request_id}"
                    )

        class _ThreadedServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self._server = _ThreadedServer((self._host, self._requested_port), _Handler)
        sock = cast(socket.socket, cast(Any, self._server).socket)
        addr = cast(tuple[str, int], sock.getsockname())
        self._bound_port = int(addr[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="prefill-server", daemon=True
        )
        self._thread.start()
        logger.info(f"Prefill server listening on {self._host}:{self._bound_port}")
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
