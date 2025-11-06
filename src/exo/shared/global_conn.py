# src/exo/shared/global_conn.py

import asyncio
import threading
from multiprocessing.connection import Connection

from exo.shared.types.worker.commands_runner import (
    RunnerMessage,
    RunnerResponse,
)


class AsyncConnection[SendT, RecvT]:
    """
    Async/sync wrapper around multiprocessing.Connection with thread-safe send.
    Use:
      - await send(...) from asyncio code
      - send_sync(...) from executor/background threads
    """

    def __init__(self, conn: Connection):
        self._conn = conn
        self._send_lock = threading.Lock()
        self._recv_lock = threading.Lock()

    # ---- sending ----
    async def send(self, obj: SendT) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._send_blocking, obj)

    def send_sync(self, obj: SendT) -> None:
        self._send_blocking(obj)

    def _send_blocking(self, obj: SendT) -> None:
        # Single critical section for the whole pickle frame
        with self._send_lock:
            self._conn.send(obj)

    # ---- receiving ----
    async def recv(self) -> RecvT:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._recv_blocking)

    def _recv_blocking(self) -> RecvT:
        # Not strictly needed in your parent, but safe if misused elsewhere
        with self._recv_lock:
            return self._conn.recv()  # type: ignore[no-any-return]

    async def poll(self, timeout: float | None = None) -> bool:
        return await asyncio.to_thread(self._conn.poll, timeout)

    def close(self) -> None:
        self._conn.close()


_conn: AsyncConnection[RunnerResponse, RunnerMessage] | None = None


def set_conn(c: AsyncConnection[RunnerResponse, RunnerMessage]) -> None:
    global _conn
    _conn = c


def get_conn() -> AsyncConnection[RunnerResponse, RunnerMessage]:
    if _conn is None:
        raise RuntimeError("Global conn has not been set yet")
    return _conn
