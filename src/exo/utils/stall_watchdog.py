import faulthandler
import os
import sys
import threading
from types import TracebackType
from typing import Self, final

from loguru import logger

STALL_EXIT_CODE = 70


@final
class StallWatchdog:
    """Terminate the process when a monitored operation stops making progress.

    A daemon thread trips after `timeout_seconds` elapse without a `kick()`.
    Used around distributed prefill, where a cross-rank scheduling stall can
    wedge every rank indefinitely: converting the hang into a runner death
    lets the supervisor tear the instance down and surface an error instead
    of hanging the request forever. All thread stacks are dumped first so the
    stall site is preserved for diagnosis.

    A `timeout_seconds` of 0 disables the watchdog entirely.
    """

    def __init__(self, timeout_seconds: float, description: str) -> None:
        self._timeout_seconds = timeout_seconds
        self._description = description
        self._kicked = threading.Event()
        self._closed = threading.Event()
        self._thread: threading.Thread | None = None
        if timeout_seconds > 0:
            self._thread = threading.Thread(
                target=self._watch, name="stall-watchdog", daemon=True
            )
            self._thread.start()

    def kick(self) -> None:
        """Record progress, granting the operation another full timeout window."""
        self._kicked.set()

    def close(self) -> None:
        self._closed.set()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _watch(self) -> None:
        while True:
            self._kicked.clear()
            if self._closed.wait(self._timeout_seconds):
                return
            if self._kicked.is_set():
                continue
            logger.critical(
                f"{self._description} made no progress for "
                f"{self._timeout_seconds:.0f}s - terminating runner"
            )
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
            os._exit(STALL_EXIT_CODE)
