import threading
import time

import pytest

import exo.utils.stall_watchdog as stall_watchdog_module
from exo.utils.stall_watchdog import StallWatchdog


@pytest.fixture
def captured_exit(monkeypatch: pytest.MonkeyPatch) -> threading.Event:
    tripped = threading.Event()

    def fake_exit(code: int) -> None:
        assert code == stall_watchdog_module.STALL_EXIT_CODE
        tripped.set()

    def fake_dump_traceback(**_kwargs: object) -> None:
        pass

    monkeypatch.setattr(stall_watchdog_module.os, "_exit", fake_exit)
    monkeypatch.setattr(
        stall_watchdog_module.faulthandler, "dump_traceback", fake_dump_traceback
    )
    return tripped


def test_trips_when_no_progress(captured_exit: threading.Event) -> None:
    with StallWatchdog(0.1, "test operation"):
        assert captured_exit.wait(timeout=2.0)


def test_kick_defers_the_trip(captured_exit: threading.Event) -> None:
    with StallWatchdog(0.5, "test operation") as watchdog:
        for _ in range(5):
            time.sleep(0.1)
            watchdog.kick()
        assert not captured_exit.is_set()


def test_close_stops_watching(captured_exit: threading.Event) -> None:
    watchdog = StallWatchdog(0.1, "test operation")
    watchdog.close()
    assert not captured_exit.wait(timeout=0.5)


def test_zero_timeout_disables(captured_exit: threading.Event) -> None:
    with StallWatchdog(0, "test operation"):
        assert not captured_exit.wait(timeout=0.3)
