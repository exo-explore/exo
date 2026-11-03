import threading
import time
from pathlib import Path

import pytest

from exo.utils import banner


def _first_run_at_a_local_desktop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Path:
    marker = tmp_path / ".dashboard_opened"
    monkeypatch.setattr(banner, "_FIRST_RUN_MARKER", marker)
    monkeypatch.delenv("EXO_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    return marker


def _recording_open(calls: list[str]):
    def recording_open(url: str) -> bool:
        calls.append(url)
        return True

    return recording_open


def test_a_browser_that_never_answers_does_not_block_the_banner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """On first run the dashboard is opened in a browser. webbrowser.open can block for minutes when no browser
    answers, and the banner is printed from the node's event loop, so it must return without waiting for it."""
    release = threading.Event()
    opened_on: list[str] = []

    def never_answering_open(url: str) -> bool:
        opened_on.append(threading.current_thread().name)
        release.wait(timeout=10)
        return True

    marker = _first_run_at_a_local_desktop(monkeypatch, tmp_path)
    monkeypatch.setattr(banner.webbrowser, "open", never_answering_open)

    start = time.monotonic()
    banner.print_startup_banner(52415)
    elapsed = time.monotonic() - start
    release.set()

    assert elapsed < 1.0
    assert marker.exists()
    deadline = time.monotonic() + 5
    while not opened_on and time.monotonic() < deadline:
        time.sleep(0.01)
    assert opened_on and opened_on[0] != threading.main_thread().name


def test_the_browser_is_opened_only_on_first_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    marker = _first_run_at_a_local_desktop(monkeypatch, tmp_path)
    marker.touch()
    calls: list[str] = []
    monkeypatch.setattr(banner.webbrowser, "open", _recording_open(calls))

    banner.print_startup_banner(52415)
    time.sleep(0.1)

    assert calls == []


def test_no_browser_is_opened_from_an_ssh_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Over ssh a browser would open on whoever is at the machine's screen, not in front of the person running
    exo. The first run still counts as done."""
    marker = _first_run_at_a_local_desktop(monkeypatch, tmp_path)
    monkeypatch.setenv("SSH_CONNECTION", "client 50000 server 22")
    calls: list[str] = []
    monkeypatch.setattr(banner.webbrowser, "open", _recording_open(calls))

    banner.print_startup_banner(52415)
    time.sleep(0.1)

    assert calls == []
    assert marker.exists()


def test_no_browser_is_opened_on_linux_without_a_graphical_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without a display, webbrowser would start a text browser on the same terminal as exo."""
    _first_run_at_a_local_desktop(monkeypatch, tmp_path)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(banner.sys, "platform", "linux")
    calls: list[str] = []
    monkeypatch.setattr(banner.webbrowser, "open", _recording_open(calls))

    banner.print_startup_banner(52415)
    time.sleep(0.1)

    assert calls == []
