import threading
import time
from pathlib import Path

import pytest

from exo.utils import banner


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

    monkeypatch.setattr(banner, "_FIRST_RUN_MARKER", tmp_path / ".dashboard_opened")
    monkeypatch.delenv("EXO_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(banner.webbrowser, "open", never_answering_open)

    start = time.monotonic()
    banner.print_startup_banner(52415)
    elapsed = time.monotonic() - start
    release.set()

    assert elapsed < 1.0
    assert (tmp_path / ".dashboard_opened").exists()
    deadline = time.monotonic() + 5
    while not opened_on and time.monotonic() < deadline:
        time.sleep(0.01)
    assert opened_on and opened_on[0] != threading.main_thread().name


def test_the_browser_is_opened_only_on_first_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    marker = tmp_path / ".dashboard_opened"
    marker.touch()
    calls: list[str] = []

    def recording_open(url: str) -> bool:
        calls.append(url)
        return True

    monkeypatch.setattr(banner, "_FIRST_RUN_MARKER", marker)
    monkeypatch.delenv("EXO_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(banner.webbrowser, "open", recording_open)

    banner.print_startup_banner(52415)
    time.sleep(0.1)

    assert calls == []
