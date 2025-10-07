from __future__ import annotations

import threading
import time
import urllib.request
import webbrowser
from typing import final


def _wait_http_ready(url: str, *, timeout_s: float = 30.0, check_interval_s: float = 0.25) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return True
        except Exception:
            time.sleep(check_interval_s)
    return False


@final
class _UrlOpenerThread(threading.Thread):
    def __init__(self, url: str, *, ready_timeout_s: float = 30.0, check_interval_s: float = 0.25) -> None:
        super().__init__(daemon=True)
        self.url = url
        self.ready_timeout_s = ready_timeout_s
        self.check_interval_s = check_interval_s

    def run(self) -> None:
        if _wait_http_ready(self.url, timeout_s=self.ready_timeout_s, check_interval_s=self.check_interval_s):
            try:
                webbrowser.open(self.url, new=2, autoraise=True)
            except Exception:
                # Last-ditch fallback. Ignore errors to avoid affecting core app lifecycle.
                pass


def open_url_in_browser_when_ready(url: str, *, ready_timeout_s: float = 30.0) -> None:
    _UrlOpenerThread(url, ready_timeout_s=ready_timeout_s).start()


