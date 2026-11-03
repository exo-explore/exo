import logging
import os
import sys
import threading
import webbrowser

from exo.shared.constants import EXO_CONFIG_HOME

logger = logging.getLogger(__name__)

_FIRST_RUN_MARKER = EXO_CONFIG_HOME / ".dashboard_opened"


def _is_first_run() -> bool:
    return not _FIRST_RUN_MARKER.exists()


def _mark_first_run_done() -> None:
    _FIRST_RUN_MARKER.parent.mkdir(parents=True, exist_ok=True)
    _FIRST_RUN_MARKER.touch()


def _has_local_display() -> bool:
    """Whether a browser opened now would appear in front of the person running exo: not in an ssh session (it
    would open on whoever is at the machine's screen), and on Linux only with a graphical session (otherwise
    webbrowser falls back to a text browser on the same terminal as exo)."""
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        return False
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def _open_browser(url: str) -> None:
    try:
        webbrowser.open(url)
        logger.info("First run detected — opening dashboard in browser")
    except Exception:
        logger.debug("Could not auto-open browser", exc_info=True)


def print_startup_banner(port: int) -> None:
    dashboard_url = f"http://localhost:{port}"
    first_run = _is_first_run()
    banner = f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║   ███████╗██╗  ██╗ ██████╗                                            ║
║   ██╔════╝╚██╗██╔╝██╔═══██╗                                           ║
║   █████╗   ╚███╔╝ ██║   ██║                                           ║
║   ██╔══╝   ██╔██╗ ██║   ██║                                           ║
║   ███████╗██╔╝ ██╗╚██████╔╝                                           ║
║   ╚══════╝╚═╝  ╚═╝ ╚═════╝                                            ║
║                                                                       ║
║   Distributed AI Inference Cluster                                    ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║  Dashboard & API Ready                                                ║
║                                                                       ║
║  {dashboard_url}{" " * (69 - len(dashboard_url))}║
║                                                                       ║
║  Click the URL above to open the dashboard in your browser            ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝

"""

    print(banner, file=sys.stderr)

    if first_run:
        # Skip browser open when running inside the native macOS app —
        # FirstLaunchPopout.swift handles the auto-open with a countdown.
        if not os.environ.get("EXO_RUNTIME_DIR") and _has_local_display():
            # webbrowser.open can block for minutes when no browser answers (e.g. started over ssh), and this
            # runs on the node's event loop, so open it on its own thread.
            threading.Thread(
                target=_open_browser,
                args=(dashboard_url,),
                name="open-dashboard",
                daemon=True,
            ).start()
        _mark_first_run_done()
