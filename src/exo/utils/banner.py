import logging
import os
import sys
import webbrowser

from exo.shared.constants import EXO_CONFIG_HOME

logger = logging.getLogger(__name__)

_FIRST_RUN_MARKER = EXO_CONFIG_HOME / ".dashboard_opened"


def _is_first_run() -> bool:
    return not _FIRST_RUN_MARKER.exists()


def _mark_first_run_done() -> None:
    _FIRST_RUN_MARKER.parent.mkdir(parents=True, exist_ok=True)
    _FIRST_RUN_MARKER.touch()


def print_startup_banner(port: int) -> None:
    dashboard_url = f"http://localhost:{port}"
    first_run = _is_first_run()
    banner = f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║   ███████╗██╗  ██╗ ██████╗               ██████╗  ██████╗ ██╗  ██╗    ║
║   ██╔════╝╚██╗██╔╝██╔═══██╗      ██      ██╔══██╗██╔════╝ ╚██╗██╔╝    ║
║   █████╗   ╚███╔╝ ██║   ██║    ██████╗   ██║  ██║██║  ███╗ ╚███╔╝     ║
║   ██╔══╝   ██╔██╗ ██║   ██║    ╚═██╔═╝   ██║  ██║██║   ██║ ██╔██╗     ║
║   ███████╗██╔╝ ██╗╚██████╔╝      ╚═╝     ██████╔╝╚██████╔╝██╔╝ ██╗    ║
║   ╚══════╝╚═╝  ╚═╝ ╚═════╝               ╚═════╝  ╚═════╝ ╚═╝  ╚═╝    ║
║                                                                       ║
║   Distributed AI Inference Cluster                                    ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║  🌐 Dashboard & API Ready                                             ║
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
        if not os.environ.get("EXO_RUNTIME_DIR"):
            try:
                webbrowser.open(dashboard_url)
                logger.info("First run detected — opening dashboard in browser")
            except Exception:
                logger.debug("Could not auto-open browser", exc_info=True)
        _mark_first_run_done()
