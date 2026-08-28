import logging
import os
import sys
import webbrowser
from pathlib import Path

import exo.shared.config as config

logger = logging.getLogger(__name__)


def _first_run_marker() -> Path:
    return config.bootstrap().exo_home.config / ".dashboard_opened"


def _is_first_run() -> bool:
    return not _first_run_marker().exists()


def _mark_first_run_done() -> None:
    first_run_marker = _first_run_marker()
    first_run_marker.parent.mkdir(parents=True, exist_ok=True)
    first_run_marker.touch()


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
        if not os.environ.get("EXO_RUNTIME_DIR"):
            try:
                webbrowser.open(dashboard_url)
                logger.info("First run detected — opening dashboard in browser")
            except Exception:
                logger.debug("Could not auto-open browser", exc_info=True)
        _mark_first_run_done()
