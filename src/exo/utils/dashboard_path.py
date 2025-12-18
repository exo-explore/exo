import os
import sys
from pathlib import Path
from typing import cast


def find_dashboard() -> Path:
    dashboard = (
        _find_dashboard_in_env()
        or _find_dashboard_in_repo()
        or _find_dashboard_in_bundle()
    )
    if not dashboard:
        raise FileNotFoundError(
            "Unable to locate dashboard assets - make sure the dashboard has been built, or export DASHBOARD_DIR if you've built the dashboard elsewhere."
        )
    return dashboard


def _find_dashboard_in_env() -> Path | None:
    env = os.environ.get("DASHBOARD_DIR")
    if not env:
        return None
    resolved_env = Path(env).expanduser().resolve()

    return resolved_env


def _find_dashboard_in_repo() -> Path | None:
    current_module = Path(__file__).resolve()
    for parent in current_module.parents:
        build = parent / "dashboard" / "build"
        if build.is_dir() and (build / "index.html").exists():
            return build
    return None


def _find_dashboard_in_bundle() -> Path | None:
    frozen_root = cast(str | None, getattr(sys, "_MEIPASS", None))
    if frozen_root is None:
        return None
    candidate = Path(frozen_root) / "dashboard"
    if candidate.is_dir():
        return candidate
    return None
