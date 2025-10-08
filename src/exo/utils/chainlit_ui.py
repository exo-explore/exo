from __future__ import annotations

import contextlib

import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import final

from loguru import logger
from pydantic import PositiveInt


@final
class ChainlitLaunchError(RuntimeError):
    """Raised when Chainlit UI fails to launch or become ready."""


@dataclass(frozen=True, slots=True)
class ChainlitConfig:
    port: int = 8001
    host: str = "127.0.0.1"
    app_path: str | None = None
    ui_dir: str | None = None

def start_chainlit(port: PositiveInt, host: str, headless: bool = False) -> subprocess.Popen[bytes] | None:
    cfg = ChainlitConfig(
        port=port,
        host=host,
        ui_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "ui")),
    )
    try:
        return launch_chainlit(cfg, wait_ready=False, headless=headless)
    except ChainlitLaunchError as e:
        logger.warning(f"Chainlit not started: {e}")
    return None

def chainlit_cleanup(chainlit_proc: subprocess.Popen[bytes] | None) -> None:
    if chainlit_proc is None:
        return
    terminate_process(chainlit_proc)

def _is_port_open(host: str, port: int, timeout_s: float = 0.5) -> bool:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.settimeout(timeout_s)
        return s.connect_ex((host, port)) == 0


def _wait_http_ready(url: str, timeout_s: float = 15.0) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return True
        except Exception:
            time.sleep(0.25)
    return False


def _find_chainlit_executable() -> list[str]:
    exe = shutil.which("chainlit")
    if exe:
        return [exe]
    # Fallback to python -m chainlit if console script is not on PATH
    return [sys.executable, "-m", "chainlit"]


def _default_app_path() -> str:
    # Resolve the packaged chainlit app location
    here = os.path.dirname(__file__)
    app = os.path.abspath(os.path.join(here, "../../ui/chainlit_app.py"))
    return app


def launch_chainlit(
    cfg: ChainlitConfig,
    *,
    wait_ready: bool = True,
    ready_timeout_s: float = 20.0,
    headless: bool = False,
) -> subprocess.Popen[bytes]:
    if _is_port_open(cfg.host, cfg.port):
        raise ChainlitLaunchError(f"Port {cfg.port} already in use on {cfg.host}")

    app_path = cfg.app_path or _default_app_path()
    if not os.path.exists(app_path):
        raise ChainlitLaunchError(f"Chainlit app not found at {app_path}")

    env = os.environ.copy()
    cmd = [*_find_chainlit_executable(), "run", app_path, "--host", cfg.host, "--port", str(cfg.port)]
    if headless:
        cmd.append("--headless")
    cwd = None
    if cfg.ui_dir:
        cwd = cfg.ui_dir

    proc = subprocess.Popen(
        cmd,
        env=env,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if wait_ready:
        ok = _wait_http_ready(f"http://{cfg.host}:{cfg.port}", timeout_s=ready_timeout_s)
        if not ok:
            try:
                out, err = proc.communicate(timeout=1)
            except Exception:
                proc.terminate()
                out, err = b"", b""
            raise ChainlitLaunchError(
                (
                    f"Chainlit did not become ready on {cfg.host}:{cfg.port}.\n"
                    f"STDOUT:\n{out.decode(errors='ignore')}\n\n"
                    f"STDERR:\n{err.decode(errors='ignore')}"
                )
            )

    return proc


def terminate_process(proc: subprocess.Popen[bytes], *, timeout_s: float = 5.0) -> None:
    try:
        proc.terminate()
        try:
            proc.wait(timeout=timeout_s)
            return
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception:
        return
