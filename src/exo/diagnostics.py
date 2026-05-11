"""Collect local exo diagnostics for postmortem analysis."""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, cast


class _ReadableResponse(Protocol):
    def read(self) -> bytes: ...

    def close(self) -> None: ...


def _get_xdg_dir(env_var: str, fallback: str) -> Path:
    exo_home = os.environ.get("EXO_HOME")
    if exo_home is not None:
        return Path.home() / exo_home
    if sys.platform != "linux":
        return Path.home() / ".exo"
    xdg_value = os.environ.get(env_var)
    if xdg_value is not None:
        return Path(xdg_value) / "exo"
    return Path.home() / fallback / "exo"


_EXO_CACHE_HOME = _get_xdg_dir("XDG_CACHE_HOME", ".cache")
_EXO_DATA_HOME = _get_xdg_dir("XDG_DATA_HOME", ".local/share")
_EXO_LOG_DIR = _EXO_CACHE_HOME / "exo_log"
_EXO_EVENT_LOG_DIR = _EXO_DATA_HOME / "event_log"


def main(argv: Sequence[str] | None = None) -> None:
    """Create a compressed local diagnostics bundle.

    Args:
        argv: Optional command-line arguments for tests or embedded callers.
    """
    parser = argparse.ArgumentParser(prog="exo-diagnostics")
    parser.add_argument(
        "bundle",
        choices=("bundle",),
        help="Collect local process, API, event-log, and file-log diagnostics.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:52415",
        help="Local exo API base URL to query.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .tar.gz path. Defaults to ./exo-diagnostics-<timestamp>.tar.gz.",
    )
    namespace = parser.parse_args(argv)
    base_url = cast(str, namespace.base_url)
    output_arg = cast(str | None, namespace.output)
    output_path = (
        Path(output_arg).expanduser()
        if output_arg is not None
        else Path.cwd() / f"exo-diagnostics-{_timestamp()}.tar.gz"
    )
    bundle_path = collect_bundle(base_url=base_url, output_path=output_path)
    print(bundle_path)


def collect_bundle(*, base_url: str, output_path: Path) -> Path:
    """Collect local diagnostics and write them to a compressed archive.

    Args:
        base_url: Local exo API base URL.
        output_path: Destination archive path.

    Returns:
        The path to the written archive.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="exo-diagnostics-") as temp_directory:
        root = Path(temp_directory) / "exo-diagnostics"
        root.mkdir()
        _write_manifest(root, base_url)
        _collect_http(base_url, root / "api")
        _collect_processes(root / "processes")
        _collect_memory(root / "memory")
        _copy_existing_path(_EXO_LOG_DIR, root / "logs")
        _copy_existing_path(_EXO_EVENT_LOG_DIR, root / "event_log")
        _copy_existing_path(_EXO_CACHE_HOME / "exo.log", root / "legacy-exo.log")
        with tarfile.open(output_path, "w:gz") as archive:
            archive.add(root, arcname=root.name)
    return output_path


def _write_manifest(root: Path, base_url: str) -> None:
    manifest = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "baseUrl": base_url,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def _collect_http(base_url: str, target: Path) -> None:
    target.mkdir()
    for endpoint in ("node_id", "state", "v1/models"):
        safe_name = endpoint.replace("/", "_")
        url = f"{base_url.rstrip('/')}/{endpoint}"
        try:
            response = cast(_ReadableResponse, urllib.request.urlopen(url, timeout=5))
            try:
                body = response.read().decode("utf-8", "replace")
            finally:
                response.close()
            (target / f"{safe_name}.json").write_text(body)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            payload = {
                "url": url,
                "errorType": type(error).__name__,
                "error": str(error),
            }
            (target / f"{safe_name}.error.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )


def _collect_processes(target: Path) -> None:
    target.mkdir()
    (target / "ps.txt").write_text(
        _run_command(("ps", "-axo", "pid,ppid,etime,rss,command"))
    )


def _collect_memory(target: Path) -> None:
    target.mkdir()
    system = platform.system()
    if system == "Darwin":
        commands = {
            "vm_stat.txt": ("vm_stat",),
            "memory_pressure.txt": ("memory_pressure",),
        }
    elif system == "Linux":
        commands = {
            "free.txt": ("free", "-m"),
        }
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            (target / "meminfo.txt").write_text(meminfo.read_text())
    else:
        commands = {"platform.txt": ("uname", "-a")}
    for filename, command in commands.items():
        (target / filename).write_text(_run_command(command))


def _run_command(command: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"{type(error).__name__}: {error}\n"
    return result.stdout


def _copy_existing_path(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
