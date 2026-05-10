"""Structured benchmark results — metadata capture + JSON output.

Every benchmark run produces a single JSON file with a stable schema:

- ``metadata``: exo SHA, ISO timestamps, hostnames, and any user-supplied
  tags identifying the run.
- ``cluster``: snapshot from the API (node identities, topology, memory).
- ``params``: the benchmark's input parameters (sweep config, etc).
- ``runs``: per-request result rows.
- ``derived``: any computed summaries (``t_cum_seconds`` for context scaling).

The format is intentionally additive so downstream tooling (plot scripts,
dashboards) can rely on optional fields being absent rather than malformed.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from exo_tools.client import ExoClient
from exo_tools.harness import capture_cluster_snapshot


def _git_describe(repo_root: Path) -> str | None:
    """Return ``<short-sha>[-dirty]`` for the repo at ``repo_root`` or None."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return None
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return sha


@dataclass
class RunMetadata:
    """Identifies a single bench run."""

    run_id: str
    benchmark: str
    started_at: str
    finished_at: str | None = None
    exo_sha: str | None = None
    hostname: str = ""
    platform: str = ""
    tags: dict[str, str] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        benchmark: str,
        repo_root: Path,
        *,
        tags: dict[str, str] | None = None,
    ) -> RunMetadata:
        now = datetime.now(timezone.utc)
        run_id = f"{benchmark}_{now.strftime('%Y%m%dT%H%M%SZ')}_{os.getpid()}"
        return cls(
            run_id=run_id,
            benchmark=benchmark,
            started_at=now.isoformat(),
            exo_sha=_git_describe(repo_root),
            hostname=socket.gethostname(),
            platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
            tags=dict(tags or {}),
        )


@dataclass
class ResultsBundle:
    """Container for a single benchmark's results, before being written."""

    metadata: RunMetadata
    params: dict[str, Any] = field(default_factory=dict)
    cluster: dict[str, Any] = field(default_factory=dict)
    runs: list[dict[str, Any]] = field(default_factory=list)
    cold_controls: list[dict[str, Any]] = field(default_factory=list)
    derived: dict[str, Any] = field(default_factory=dict)

    def capture_cluster(self, client: ExoClient) -> None:
        """Snapshot the cluster state into ``self.cluster``."""
        try:
            snapshot = capture_cluster_snapshot(client)
            if snapshot:
                self.cluster.update(snapshot)
        except Exception:
            # Non-fatal: a benchmark without cluster snapshot is still valid
            pass

    def write_json(self, output_dir: Path) -> Path:
        """Write the bundle as ``<output_dir>/<run_id>.json`` and return the path."""
        if self.metadata.finished_at is None:
            self.metadata.finished_at = datetime.now(timezone.utc).isoformat()
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{self.metadata.run_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
        return path


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upwards from ``start`` (or this file) until a ``.git`` dir is found."""
    cur = (start or Path(__file__)).resolve()
    for parent in (cur, *cur.parents):
        if (parent / ".git").is_dir() or (parent / ".git").is_file():
            return parent
    raise RuntimeError(f"Could not locate repo root above {cur}")
