# type: ignore
"""Cluster lifecycle management via eco.

Provides subprocess wrappers for eco commands (deploy, stop, start, release,
logs, exec) and a ClusterInfo dataclass. Reusable by integration tests,
bench, eval, and CI workflows.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import signal
import subprocess
import uuid
from dataclasses import dataclass, field

from .client import ExoClient

logger = logging.getLogger("exo_tools.cluster")

# When set, deploy from a GitHub branch/tag instead of local source (rsync).
_EXO_REF = os.environ.get("EXO_REF")


@dataclass
class ClusterInfo:
    """Holds the result of an `eco start --deploy` invocation."""

    hosts: list[str]
    namespace: str
    api_endpoints: dict[str, str]  # host -> url
    api_url: str  # primary endpoint for ExoClient

    primary_host: str = ""
    _host: str = field(init=False, repr=False, default="")
    _port: int = field(init=False, repr=False, default=52415)

    def __post_init__(self) -> None:
        if not self.primary_host:
            self.primary_host = self.hosts[0]
        url = self.api_url.replace("http://", "").replace("https://", "")
        parts = url.split(":")
        self._host = parts[0]
        self._port = int(parts[1]) if len(parts) > 1 else 52415

    def make_client(self, timeout_s: float = 7200.0) -> ExoClient:
        return ExoClient(self._host, self._port, timeout_s=timeout_s)


class EcoSession:
    """Manages an eco session with a unique user and automatic cleanup.

    Usage:
        session = EcoSession(user_prefix="test")
        cluster = session.start_deploy(count=2, thunderbolt=True)
        ...
        session.stop_all()  # or let atexit handle it

    The session registers atexit and signal handlers to ensure cleanup
    on normal exit, uncaught exceptions, SIGTERM, and SIGHUP. SIGINT
    is left unhandled so KeyboardInterrupt propagates normally.
    """

    def __init__(self, user_prefix: str = "test") -> None:
        self._session_id = uuid.uuid4().hex[:8]
        self.user = f"{user_prefix}-{self._session_id}"
        self._env = {**os.environ, "USER": self.user}

        # Register cleanup handlers
        atexit.register(self.stop_all)
        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, self._signal_handler)

    def _signal_handler(self, signum: int, _frame: object) -> None:
        self.stop_all()
        raise SystemExit(128 + signum)

    def stop_all(self) -> None:
        """Stop all clusters and release all reservations for this session."""
        with contextlib.suppress(Exception):
            subprocess.run(
                ["eco", "stop"],
                capture_output=True,
                text=True,
                timeout=30,
                env=self._env,
            )

    def _run(
        self, args: list[str], *, check: bool = True, timeout: int = 120
    ) -> subprocess.CompletedProcess[str]:
        """Run an eco command as this session's user.

        stdout is captured (JSON output), stderr is passed through to the
        console so eco's progress messages are visible.
        """
        logger.info(f"eco: {' '.join(args)}")
        return subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            check=check,
            timeout=timeout,
            env=self._env,
        )

    def start_deploy(
        self,
        hosts: list[str] | None = None,
        *,
        count: int | None = None,
        thunderbolt: bool = False,
        wait: bool = True,
        ref: str | None = _EXO_REF,
        timeout: int = 600,
    ) -> ClusterInfo:
        """Start and deploy exo on a set of hosts via eco.

        By default, deploys from local source via rsync. Set EXO_REF
        or pass ref= to deploy from a GitHub branch/tag instead (for CI).
        """
        cmd: list[str] = ["eco", "--json", "start", "--deploy"]
        if hosts:
            cmd.extend(hosts)
        if count is not None:
            cmd.extend(["--count", str(count)])
        if thunderbolt:
            cmd.append("--tb-a2a")
        if wait:
            cmd.append("--wait")
        if ref:
            cmd.extend(["--ref", ref])

        result = self._run(cmd, timeout=timeout)
        data = json.loads(result.stdout)["data"]
        endpoints: dict[str, str] = data["api_endpoints"]
        primary_host = data["hosts"][0]

        return ClusterInfo(
            hosts=data["hosts"],
            namespace=data["namespace"],
            api_endpoints=endpoints,
            api_url=endpoints[primary_host],
            primary_host=primary_host,
        )

    def stop(self, hosts: list[str], *, keep: bool = False, timeout: int = 120) -> None:
        """Stop exo on the given hosts. If keep=True, keep the reservation."""
        cmd: list[str] = ["eco", "stop"]
        cmd.extend(hosts)
        if keep:
            cmd.append("--keep")
        self._run(cmd, timeout=timeout)

    def start_hosts(
        self, hosts: list[str], *, namespace: str, timeout: int = 300
    ) -> None:
        """Start (previously stopped) hosts back into an existing namespace."""
        cmd: list[str] = ["eco", "--json", "start"]
        cmd.extend(hosts)
        cmd.extend(["--namespace", namespace])
        self._run(cmd, timeout=timeout)

    def release(self, hosts: list[str], timeout: int = 120) -> None:
        """Release hosts from the reservation."""
        cmd: list[str] = ["eco", "release"]
        cmd.extend(hosts)
        self._run(cmd, timeout=timeout)

    def logs(
        self, hosts: list[str], lines: int = 500, timeout: int = 60
    ) -> dict[str, list[str]]:
        """Fetch recent logs from cluster hosts."""
        cmd: list[str] = ["eco", "--json", "logs"]
        cmd.extend(hosts)
        cmd.extend(["-n", str(lines), "--raw"])
        result = self._run(cmd, check=False, timeout=timeout)
        if result.returncode != 0:
            return {"_error": [result.stderr]}
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return {"_raw": result.stdout.splitlines()}

    def exec(self, hosts: list[str], command: str, timeout: int = 120) -> str:
        """Run an arbitrary command on the given hosts via eco."""
        cmd: list[str] = ["eco", "exec"]
        cmd.extend(hosts)
        cmd.append("--")
        cmd.extend(command.split())
        result = self._run(cmd, check=False, timeout=timeout)
        return result.stdout


def make_client(cluster: ClusterInfo, timeout_s: float = 7200.0) -> ExoClient:
    """Create an ExoClient from a ClusterInfo."""
    return cluster.make_client(timeout_s=timeout_s)


def make_client_from_url(url: str, timeout_s: float = 7200.0) -> ExoClient:
    """Create an ExoClient from a URL string like 'http://host:port'."""
    url_clean = url.replace("http://", "").replace("https://", "")
    parts = url_clean.split(":")
    host = parts[0]
    port = int(parts[1]) if len(parts) > 1 else 52415
    return ExoClient(host, port, timeout_s=timeout_s)
