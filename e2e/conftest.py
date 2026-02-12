"""Shared E2E test infrastructure for exo cluster tests."""

import asyncio
import os
import sys
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

E2E_DIR = Path(__file__).parent.resolve()
TIMEOUT = int(os.environ.get("E2E_TIMEOUT", "120"))


class Cluster:
    """Async wrapper around a docker compose exo cluster."""

    def __init__(self, name: str, overrides: list[str] | None = None):
        self.name = name
        self.project = f"e2e-{name}"
        compose_files = [str(E2E_DIR / "docker-compose.yml")]
        for path in overrides or []:
            compose_files.append(str(E2E_DIR / path))
        self._compose_base = [
            "docker", "compose",
            "-p", self.project,
            *[arg for f in compose_files for arg in ("-f", f)],
        ]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.stop()

    async def _run(self, *args: str, check: bool = True) -> str:
        proc = await asyncio.create_subprocess_exec(
            *self._compose_base, *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        output = stdout.decode()
        if check and proc.returncode != 0:
            print(output, file=sys.stderr)
            raise RuntimeError(f"docker compose {' '.join(args)} failed (rc={proc.returncode})")
        return output

    async def build(self):
        print("  Building images...")
        await self._run("build", "--quiet")

    async def start(self):
        print("  Starting cluster...")
        await self._run("up", "-d")

    async def stop(self):
        print("  Cleaning up...")
        await self._run("down", "--timeout", "5", check=False)

    async def logs(self) -> str:
        return await self._run("logs", check=False)

    async def wait_for(self, description: str, check_fn, timeout: int = TIMEOUT):
        """Poll check_fn every 2s until it returns True or timeout expires."""
        print(f"  Waiting for {description}...")
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            if await check_fn():
                print(f"  {description}")
                return
            await asyncio.sleep(2)
        output = await self.logs()
        print(f"--- cluster logs ---\n{output}\n---", file=sys.stderr)
        raise TimeoutError(f"Timed out waiting for {description}")

    async def assert_healthy(self):
        """Verify the cluster formed correctly: nodes started, discovered each other, elected a master, API responds."""

        async def both_nodes_started():
            log = await self.logs()
            return log.count("Starting node") >= 2

        async def nodes_discovered():
            log = await self.logs()
            return log.count("ConnectionMessageType.Connected") >= 2

        async def master_elected():
            log = await self.logs()
            return "demoting self" in log

        async def api_responding():
            try:
                with urlopen("http://localhost:52415/v1/models", timeout=3) as resp:
                    return resp.status == 200
            except (URLError, OSError):
                return False

        await self.wait_for("Both nodes started", both_nodes_started)
        await self.wait_for("Nodes discovered each other", nodes_discovered)
        await self.wait_for("Master election resolved", master_elected)
        await self.wait_for("API responding", api_responding)
