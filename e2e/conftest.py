"""Shared E2E test infrastructure for exo cluster tests."""

import asyncio
import json
import os
import sys
from pathlib import Path
from urllib.request import urlopen, Request
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

    async def exec(self, service: str, *cmd: str, check: bool = True) -> tuple[int, str]:
        """Run a command inside a running container. Returns (returncode, output)."""
        proc = await asyncio.create_subprocess_exec(
            *self._compose_base, "exec", "-T", service, *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        output = stdout.decode()
        if check and proc.returncode != 0:
            raise RuntimeError(f"exec {' '.join(cmd)} in {service} failed (rc={proc.returncode})")
        return proc.returncode, output

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

    async def _api(self, method: str, path: str, body: dict | None = None, timeout: int = 30) -> dict:
        """Make an API request to the cluster. Returns parsed JSON."""
        url = f"http://localhost:52415{path}"
        data = json.dumps(body).encode() if body else None
        req = Request(url, data=data, headers={"Content-Type": "application/json"}, method=method)
        loop = asyncio.get_event_loop()
        resp_bytes = await loop.run_in_executor(None, lambda: urlopen(req, timeout=timeout).read())
        return json.loads(resp_bytes)

    async def place_model(self, model: str, timeout: int = 600):
        """Place a model instance on the cluster (triggers download) and wait until it's ready."""
        await self._api("POST", "/place_instance", {"model_id": model})

        async def model_ready():
            try:
                resp = await self._api("GET", "/v1/models")
                return any(m.get("id") == model for m in resp.get("data", []))
            except Exception:
                return False

        await self.wait_for(f"Model {model} ready", model_ready, timeout=timeout)

    async def chat(self, model: str, messages: list[dict], timeout: int = 600, **kwargs) -> dict:
        """Send a chat completion request. Retries until model is downloaded and inference completes."""
        body = json.dumps({"model": model, "messages": messages, **kwargs}).encode()
        deadline = asyncio.get_event_loop().time() + timeout
        last_error = None

        while asyncio.get_event_loop().time() < deadline:
            try:
                req = Request(
                    "http://localhost:52415/v1/chat/completions",
                    data=body,
                    headers={"Content-Type": "application/json"},
                )
                loop = asyncio.get_event_loop()
                resp_bytes = await loop.run_in_executor(None, lambda: urlopen(req, timeout=300).read())
                return json.loads(resp_bytes)
            except Exception as e:
                last_error = e
                await asyncio.sleep(5)

        raise TimeoutError(f"Chat request failed after {timeout}s: {last_error}")
