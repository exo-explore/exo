# type: ignore
"""Shared helpers for exo integration tests.

eco subprocess wrappers, cluster management utilities, and common test helpers.
ExoClient and related utilities are imported from exo.client.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass, field

from exo_tools.client import ExoClient
from exo_tools.client import ExoHttpError as ExoHttpError  # re-exported
from exo_tools.harness import (
    capture_cluster_snapshot as capture_cluster_snapshot,  # re-exported
)
from exo_tools.harness import (
    instance_id_from_instance,
    nodes_used_in_instance,
    wait_for_instance_gone,
    wait_for_instance_ready,
)

logger = logging.getLogger("integration_tests")

DEFAULT_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"

# Each test session gets a unique eco user to isolate reservations.
_SESSION_ID = uuid.uuid4().hex[:8]
_ECO_USER = f"test-{_SESSION_ID}"
_ECO_ENV = {**os.environ, "USER": _ECO_USER}

# When set, deploy from a GitHub branch/tag instead of local source (rsync).
# Useful for CI where the local worktree may not be the code under test.
_EXO_REF = os.environ.get("EXO_REF")


def _release_all_reservations() -> None:
    """Stop all clusters and release all reservations for this test session."""
    with contextlib.suppress(Exception):
        subprocess.run(
            ["eco", "stop"],
            capture_output=True,
            text=True,
            timeout=30,
            env=_ECO_ENV,
        )


# atexit covers normal exit and uncaught exceptions.
# SIGTERM/SIGHUP get explicit handlers for external kills.
# SIGINT is left to pytest so KeyboardInterrupt aborts tests naturally.
atexit.register(_release_all_reservations)


def _signal_handler(signum: int, _frame: object) -> None:
    _release_all_reservations()
    raise SystemExit(128 + signum)


for _sig in (signal.SIGTERM, signal.SIGHUP):
    signal.signal(_sig, _signal_handler)


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


# ---------------------------------------------------------------------------
# eco subprocess wrappers
# ---------------------------------------------------------------------------


def _run_eco(
    args: list[str], *, check: bool = True, timeout: int = 120
) -> subprocess.CompletedProcess[str]:
    """Run an eco command with USER=test.

    stdout is captured (JSON output), stderr is passed through to the
    console so eco's progress messages are visible during test runs.
    """
    logger.info(f"eco: {' '.join(args)}")
    return subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        check=check,
        timeout=timeout,
        env=_ECO_ENV,
    )


def eco_start_deploy(
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
        cmd.append("--thunderbolt")
    if wait:
        cmd.append("--wait")
    if ref:
        cmd.extend(["--ref", ref])

    result = _run_eco(cmd, timeout=timeout)
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


def eco_stop(hosts: list[str], *, keep: bool = False, timeout: int = 120) -> None:
    """Stop exo on the given hosts. If keep=True, keep the reservation."""
    cmd: list[str] = ["eco", "stop"]
    cmd.extend(hosts)
    if keep:
        cmd.append("--keep")
    _run_eco(cmd, timeout=timeout)


def eco_start_hosts(hosts: list[str], *, namespace: str, timeout: int = 300) -> None:
    """Start (previously stopped) hosts back into an existing namespace."""
    cmd: list[str] = ["eco", "--json", "start"]
    cmd.extend(hosts)
    cmd.extend(["--namespace", namespace])
    _run_eco(cmd, timeout=timeout)


def eco_release(hosts: list[str], timeout: int = 120) -> None:
    """Release hosts from the reservation."""
    cmd: list[str] = ["eco", "release"]
    cmd.extend(hosts)
    _run_eco(cmd, timeout=timeout)


def eco_logs(
    hosts: list[str], lines: int = 500, timeout: int = 60
) -> dict[str, list[str]]:
    """Fetch recent logs from cluster hosts."""
    cmd: list[str] = ["eco", "--json", "logs"]
    cmd.extend(hosts)
    cmd.extend(["-n", str(lines), "--raw"])
    result = _run_eco(cmd, check=False, timeout=timeout)
    if result.returncode != 0:
        return {"_error": [result.stderr]}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"_raw": result.stdout.splitlines()}


def eco_exec(hosts: list[str], command: str, timeout: int = 120) -> str:
    """Run an arbitrary command on the given hosts via eco."""
    cmd: list[str] = ["eco", "exec"]
    cmd.extend(hosts)
    cmd.append("--")
    cmd.extend(command.split())
    result = _run_eco(cmd, check=False, timeout=timeout)
    return result.stdout


# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------


def log_state(client: ExoClient, label: str) -> None:
    """Print a concise cluster state snapshot for debugging."""
    try:
        state = client.request_json("GET", "/state")
        if not state:
            print(f"[{label}] state=None")
            return
        ni = len(state.get("nodeIdentities", {}))
        nm = len(state.get("nodeMemory", {}))
        inst = len(state.get("instances", {}))
        topo = len(state.get("topology", {}).get("connections", []))
        print(f"[{label}] nodes={ni} memory={nm} instances={inst} topo_conns={topo}")
    except Exception as e:
        print(f"[{label}] ERROR: {e}")


def log_node_ids(cluster: ClusterInfo, label: str) -> None:
    """Print node IDs from each endpoint."""
    for host, url in cluster.api_endpoints.items():
        c = make_client_from_url(url)
        try:
            nid = c.request_json("GET", "/node_id")
            print(f"[{label}] {host} node_id={nid}")
        except Exception as e:
            print(f"[{label}] {host} UNREACHABLE: {e}")


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


def cleanup_all_instances(client: ExoClient) -> None:
    """Remove all running instances from the cluster."""
    state = client.request_json("GET", "/state")
    if not state:
        return
    instances = state.get("instances", {})
    for _id, instance in instances.items():
        try:
            iid = instance_id_from_instance(instance)
            client.request_json("DELETE", f"/instance/{iid}")
            wait_for_instance_gone(client, iid, timeout=30.0)
        except Exception as exc:
            logger.warning("Failed to clean up instance %s: %s", _id, exc)


# ---------------------------------------------------------------------------
# Instance helpers
# ---------------------------------------------------------------------------


def get_instance_ids(client: ExoClient) -> set[str]:
    """Get the set of current instance IDs from cluster state."""
    state = client.request_json("GET", "/state")
    if not state:
        return set()
    instances = state.get("instances", {})
    result = set()
    for _key, instance in instances.items():
        with contextlib.suppress(Exception):
            result.add(instance_id_from_instance(instance))
    return result


def wait_for_cluster_ready(
    client: ExoClient, expected_nodes: int = 1, timeout: float = 120.0
) -> None:
    """Wait until the cluster has all expected nodes with memory info.

    Placement requires nodeMemory for all nodes in a cycle. This function
    waits until both nodeIdentities and nodeMemory have at least
    `expected_nodes` entries, so the first placement attempt succeeds.
    """
    start = time.time()
    identities_count = 0
    memory_count = 0
    while time.time() - start < timeout:
        try:
            state = client.request_json("GET", "/state")
            if not state:
                time.sleep(1.0)
                continue
            identities_count = len(state.get("nodeIdentities", {}))
            memory_count = len(state.get("nodeMemory", {}))
            if identities_count >= expected_nodes and memory_count >= expected_nodes:
                return
        except Exception:
            pass
        time.sleep(1.0)

    raise TimeoutError(
        f"Cluster not ready: expected {expected_nodes} nodes, "
        f"got {identities_count} identities / {memory_count} with memory "
        f"after {timeout}s"
    )


def wait_for_cluster_nodes(
    client: ExoClient, expected_count: int, timeout: float = 120.0
) -> None:
    """Wait until the cluster reports at least `expected_count` nodes in nodeIdentities."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            state = client.request_json("GET", "/state")
            identities = state.get("nodeIdentities", {})
            if len(identities) >= expected_count:
                return
        except Exception:
            pass
        time.sleep(2.0)
    raise TimeoutError(
        f"Cluster did not reach {expected_count} nodes within {timeout}s"
    )


def wait_for_valid_placement(
    client: ExoClient,
    model_id: str = DEFAULT_MODEL,
    *,
    sharding: str = "Pipeline",
    instance_meta: str = "MlxRing",
    min_nodes: int = 1,
    timeout: float = 120.0,
) -> None:
    """Wait until the cluster can produce a valid placement preview.

    Polls /instance/previews until at least one error-free preview exists
    that matches the requested sharding/instance_meta and uses >= min_nodes.
    Useful after cluster topology changes (node disconnect/reconnect).
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = client.request_json(
                "GET", "/instance/previews", params={"model_id": model_id}
            )
            for preview in resp.get("previews", []):
                if preview.get("error") is not None:
                    continue
                if preview.get("sharding") != sharding:
                    continue
                if preview.get("instance_meta") != instance_meta:
                    continue
                instance = preview.get("instance")
                if instance and nodes_used_in_instance(instance) >= min_nodes:
                    return
        except Exception:
            pass
        time.sleep(2.0)
    raise TimeoutError(
        f"No valid {sharding}/{instance_meta} placement with >= {min_nodes} nodes "
        f"after {timeout}s"
    )


def place_and_wait(
    client: ExoClient,
    model_id: str = DEFAULT_MODEL,
    sharding: str = "Pipeline",
    instance_meta: str = "MlxRing",
    min_nodes: int = 1,
    timeout: float = 600.0,
    placement_retries: int = 10,
    placement_retry_delay: float = 10.0,
) -> str:
    """Place a model instance, find its instance_id, and wait for it to be ready.

    The /place_instance API returns a command_id, but instances are stored
    under their own generated instance_id. This function:
    1. Waits for the cluster to be ready (nodeMemory populated)
    2. Retries placement if it fails (e.g., due to timing)
    3. Polls state to find the newly-created instance
    4. Waits for that instance to become ready

    Returns the instance_id.
    """
    # Wait for cluster to have memory info before attempting placement
    wait_for_cluster_ready(client, expected_nodes=min_nodes)

    body = {
        "model_id": model_id,
        "sharding": sharding,
        "instance_meta": instance_meta,
        "min_nodes": min_nodes,
    }

    command_id = None
    instance_id = None

    # Retry placement — the first attempt can fail if the cluster is still settling
    for attempt in range(placement_retries):
        before_ids = get_instance_ids(client)

        resp = client.request_json("POST", "/place_instance", body=body)
        command_id = resp["command_id"]

        # Poll for the new instance to appear (give it some time per attempt)
        poll_deadline = time.time() + 30.0  # 30s to detect the new instance
        while time.time() < poll_deadline:
            current_ids = get_instance_ids(client)
            new_ids = current_ids - before_ids
            if new_ids:
                instance_id = next(iter(new_ids))
                break
            time.sleep(1.0)

        if instance_id is not None:
            break

        # Instance didn't appear — placement likely failed, retry
        if attempt < placement_retries - 1:
            time.sleep(placement_retry_delay)

    if instance_id is None:
        raise TimeoutError(
            f"No new instance appeared after {placement_retries} placement attempts "
            f"(last command_id={command_id}) within {timeout}s. "
            f"The placement may have failed — check cluster logs."
        )

    # Wait for the instance to become ready
    wait_for_instance_ready(client, instance_id, timeout=timeout)
    return instance_id


def chat_completion(
    client: ExoClient,
    model_id: str = DEFAULT_MODEL,
    prompt: str = "Say hello in one sentence.",
    max_tokens: int = 50,
    messages: list[dict[str, str]] | None = None,
):
    """Send a chat completion request and return the response."""
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
    return client.request_json(
        "POST",
        "/v1/chat/completions",
        body={
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
        },
    )


def chat_and_assert(client: ExoClient, model_id: str = DEFAULT_MODEL):
    """Send a chat request and assert we get a non-empty response. Returns the response."""
    resp = chat_completion(client, model_id)
    assert resp is not None
    choices = resp.get("choices", [])
    assert len(choices) > 0
    content = choices[0]["message"]["content"]
    assert len(content) > 0, "Expected non-empty response content"
    return resp


def verify_node_count(
    client: ExoClient, *, expected: int | None = None, expected_min: int | None = None
):
    """Verify the placed instance uses the expected number of nodes.

    Pass `expected` for an exact count or `expected_min` for a minimum.
    """
    assert (expected is not None) != (expected_min is not None), (
        "Pass exactly one of expected or expected_min"
    )

    state = client.request_json("GET", "/state")
    instances = state.get("instances", {})
    assert len(instances) > 0, "No instances found after placement"

    for _key, instance in instances.items():
        n = nodes_used_in_instance(instance)
        if expected is not None:
            assert n == expected, f"Expected exactly {expected} nodes, got {n}"
        else:
            assert n >= expected_min, f"Expected at least {expected_min} nodes, got {n}"
