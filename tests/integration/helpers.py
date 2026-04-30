# type: ignore
"""Shared helpers for exo integration tests.

Test-specific helpers built on top of exo_tools.cluster and exo_tools.harness.
"""

from __future__ import annotations

import contextlib
import logging
import time

from exo_tools.client import ExoClient
from exo_tools.client import ExoHttpError as ExoHttpError  # re-exported
from exo_tools.cluster import ClusterInfo as ClusterInfo  # re-exported
from exo_tools.cluster import EcoSession
from exo_tools.cluster import make_client as make_client  # re-exported
from exo_tools.cluster import (
    make_client_from_url as make_client_from_url,  # re-exported
)
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

# Single eco session for the entire test process.
eco = EcoSession(user_prefix="test")


# ---------------------------------------------------------------------------
# Instance helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Cluster readiness helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Placement + inference helpers
# ---------------------------------------------------------------------------


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
