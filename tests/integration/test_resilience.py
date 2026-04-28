# type: ignore
"""Resilience tests: disconnect/reconnect nodes and verify cluster recovery.

Run with:
    uv run pytest tests/integration/test_resilience.py -v
"""

from __future__ import annotations

import time

from .helpers import (
    ClusterInfo,
    chat_and_assert,
    cleanup_all_instances,
    eco_start_hosts,
    eco_stop,
    make_client,
    make_client_from_url,
    place_and_wait,
    verify_node_count,
    wait_for_cluster_nodes,
    wait_for_valid_placement,
)


class TestResilience:
    """Tests for cluster resilience during node disconnects."""

    def test_disconnect_reconnect(self, two_node_cluster: ClusterInfo):
        """Full disconnect/reconnect cycle:

        1. Place a 2-node instance, verify inference
        2. Stop one node, wait for instance to error out
        3. Clean up failed instance, place a 1-node instance on remaining node
        4. Verify inference works with 1 node
        5. Restart stopped node, wait for it to rejoin
        6. Clean up 1-node instance, place a 2-node instance again
        7. Verify inference works with both nodes
        """
        cluster = two_node_cluster
        client = make_client(cluster)

        # --- Phase 1: 2-node inference ---
        place_and_wait(
            client, sharding="Pipeline", instance_meta="MlxRing", min_nodes=2
        )
        verify_node_count(client, expected=2)
        chat_and_assert(client)

        # --- Phase 2: disconnect one node ---
        disconnected_host = cluster.hosts[1]
        eco_stop([disconnected_host], keep=True)
        time.sleep(10.0)

        # Switch to the remaining node's API endpoint
        remaining_host = cluster.hosts[0]
        remaining_url = cluster.api_endpoints[remaining_host]
        remaining_client = make_client_from_url(remaining_url)

        # Clean up the (now broken) 2-node instance
        cleanup_all_instances(remaining_client)

        # --- Phase 3: 1-node inference on remaining node ---
        place_and_wait(remaining_client, min_nodes=1)
        chat_and_assert(remaining_client)

        # --- Phase 4: reconnect and restore 2-node cluster ---
        cleanup_all_instances(remaining_client)
        eco_start_hosts([disconnected_host], namespace=cluster.namespace)
        wait_for_cluster_nodes(remaining_client, expected_count=2, timeout=120.0)

        # --- Phase 5: 2-node inference again ---
        wait_for_valid_placement(
            remaining_client,
            sharding="Pipeline",
            instance_meta="MlxRing",
            min_nodes=2,
        )
        place_and_wait(
            remaining_client,
            sharding="Pipeline",
            instance_meta="MlxRing",
            min_nodes=2,
        )
        verify_node_count(remaining_client, expected=2)
        chat_and_assert(remaining_client)
