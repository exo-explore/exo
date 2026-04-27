# type: ignore
"""Resilience tests: disconnect/reconnect nodes and verify cluster recovery.

Run with:
    uv run pytest integration_tests/test_resilience.py -v
"""
from __future__ import annotations

import time

from .helpers import (
    ClusterInfo,
    ExoHttpError,
    eco_start_hosts,
    eco_stop,
    make_client,
    make_client_from_url,
    place_and_wait,
    wait_for_cluster_nodes,
)


class TestResilience:
    """Tests for cluster resilience during node disconnects."""

    def test_disconnect_reconnect(self, two_node_cluster: ClusterInfo):
        """Disconnect one node, verify cluster survives, reconnect and re-verify."""
        client = make_client(two_node_cluster)

        # Place a model on the cluster (single-node placement so it survives disconnect)
        place_and_wait(client)

        # Verify both nodes are in the cluster
        wait_for_cluster_nodes(client, expected_count=2)

        # Disconnect the second host (keep reservation so we can restart it)
        disconnected_host = two_node_cluster.hosts[1]
        eco_stop([disconnected_host], keep=True)

        # Wait for cluster to detect the disconnect
        time.sleep(5.0)

        # Use the first node's endpoint directly
        first_host = two_node_cluster.hosts[0]
        first_url = two_node_cluster.api_endpoints[first_host]
        remaining_client = make_client_from_url(first_url)

        # Verify the remaining node is still running
        state = remaining_client.request_json("GET", "/state")
        assert state is not None

        # Reconnect the disconnected host
        eco_start_hosts([disconnected_host], namespace=two_node_cluster.namespace)

        # Wait for cluster to reform with both nodes
        wait_for_cluster_nodes(remaining_client, expected_count=2, timeout=120.0)

        # Verify the full cluster is operational again
        state = remaining_client.request_json("GET", "/state")
        identities = state.get("nodeIdentities", {})
        assert len(identities) >= 2, f"Expected 2 nodes after reconnect, got {len(identities)}"

    def test_api_resilience_during_disconnect(self, two_node_cluster: ClusterInfo):
        """Verify the API server stays alive and responds during a node disconnect.

        We specifically check that the remaining node's API server is still
        reachable (no ConnectionRefusedError) and returns valid responses.
        HTTP errors from exo are acceptable — crashes/connection refusals are not.
        """
        disconnected_host = two_node_cluster.hosts[1]
        eco_stop([disconnected_host], keep=True)

        time.sleep(3.0)

        first_host = two_node_cluster.hosts[0]
        first_url = two_node_cluster.api_endpoints[first_host]
        remaining_client = make_client_from_url(first_url)

        # Server should still be alive — allow HTTP errors but not connection failures
        try:
            state = remaining_client.request_json("GET", "/state")
            assert state is not None, "Expected /state to return a response"
        except ExoHttpError:
            pass  # HTTP errors are OK — the server is still responding

        try:
            models = remaining_client.request_json("GET", "/models")
            assert models is not None, "Expected /models to return a response"
        except ExoHttpError:
            pass  # HTTP errors are OK — the server is still responding

        # Reconnect
        eco_start_hosts([disconnected_host], namespace=two_node_cluster.namespace)
        wait_for_cluster_nodes(remaining_client, expected_count=2)
