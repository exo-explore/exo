# type: ignore
"""Four-node integration tests.

Uses hosts s2, s4, s9, s10.

Run with:
    uv run pytest integration_tests/test_4node.py -v
"""
from __future__ import annotations

from .helpers import (
    ClusterInfo,
    chat_and_assert,
    make_client,
    place_and_wait,
    verify_node_count,
)


class TestFourNodeInference:
    """Four-node inference tests."""

    def test_4node_pipeline_ring(self, four_node_cluster: ClusterInfo):
        """Place a model across 4 nodes with pipeline/ring and verify inference."""
        client = make_client(four_node_cluster)

        place_and_wait(client, sharding="Pipeline", instance_meta="MlxRing", min_nodes=4, timeout=900.0)
        verify_node_count(client, expected=4)
        chat_and_assert(client)

    def test_4node_cluster_state(self, four_node_cluster: ClusterInfo):
        """Verify the 4-node cluster reports all nodes in its state."""
        client = make_client(four_node_cluster)

        state = client.request_json("GET", "/state")
        assert state is not None

        identities = state.get("nodeIdentities", {})
        assert len(identities) >= 4, f"Expected at least 4 node identities, got {len(identities)}"
