# type: ignore
"""Four-node integration tests.

Uses hosts s2, s4, s9, s10.

Run with:
    uv run pytest tests/integration/test_4node.py -v

Note: tensor/jaccl is excluded because the current 4-node topology (star via s9)
doesn't form a 4-node jaccl cycle with the default small model. Add it back
when using a larger model or a full-mesh TB topology.
"""

from __future__ import annotations

import pytest

from .helpers import (
    ClusterInfo,
    chat_and_assert,
    make_client,
    place_and_wait,
    verify_node_count,
)

# Only pipeline/ring for now — see module docstring.
PARALLELISM = [
    ("Pipeline", "MlxRing"),
]


class TestFourNodeInference:
    """Four-node inference tests."""

    @pytest.mark.parametrize(
        "sharding,instance_meta", PARALLELISM, ids=["pipeline-ring"]
    )
    def test_4node_inference(
        self, four_node_cluster: ClusterInfo, sharding: str, instance_meta: str
    ):
        """Place a model across 4 nodes and verify inference."""
        client = make_client(four_node_cluster)

        place_and_wait(
            client,
            sharding=sharding,
            instance_meta=instance_meta,
            min_nodes=4,
            timeout=900.0,
        )
        verify_node_count(client, expected=4)
        chat_and_assert(client)

    def test_4node_cluster_state(self, four_node_cluster: ClusterInfo):
        """Verify the 4-node cluster reports all nodes in its state."""
        client = make_client(four_node_cluster)

        state = client.request_json("GET", "/state")
        assert state is not None

        identities = state.get("nodeIdentities", {})
        assert len(identities) >= 4, (
            f"Expected at least 4 node identities, got {len(identities)}"
        )
