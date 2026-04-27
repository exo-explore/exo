# type: ignore
"""Two-node integration tests (ring + jaccl parallelism).

Hosts s9 and s10 must be Thunderbolt-connected for jaccl tests.

Run with:
    uv run pytest integration_tests/test_2node.py -v
"""

from __future__ import annotations

from .helpers import (
    ClusterInfo,
    chat_and_assert,
    chat_completion,
    make_client,
    place_and_wait,
    verify_node_count,
)


class TestTwoNodeInference:
    """Two-node inference tests with different parallelism strategies."""

    def test_2node_tensor_jaccl(self, two_node_cluster: ClusterInfo):
        """Place a model across 2 nodes with tensor/jaccl and verify inference."""
        client = make_client(two_node_cluster)

        place_and_wait(client, sharding="Tensor", instance_meta="MlxJaccl", min_nodes=2)
        verify_node_count(client, expected=2)
        chat_and_assert(client)

    def test_2node_pipeline_ring(self, two_node_cluster: ClusterInfo):
        """Place a model across 2 nodes with pipeline/ring and verify inference."""
        client = make_client(two_node_cluster)

        place_and_wait(
            client, sharding="Pipeline", instance_meta="MlxRing", min_nodes=2
        )
        verify_node_count(client, expected=2)
        chat_and_assert(client)

    def test_2node_ring_multi_turn(self, two_node_cluster: ClusterInfo):
        """Multi-turn conversation across 2 nodes with pipeline/ring."""
        client = make_client(two_node_cluster)

        place_and_wait(
            client, sharding="Pipeline", instance_meta="MlxRing", min_nodes=2
        )

        messages = [{"role": "user", "content": "What is the capital of France?"}]
        resp = chat_completion(client, messages=messages)
        first_reply = resp["choices"][0]["message"]["content"]
        assert len(first_reply) > 0

        messages.append({"role": "assistant", "content": first_reply})
        messages.append({"role": "user", "content": "What country is it in?"})

        resp = chat_completion(client, messages=messages)
        second_reply = resp["choices"][0]["message"]["content"]
        assert len(second_reply) > 0
