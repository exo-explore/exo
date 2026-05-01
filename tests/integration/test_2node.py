# type: ignore
"""Two-node integration tests (ring + jaccl parallelism).

Hosts s9 and s10 must be Thunderbolt-connected for jaccl tests.

Run with:
    uv run pytest tests/integration/test_2node.py -v
"""

from __future__ import annotations

import pytest

from .helpers import (
    ClusterInfo,
    chat_and_assert,
    chat_completion,
    make_client,
    place_and_wait,
    verify_node_count,
)

PARALLELISM = [
    ("Tensor", "MlxJaccl"),
    ("Pipeline", "MlxRing"),
]


class TestTwoNodeInference:
    """Two-node inference tests with different parallelism strategies."""

    @pytest.mark.parametrize(
        "sharding,instance_meta", PARALLELISM, ids=["tensor-jaccl", "pipeline-ring"]
    )
    def test_2node_inference(
        self, two_node_cluster: ClusterInfo, sharding: str, instance_meta: str
    ):
        """Place a model across 2 nodes and verify inference."""
        client = make_client(two_node_cluster)

        place_and_wait(
            client, sharding=sharding, instance_meta=instance_meta, min_nodes=2
        )
        verify_node_count(client, expected=2)
        chat_and_assert(client)

    @pytest.mark.parametrize(
        "sharding,instance_meta", PARALLELISM, ids=["tensor-jaccl", "pipeline-ring"]
    )
    def test_2node_multi_turn(
        self, two_node_cluster: ClusterInfo, sharding: str, instance_meta: str
    ):
        """Multi-turn conversation across 2 nodes."""
        client = make_client(two_node_cluster)

        place_and_wait(
            client, sharding=sharding, instance_meta=instance_meta, min_nodes=2
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
