# type: ignore
"""Single-node integration tests.

Run with:
    uv run pytest tests/integration/test_1node.py -v
"""

from __future__ import annotations

from .helpers import (
    DEFAULT_MODEL,
    ClusterInfo,
    capture_cluster_snapshot,
    chat_completion,
    eco,
    get_instance_ids,
    make_client,
    place_and_wait,
    wait_for_instance_gone,
)


class TestSingleNodeInference:
    """Tests that run on a single-node cluster."""

    def test_place_instance_and_chat(self, single_node_cluster: ClusterInfo):
        """Place a small model, send a chat message, and verify we get a response."""
        client = make_client(single_node_cluster)

        place_and_wait(client)

        resp = chat_completion(client)
        assert resp is not None
        choices = resp.get("choices", [])
        assert len(choices) > 0
        content = choices[0].get("message", {}).get("content", "")
        assert len(content) > 0, "Expected non-empty response content"

    def test_chat_multiple_turns(self, single_node_cluster: ClusterInfo):
        """Verify multi-turn conversation works."""
        client = make_client(single_node_cluster)
        place_and_wait(client)

        messages = [{"role": "user", "content": "What is 2 + 2?"}]
        resp = chat_completion(client, messages=messages)
        first_reply = resp["choices"][0]["message"]["content"]
        assert len(first_reply) > 0

        messages.append({"role": "assistant", "content": first_reply})
        messages.append({"role": "user", "content": "Now multiply that by 3."})

        resp = chat_completion(client, messages=messages)
        second_reply = resp["choices"][0]["message"]["content"]
        assert len(second_reply) > 0

    def test_delete_instance(self, single_node_cluster: ClusterInfo):
        """Place a model, then delete it and verify it's gone."""
        client = make_client(single_node_cluster)

        instance_id = place_and_wait(client)

        # Delete the instance
        client.request_json("DELETE", f"/instance/{instance_id}")
        wait_for_instance_gone(client, instance_id, timeout=30.0)

        # Verify no instances remain
        remaining = get_instance_ids(client)
        assert len(remaining) == 0, f"Expected no instances, found {len(remaining)}"

    def test_state_endpoint(self, single_node_cluster: ClusterInfo):
        """Verify the /state endpoint returns valid cluster state."""
        client = make_client(single_node_cluster)

        state = client.request_json("GET", "/state")
        assert state is not None
        assert "instances" in state

    def test_models_endpoint(self, single_node_cluster: ClusterInfo):
        """Verify the /models endpoint returns a list of available models."""
        client = make_client(single_node_cluster)

        models = client.request_json("GET", "/models")
        assert models is not None
        assert "data" in models
        assert isinstance(models["data"], list)
        assert len(models["data"]) > 0, "Expected at least one model in the catalog"

    def test_cluster_snapshot(self, single_node_cluster: ClusterInfo):
        """Verify capture_cluster_snapshot returns node info."""
        client = make_client(single_node_cluster)
        snapshot = capture_cluster_snapshot(client)
        assert snapshot is not None
        assert len(snapshot) > 0, "Expected non-empty cluster snapshot"


class TestSingleNodeDownload:
    """Tests involving model download."""

    def test_download_from_scratch(self, single_node_cluster: ClusterInfo):
        """Remove cached model, place it again (triggers download), and verify inference."""
        client = make_client(single_node_cluster)
        model = DEFAULT_MODEL

        # Clear the model cache on the host
        cache_name = model.replace("/", "--")
        eco.exec(
            single_node_cluster.hosts,
            f"rm -rf ~/.cache/huggingface/hub/models--{cache_name}",
        )

        # Place model — this will trigger a fresh download
        place_and_wait(client, model, timeout=900.0)

        # Verify inference works after fresh download
        resp = chat_completion(client, model)
        assert resp is not None
        content = resp["choices"][0]["message"]["content"]
        assert len(content) > 0
