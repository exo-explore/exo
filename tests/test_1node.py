# type: ignore
"""Single-node integration tests.

Run with:
    uv run pytest tests/test_1node.py -v
"""

from __future__ import annotations

import time

import pytest
from exo_tools.harness import is_model_downloaded, place_instance

from .framework import DEFAULT_MODEL, InstanceSpec


@pytest.mark.cluster(count=1)
@pytest.mark.instance(DEFAULT_MODEL)
def test_place_instance_and_chat(session):
    resp = session.chat("Say hello in one sentence.")
    assert len(resp) > 0


@pytest.mark.cluster(count=1)
@pytest.mark.instance(DEFAULT_MODEL)
def test_chat_multiple_turns(session):
    first_reply = session.chat("What is 2 + 2?")
    assert len(first_reply) > 0

    second_reply = session.multi_turn(
        [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": first_reply},
            {"role": "user", "content": "Now multiply that by 3."},
        ]
    )
    assert len(second_reply) > 0


@pytest.mark.cluster(count=1)
@pytest.mark.instance(DEFAULT_MODEL)
def test_delete_instance(session):
    from exo_tools.harness import wait_for_instance_gone

    session.client.request_json("DELETE", f"/instance/{session.instance_id}")
    wait_for_instance_gone(session.client, session.instance_id, timeout=30.0)
    assert len(session.instances) == 0, (
        f"Expected no instances, found {len(session.instances)}"
    )


@pytest.mark.cluster(count=1)
def test_download_from_scratch(session):
    """Ensure the model is not on the cluster, then place an instance to
    trigger a fresh download and verify inference.
    """
    node_id = next(iter(session.state.get("nodeIdentities", {})))

    # Delete any existing download — the API call is idempotent
    session.client.request_json("DELETE", f"/download/{node_id}/{DEFAULT_MODEL}")

    # Poll until the model is gone (it may already be gone)
    deadline = time.time() + 60.0
    while time.time() < deadline:
        if not is_model_downloaded(session.client, DEFAULT_MODEL):
            break
        time.sleep(2.0)
    else:
        raise AssertionError(f"Expected {DEFAULT_MODEL} to be deleted from cluster")

    place_instance(session.client, DEFAULT_MODEL, timeout=900.0)
    session.instance_spec = InstanceSpec(model_id=DEFAULT_MODEL)
    resp = session.chat("Say hello in one sentence.")
    assert len(resp) > 0
