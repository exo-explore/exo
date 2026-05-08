# type: ignore
"""Resilience tests: disconnect/reconnect nodes and verify cluster recovery.

Run with:
    uv run pytest tests/test_resilience.py -v
"""

from __future__ import annotations

import pytest
from exo_tools.cluster import Thunderbolt
from exo_tools.harness import Comm, Sharding, cleanup_all_instances, place_instance

from .framework import DEFAULT_MODEL, InstanceSpec


@pytest.mark.cluster(count=2, thunderbolt=Thunderbolt.A2A)
@pytest.mark.instance(
    DEFAULT_MODEL, sharding=Sharding.PIPELINE, comm=Comm.RING, min_nodes=2
)
def test_node_recovery(session):
    """Full disconnect/reconnect cycle.

    1. Place a 2-node instance, verify inference
    2. Disconnect one node
    3. Place a 1-node instance on remaining node, verify inference
    4. Reconnect the stopped node, wait for the cluster to reform
    5. Place a 2-node instance again, verify inference
    """
    # --- Phase 1: 2-node inference ---
    resp = session.chat("Hello")
    assert len(resp) > 0

    # --- Phase 2: disconnect one node ---
    session.disconnect_node(1)
    session.wait_ready(60)

    # Clean up the now-broken 2-node instance
    cleanup_all_instances(session.client)

    # --- Phase 3: 1-node inference on the remaining node ---
    place_instance(session.client, DEFAULT_MODEL, min_nodes=1)
    session.instance_spec = InstanceSpec(model_id=DEFAULT_MODEL, min_nodes=1)
    resp = session.chat("Hello")
    assert len(resp) > 0

    # --- Phase 4: reconnect and restore 2-node cluster ---
    cleanup_all_instances(session.client)
    session.reconnect_node(1)
    session.wait_ready(60)

    # --- Phase 5: 2-node inference again ---
    place_instance(session.client, DEFAULT_MODEL, min_nodes=2)
    session.instance_spec = InstanceSpec(model_id=DEFAULT_MODEL, min_nodes=2)
    resp = session.chat("Hello again")
    assert len(resp) > 0
