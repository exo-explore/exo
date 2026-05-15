# type: ignore
"""Four-node integration tests.

Run with:
    uv run pytest tests/test_4node.py -v
"""

from __future__ import annotations

import pytest
from exo_tools.cluster import Thunderbolt
from exo_tools.harness import Comm, Sharding

from .framework import DEFAULT_MODEL


@pytest.mark.cluster(count=4, thunderbolt=Thunderbolt.A2A)
@pytest.mark.instance(
    DEFAULT_MODEL, sharding=Sharding.PIPELINE, comm=Comm.RING, min_nodes=4
)
def test_4node_pipeline_ring(session):
    resp = session.chat("Say hello in one sentence.")
    assert len(resp) > 0


@pytest.mark.cluster(count=4, thunderbolt=Thunderbolt.A2A)
@pytest.mark.instance(
    DEFAULT_MODEL, sharding=Sharding.TENSOR, comm=Comm.JACCL, min_nodes=4
)
def test_4node_tensor_jaccl(session):
    resp = session.chat("Say hello in one sentence.")
    assert len(resp) > 0
