# type: ignore
"""Four-node integration tests.

Run with:
    uv run pytest tests/test_4node.py -v
"""

from __future__ import annotations

import pytest

from .framework import DEFAULT_MODEL


@pytest.mark.cluster(count=4, thunderbolt="a2a")
@pytest.mark.instance(DEFAULT_MODEL, sharding="pipeline", comm="ring", min_nodes=4)
def test_4node_pipeline_ring(session):
    resp = session.chat("Say hello in one sentence.")
    assert len(resp) > 0


@pytest.mark.cluster(count=4, thunderbolt="a2a")
@pytest.mark.instance(DEFAULT_MODEL, sharding="tensor", comm="jaccl", min_nodes=4)
def test_4node_tensor_jaccl(session):
    resp = session.chat("Say hello in one sentence.")
    assert len(resp) > 0
