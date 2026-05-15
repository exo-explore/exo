# type: ignore
"""Two-node integration tests (ring + jaccl parallelism).

Run with:
    uv run pytest tests/test_2node.py -v
"""

from __future__ import annotations

import pytest
from exo_tools.cluster import Thunderbolt
from exo_tools.harness import Comm, Sharding

from .framework import DEFAULT_MODEL


@pytest.mark.cluster(count=2, thunderbolt=Thunderbolt.A2A)
@pytest.mark.instance(
    DEFAULT_MODEL, sharding=Sharding.TENSOR, comm=Comm.JACCL, min_nodes=2
)
def test_2node_jaccl(session):
    resp = session.chat("Say hello in one sentence.")
    assert len(resp) > 0


@pytest.mark.cluster(count=2, thunderbolt=Thunderbolt.A2A)
@pytest.mark.instance(
    DEFAULT_MODEL, sharding=Sharding.PIPELINE, comm=Comm.RING, min_nodes=2
)
def test_2node_ring(session):
    resp = session.chat("Say hello in one sentence.")
    assert len(resp) > 0


@pytest.mark.cluster(count=2, thunderbolt=Thunderbolt.A2A)
@pytest.mark.instance(
    DEFAULT_MODEL, sharding=Sharding.TENSOR, comm=Comm.JACCL, min_nodes=2
)
def test_2node_jaccl_multi_turn(session):
    first = session.chat("What is the capital of France?")
    assert len(first) > 0
    second = session.multi_turn(
        [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": first},
            {"role": "user", "content": "What country is it in?"},
        ]
    )
    assert len(second) > 0
