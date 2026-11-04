"""The harness's own helpers, without starting nodes."""

import fcntl
from collections.abc import AsyncGenerator, Mapping
from pathlib import Path

import pytest

from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.profiling import NodeNetworkInfo
from exo.testkit import node as testkit_node
from exo.testkit.cluster import ChatResult, Cluster, HarnessTimeoutError
from exo.testkit.network_table import NETWORK_TABLE_ENV, NetworkTable, write_table
from exo.testkit.ports import LEASES, PORTS
from exo.testkit.stub.directives import expected_reply


def test_node_ports_are_leased_below_the_ephemeral_range_until_close(
    tmp_path: Path,
) -> None:
    """A port found by binding port 0 can be handed to another socket before the node binds
    it, and for a discovery port exo doesn't notice (exo.testkit.ports)."""
    cluster = Cluster(3, tmp_path)
    ports = [
        port
        for node in cluster.nodes
        for port in (node.api_port, node.zenoh_port, node.discovery_port)
    ]
    assert len(set(ports)) == 9
    assert all(port in PORTS for port in ports)
    with (LEASES / f"{ports[0]}.lock").open("a") as other:
        with pytest.raises(OSError):
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cluster.close()
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(other, fcntl.LOCK_UN)


def test_wait_for_ends_only_when_the_predicate_is_true(tmp_path: Path) -> None:
    cluster = Cluster(1, tmp_path)
    answers = [False, False, True]
    cluster.wait_for(lambda: answers.pop(0), timeout=5, what="it")
    assert answers == []
    with pytest.raises(HarnessTimeoutError, match="waiting for nothing"):
        cluster.wait_for(lambda: False, timeout=0.3, what="nothing")


@pytest.mark.parametrize("not_a_bool", [None, 0, 1, "", [], ["x"]])
def test_wait_for_refuses_a_predicate_that_returns_something_else(
    tmp_path: Path, not_a_bool: object
) -> None:
    """The trap: an empty list or a 0 isn't "not yet", it's a value."""
    cluster = Cluster(1, tmp_path)
    with pytest.raises(TypeError, match="wait_for_value"):
        cluster.wait_for(lambda: not_a_bool, timeout=5, what="it")  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("value", [0, False, "", [], {}])
def test_wait_for_value_returns_any_value_but_none(
    tmp_path: Path, value: object
) -> None:
    """A master at index 0, an empty list of violations: values, not "not yet"."""
    cluster = Cluster(1, tmp_path)
    answers: list[object] = [None, None, value]
    assert cluster.wait_for_value(lambda: answers.pop(0), timeout=5, what="it") == value
    with pytest.raises(HarnessTimeoutError):
        cluster.wait_for_value(lambda: None, timeout=0.3, what="nothing")


async def test_the_sweep_reports_each_peer_at_its_own_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """exo's worker tells edges apart only by address: two peers both at 127.0.0.1 would
    share one edge value, and the edge to the second would never be made."""
    a, b = NodeId("a"), NodeId("b")
    table = tmp_path / "network.json"
    write_table(table, NetworkTable(addresses={a: "127.0.0.2", b: "127.0.0.3"}))
    monkeypatch.setenv(NETWORK_TABLE_ENV, str(table))

    async def every_peer_on_loopback(
        topology: Topology,
        self_node_id: NodeId,
        node_network: Mapping[NodeId, NodeNetworkInfo],
        api_port: int,
    ) -> AsyncGenerator[tuple[str, NodeId], None]:
        for peer in (a, b):
            yield "127.0.0.1", peer

    monkeypatch.setattr(testkit_node, "_real_check_reachable", every_peer_on_loopback)
    found = [
        pair async for pair in testkit_node.check_reachable(Topology(), a, {}, 52415)
    ]
    assert found == [("127.0.0.2", a), ("127.0.0.3", b)]


def _result(content: str, finish: str | None, error: str | None = None) -> ChatResult:
    return ChatResult(
        prompt="p [stub tokens=4]",
        via="n0",
        stream=True,
        max_tokens=None,
        started=0.0,
        content=content,
        finish_reason=finish,
        error=error,
    )


def test_a_reply_is_checked_token_by_token() -> None:
    tokens = expected_reply("p [stub tokens=4]").tokens
    assert _result("".join(tokens), "stop").problems() == []
    swapped = tokens[1] + tokens[0] + tokens[2] + tokens[3]
    assert "token 0" in _result(swapped, "stop").problems()[0]
    assert _result("".join(tokens[:3]), "stop").problems() != []
    # Cut short by an error: fine if it's an exact prefix, not if anything is out of place
    assert _result("".join(tokens[:2]), None, error="gone").problems() == []
    assert _result(tokens[1], None, error="gone").problems() != []
