"""Run one exo node for the harness: `python -m exo.testkit.node NODE_ID [exo arguments...]`.

This is exo's own `exo.main.main()`, with three changes made in this process before it starts:

1. The node id is NODE_ID instead of a random one, so the harness knows every node's id (and so
   which one an election picks) before it starts. A restart "as a new device" passes a new id.
2. Runners use the stub engine (exo.testkit.stub) instead of MLX.
3. The worker's reachability probe asks each node at that node's own API port (every node of
   a real cluster listens on the same port, here each has its own), only on 127.0.0.1, and not
   across a pair the harness has cut. It reports each peer under a label of its own
   (127.0.0.k): exo's worker tells topology edges apart only by address, so with every peer
   at 127.0.0.1 an edge to one peer would hide the edge to another. The labels are only
   labels: Linux routes all of 127/8 to the loopback but macOS doesn't, so nothing may dial
   them (the stub ring's connections go through the harness, which maps them to 127.0.0.1).

Everything else - discovery, zenoh, election, master, worker, API, downloads, runners - is the
production code, unchanged.
"""

import os
import sys
from collections.abc import AsyncGenerator, Mapping

import anyio
import httpx

import exo.main
import exo.utils.info_gatherer.net_profile as net_profile
import exo.worker.main as worker_main
import exo.worker.runner.supervisor as supervisor
from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.profiling import NodeNetworkInfo
from exo.testkit.network_table import PROBE_OWN_PORT_ENV, read_table
from exo.testkit.stub.entrypoint import entrypoint as stub_entrypoint

# A probe of a blackholed address fails only after every attempt times out: httpx's 5 s
# timeout plus net_profile's 1 s pause, REACHABILITY_ATTEMPTS times
BLACKHOLED_PROBE_SECONDS = net_profile.REACHABILITY_ATTEMPTS * (5.0 + 1.0)

_real_check_reachability = net_profile.check_reachability
_real_check_reachable = net_profile.check_reachable


def _probe_for(self_id: NodeId):
    async def check_reachability(
        target_ip: str,
        expected_node_id: NodeId,
        out: dict[NodeId, set[str]],
        client: httpx.AsyncClient,
        api_port: int,
    ) -> None:
        if target_ip != "127.0.0.1":
            return
        table = read_table()
        port = (
            api_port
            if os.environ.get(PROBE_OWN_PORT_ENV) == "1"
            else table.api_ports.get(expected_node_id)
        )
        if port is None:
            return
        match table.cut_between(self_id, expected_node_id):
            case "blackhole":
                await anyio.sleep(BLACKHOLED_PROBE_SECONDS)
                return
            case "reset":
                return
            case None:
                pass
        await _real_check_reachability(
            target_ip, expected_node_id, out, client, api_port=port
        )

    return check_reachability


async def check_reachable(
    topology: Topology,
    self_node_id: NodeId,
    node_network: Mapping[NodeId, NodeNetworkInfo],
    api_port: int,
) -> AsyncGenerator[tuple[str, NodeId], None]:
    """exo's sweep, with each reachable peer reported at its own address."""
    addresses = read_table().addresses
    async for ip, node_id in _real_check_reachable(
        topology, self_node_id, node_network, api_port
    ):
        yield addresses.get(node_id, ip), node_id


def install(node_id: NodeId) -> None:
    exo.main.get_node_zid = lambda: node_id
    supervisor.entrypoint = stub_entrypoint
    net_profile.check_reachability = _probe_for(node_id)
    worker_main.check_reachable = check_reachable


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m exo.testkit.node NODE_ID [exo arguments...]")
    node_id = NodeId(sys.argv[1])
    install(node_id)
    sys.argv = ["exo", *sys.argv[2:]]
    exo.main.main()


if __name__ == "__main__":
    main()
