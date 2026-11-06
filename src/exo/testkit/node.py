"""Run one exo node for the harness: `python -m exo.testkit.node NODE_ID [exo arguments...]`.

This is exo's own `exo.main.main()`, with these changes made in this process before it starts:

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
4. A node reaches another's API (to serve a request without the master) through the simulated
   network's proxy for the pair, so a cut applies to it too. A test may also shorten the wedge
   bound (TESTKIT_WEDGE_SECONDS), or make the nodes refuse every delivery of a request
   (TESTKIT_REFUSE_DELIVERIES=1). A build without serving without the master runs without these.

Everything else - discovery, zenoh, election, master, worker, API, downloads, runners - is the
production code, unchanged.
"""

import os
import sys
from collections.abc import AsyncGenerator, Mapping

import anyio
import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response

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


def node_urls(
    _state: object, source: NodeId, sink: NodeId, _api_port: int
) -> list[str]:
    """Where `source` reaches `sink`'s API: the pair's proxy in the simulated network (exo's
    assumes every node serves its API at the same port)"""
    port = read_table().api_routes.get(source, {}).get(sink)
    return [] if port is None else [f"http://[::1]:{port}"]


# The data plane's wedge bound (30 s), shortened so a test can freeze a node longer than it but
# not long enough for the master to remove it (15 s)
WEDGE_SECONDS_ENV = "TESTKIT_WEDGE_SECONDS"
# 1: answer every delivery of a request to the runner with a 403, as a member check that
# doesn't know the sender would
REFUSE_DELIVERIES_ENV = "TESTKIT_REFUSE_DELIVERIES"


# Serving without the master's modules: a build without them (before it, a bisect, a revert)
# runs without its hooks
_SERVING_MODULES = (
    "exo.api.serving_client",
    "exo.shared.serving",
    "exo.worker.serving",
    "exo.api.cluster_routes",
)


def _install_serving_hooks() -> None:
    try:
        import exo.api.serving_client as serving_client
        import exo.shared.serving as serving
        import exo.worker.serving as worker_serving
        from exo.api.cluster_routes import ClusterRoutes
    except ModuleNotFoundError as error:
        if error.name not in _SERVING_MODULES:
            raise
        return
    serving.node_urls = node_urls
    if seconds := os.environ.get(WEDGE_SECONDS_ENV):
        for module in (worker_serving, serving_client):
            module.WEDGE_SECONDS = float(seconds)  # pyright: ignore[reportAttributeAccessIssue]
            module.STREAM_HEARTBEAT_INTERVAL = float(seconds) / 10  # pyright: ignore[reportAttributeAccessIssue]
    if os.environ.get(REFUSE_DELIVERIES_ENV) == "1":
        member_only = ClusterRoutes._member_only  # pyright: ignore[reportPrivateUsage]

        def refusing(self: ClusterRoutes, request: Request) -> Response | None:
            if request.url.path == worker_serving.DELIVER_PATH:
                return JSONResponse(status_code=403, content={"detail": "not a member"})
            return member_only(self, request)

        setattr(ClusterRoutes, "_member_only", refusing)  # noqa: B010


def install(node_id: NodeId) -> None:
    exo.main.get_node_zid = lambda: node_id
    supervisor.entrypoint = stub_entrypoint
    net_profile.check_reachability = _probe_for(node_id)
    worker_main.check_reachable = check_reachable
    _install_serving_hooks()


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m exo.testkit.node NODE_ID [exo arguments...]")
    node_id = NodeId(sys.argv[1])
    install(node_id)
    sys.argv = ["exo", *sys.argv[2:]]
    exo.main.main()


if __name__ == "__main__":
    main()
