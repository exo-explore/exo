"""State-driven reconciler for hardware and link profiles.

Every tick the manager reads the current State and decides what's missing or
stale, then probes it. This is the same controller pattern used by
Kubernetes: the desired state is "every node has a fresh GPU profile and
every outgoing topology edge has a fresh link profile". Reactive code that
fired off probes in response to specific events would be wrong here because
EXO replays the event log on master changes — replays must not have side
effects beyond updating state.

Cancellation order: `shutdown()` cancels the manager's task group, which
in turn cancels in-flight probe tasks. Since the worker shares its task
group with this manager (see `Worker.run`), shutting down the worker shuts
down the manager.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import anyio
import httpx
from anyio import fail_after
from loguru import logger

from exo.shared.types.common import NodeId
from exo.shared.types.profiling import (
    NodeRdmaLinkProfile,
    NodeSocketLinkProfile,
)
from exo.shared.types.state import State
from exo.shared.types.topology import (
    Connection,
    RDMAConnection,
    SocketConnection,
)
from exo.utils.channels import Sender
from exo.utils.info_gatherer.info_gatherer import GatheredInfo
from exo.utils.profilers.gpu_profiler import GpuProfile
from exo.utils.profilers.link_profiler import (
    PROBE_TIMEOUT_SECONDS,
    RDMALinkProfile,
    SocketLinkProfile,
)
from exo.utils.profilers.rdma_probe import RdmaProbeBusyError
from exo.utils.task_group import TaskGroup

GPU_TTL = timedelta(hours=1)
SOCKET_LINK_TTL = timedelta(minutes=5)
RDMA_LINK_TTL = timedelta(hours=6)
RECONCILE_TICK_SECONDS = 15.0
GPU_PROBE_HARD_TIMEOUT_SECONDS = 60.0
SOCKET_PROBE_HARD_TIMEOUT_SECONDS = 30.0
RDMA_PROBE_HARD_TIMEOUT_SECONDS = 90.0


# (source, sink, transport, edge_discriminator) — uniquely identifies one edge
# we are currently probing. transport is "socket" | "rdma"; the discriminator
# is the sink IP for socket edges, or the (source_iface, sink_iface) tuple
# for RDMA edges.
LinkKey = tuple[NodeId, NodeId, str, str]


@dataclass
class ProfilerManager:
    info_sender: Sender[GatheredInfo]
    node_id: NodeId
    api_port: int
    state_view: Callable[[], State]
    _tg: TaskGroup = field(init=False, default_factory=TaskGroup)
    _gpu_in_flight: bool = field(init=False, default=False)
    _link_in_flight: set[LinkKey] = field(init=False, default_factory=set)

    async def run(self) -> None:
        async with self._tg as tg:
            tg.start_soon(self._reconcile_gpu, RECONCILE_TICK_SECONDS)
            tg.start_soon(self._reconcile_links, RECONCILE_TICK_SECONDS)

    def shutdown(self) -> None:
        self._tg.cancel_tasks()

    # ----- GPU --------------------------------------------------------------

    async def _reconcile_gpu(self, tick_seconds: float) -> None:
        while True:
            try:
                self._maybe_start_gpu_probe()
            except Exception as e:
                logger.opt(exception=e).warning("GPU reconcile error")
            await anyio.sleep(tick_seconds)

    def _maybe_start_gpu_probe(self) -> None:
        state = self.state_view()
        if self._gpu_in_flight:
            return
        if state.runners:
            return
        existing = state.node_gpu_profile.get(self.node_id)
        if existing is not None and not _is_stale(existing.measured_at, GPU_TTL):
            return
        self._gpu_in_flight = True
        self._tg.start_soon(self._do_gpu_probe)

    async def _do_gpu_probe(self) -> None:
        try:
            with fail_after(GPU_PROBE_HARD_TIMEOUT_SECONDS):
                profile = await GpuProfile.measure()
            if profile is not None:
                await self.info_sender.send(profile)
        except Exception as e:
            logger.opt(exception=e).warning("GPU probe failed")
        finally:
            self._gpu_in_flight = False

    # ----- Links ------------------------------------------------------------

    async def _reconcile_links(self, tick_seconds: float) -> None:
        timeout = httpx.Timeout(timeout=PROBE_TIMEOUT_SECONDS)
        async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
            while True:
                try:
                    self._maybe_start_link_probes(client)
                except Exception as e:
                    logger.opt(exception=e).warning("Link reconcile error")
                await anyio.sleep(tick_seconds)

    def _maybe_start_link_probes(self, client: httpx.AsyncClient) -> None:
        state = self.state_view()
        for connection in state.topology.out_edges(self.node_id):
            self._maybe_start_one_link_probe(client, state, connection)

    def _maybe_start_one_link_probe(
        self, client: httpx.AsyncClient, state: State, connection: Connection
    ) -> None:
        edge = connection.edge
        sink = connection.sink
        existing_profiles = list(
            state.node_link_profiles.get(self.node_id, {}).get(sink, ())
        )

        match edge:
            case SocketConnection():
                key: LinkKey = (
                    self.node_id,
                    sink,
                    "socket",
                    edge.sink_multiaddr.ip_address,
                )
                if key in self._link_in_flight:
                    return
                fresh = any(
                    isinstance(p, NodeSocketLinkProfile)
                    and p.sink_ip == edge.sink_multiaddr.ip_address
                    and not _is_stale(p.measured_at, SOCKET_LINK_TTL)
                    for p in existing_profiles
                )
                if fresh:
                    return
                self._link_in_flight.add(key)
                self._tg.start_soon(
                    self._do_socket_probe,
                    client,
                    key,
                    sink,
                    edge.sink_multiaddr.ip_address,
                )
            case RDMAConnection():
                if state.runners:
                    return
                key = (
                    self.node_id,
                    sink,
                    "rdma",
                    f"{edge.source_rdma_iface}/{edge.sink_rdma_iface}",
                )
                if key in self._link_in_flight:
                    return
                fresh = any(
                    isinstance(p, NodeRdmaLinkProfile)
                    and p.source_rdma_iface == edge.source_rdma_iface
                    and p.sink_rdma_iface == edge.sink_rdma_iface
                    and p.upload_mbps is not None
                    and p.download_mbps is not None
                    and not _is_stale(p.measured_at, RDMA_LINK_TTL)
                    for p in existing_profiles
                )
                if fresh:
                    return
                coordinator_ip = _resolve_coordinator_ip(state, self.node_id, sink)
                sink_ip = _resolve_socket_sink_ip(state, self.node_id, sink)
                if coordinator_ip is None or sink_ip is None:
                    # No reachable socket path on which to coordinate jaccl;
                    # skip until reachability info catches up.
                    return
                self._link_in_flight.add(key)
                self._tg.start_soon(
                    self._do_rdma_probe,
                    client,
                    key,
                    sink,
                    sink_ip,
                    coordinator_ip,
                    edge,
                )

    async def _do_socket_probe(
        self,
        client: httpx.AsyncClient,
        key: LinkKey,
        sink_node_id: NodeId,
        sink_ip: str,
    ) -> None:
        try:
            with fail_after(SOCKET_PROBE_HARD_TIMEOUT_SECONDS):
                profile = await SocketLinkProfile.measure(
                    client=client,
                    sink_ip=sink_ip,
                    expected_sink_node_id=sink_node_id,
                    api_port=self.api_port,
                )
            if profile is not None:
                await self.info_sender.send(profile)
        except Exception as e:
            logger.opt(exception=e).warning(
                f"Socket link probe to {sink_node_id} via {sink_ip} failed"
            )
        finally:
            self._link_in_flight.discard(key)

    async def _do_rdma_probe(
        self,
        client: httpx.AsyncClient,
        key: LinkKey,
        sink_node_id: NodeId,
        sink_ip: str,
        coordinator_ip: str,
        edge: RDMAConnection,
    ) -> None:
        try:
            with fail_after(RDMA_PROBE_HARD_TIMEOUT_SECONDS):
                profile = await RDMALinkProfile.measure(
                    client=client,
                    sink_ip=sink_ip,
                    sink_node_id=sink_node_id,
                    api_port=self.api_port,
                    source_rdma_iface=edge.source_rdma_iface,
                    sink_rdma_iface=edge.sink_rdma_iface,
                    coordinator_ip=coordinator_ip,
                )
            if profile is not None:
                await self.info_sender.send(profile)
        except RdmaProbeBusyError:
            # Local lock held — try again next tick.
            pass
        except Exception as e:
            logger.opt(exception=e).warning(
                f"RDMA link probe to {sink_node_id} via {coordinator_ip} failed"
            )
        finally:
            self._link_in_flight.discard(key)


def _is_stale(measured_at: datetime, ttl: timedelta) -> bool:
    if measured_at.tzinfo is None:
        measured_at = measured_at.replace(tzinfo=timezone.utc)
    return datetime.now(tz=timezone.utc) - measured_at > ttl


def _resolve_socket_sink_ip(state: State, source: NodeId, sink: NodeId) -> str | None:
    """Pick an IP from the topology to reach `sink` from `source`.

    Used as the address for the HTTP rendezvous request when initiating an
    RDMA probe. Any reachable socket edge will do — the OS picks the route.
    """
    for connection in state.topology.out_edges(source):
        if connection.sink != sink:
            continue
        if isinstance(connection.edge, SocketConnection):
            return connection.edge.sink_multiaddr.ip_address
    return None


def _resolve_coordinator_ip(state: State, source: NodeId, sink: NodeId) -> str | None:
    """Find an IP of `source` that `sink` can reach, for the jaccl coordinator.

    We invert the direction: the topology stores `(sink → source)` edges from
    sink's perspective, so the sink_multiaddr on those edges is *source's* IP
    as the sink sees it.
    """
    for connection in state.topology.out_edges(sink):
        if connection.sink != source:
            continue
        if isinstance(connection.edge, SocketConnection):
            return connection.edge.sink_multiaddr.ip_address
    return None
