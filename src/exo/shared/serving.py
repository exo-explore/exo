"""Serving without the master: which instance serves a request, which node of it streams the
output, and where to reach that node. Pure functions of a node's own view of the state.

The API node that receives a request chooses the instance from its own state, sends the request
to the instance's output node over HTTP, and reads the output back on the same response
(exo.api.serving_client). The output node hands the task to every rank (exo.worker.serving).
The master only places instances.
"""

import ipaddress
import random
from collections.abc import Callable, Iterator, Mapping, Set
from typing import Final, Protocol, final

from anyio import BrokenResourceError, ClosedResourceError

from exo.master.placement_utils import find_ip_prioritised
from exo.routing.membership import WEDGE_TIMEOUT, MembershipChange, Reachability
from exo.shared.model_readiness import (
    instance_is_ready,
    instance_readiness,
    is_launched,
    model_readiness,
)
from exo.shared.models.model_cards import ModelId
from exo.shared.request_errors import (
    INSTANCE_GONE_BEFORE_START,
    INTERRUPTED,
    NO_INSTANCE_FOR_MODEL,
    PEER_FAILED,
    removal_message,
)
from exo.shared.types.common import NodeId
from exo.shared.types.state import State
from exo.shared.types.tasks import GenerationTask, ImageEdits, ImageGeneration
from exo.shared.types.topology import SocketConnection
from exo.shared.types.worker.instances import (
    Instance,
    InstanceId,
    InstanceRemovalReason,
)
from exo.shared.types.worker.runners import RunnerRunning
from exo.shared.types.worker.shards import is_primary_output_shard
from exo.utils.channels import Receiver, Sender, channel

# How long a node whose transport is up may go without answering before it is taken to be
# wedged: the membership's WEDGE_TIMEOUT (rust/networking/src/membership.rs, where it is
# derived), so a request and the membership agree on what "wedged" means. The API node waits this
# long for a line of a request's stream, and the output node this long for a rank to take a
# request or its cancellation. A pause shorter than the transport lease (10 s) is survived; a
# frozen, sleeping or lost node is ended sooner, as soon as the membership finds it unreachable.
WEDGE_SECONDS: Final = WEDGE_TIMEOUT.total_seconds()
# How often the output node writes a heartbeat line on a request's stream while it has nothing
# else to send (a long prompt being read, a rank being slow to take the request), so that a
# live node is never taken for a wedged one: ten heartbeats to one WEDGE_SECONDS.
STREAM_HEARTBEAT_INTERVAL: Final = WEDGE_SECONDS / 10


class NoInstanceForModelError(Exception):
    """No instance of the model can serve a request now; the message says why, for the user."""

    def __init__(self, message: str, retry_after_seconds: int | None = None):
        super().__init__(message)
        self.message = message
        self.retry_after_seconds = retry_after_seconds


def is_image_task(task: GenerationTask) -> bool:
    return isinstance(task, (ImageGeneration, ImageEdits))


def output_node(instance: Instance, *, images: bool) -> NodeId:
    """The node whose runner emits a request's output: device rank 0 for text; for images the
    primary output shard (the last pipeline stage)."""
    assignments = instance.shard_assignments
    for node_id, runner_id in assignments.node_to_runner.items():
        shard = assignments.runner_to_shard[runner_id]
        if images:
            if is_primary_output_shard(shard) or len(assignments.node_to_runner) == 1:
                return node_id
        elif shard.device_rank == 0:
            return node_id
    raise ValueError(f"instance {instance.instance_id} has no output shard")


def why_runner_stopped(
    state: State, instance: Instance, removal: InstanceRemovalReason | None
) -> str:
    """Why a runner of `instance` is being stopped, as its requests are told: a runner it works
    with is going (the instance is still in the state: the only other reason the worker stops
    one, exo.worker.plan), or its instance was removed, for `removal` (removal_message). A
    removal this node didn't see the reason of reads as the request being interrupted."""
    if instance.instance_id in state.instances:
        return PEER_FAILED
    if removal is not None:
        return removal_message(removal)
    return INTERRUPTED


def _prefill_only(state: State) -> set[InstanceId]:
    """Instances that only prefill for others (prefill side of a link, not a decode side)"""
    prefill: set[InstanceId] = set()
    for link in state.instance_links.values():
        prefill.update(link.prefill_instances)
    for link in state.instance_links.values():
        prefill.difference_update(link.decode_instances)
    return prefill


def candidate_instances(
    state: State,
    model: ModelId,
    in_flight: Mapping[InstanceId, int],
    usable: Callable[[Instance], bool],
) -> list[InstanceId]:
    """The instances that may serve a request for the model, best first.

    Only ready instances that aren't prefill-only and that `usable` accepts (the API node's
    check: it reaches every node of the instance, and has an address of its output node). Fewest
    requests in flight from this API node first,
    then those whose runners are not busy (every runner publishes Running and Ready as it starts
    and finishes work), then at random. Each API node only knows its own requests, so several
    API nodes balance less evenly than one router would. Raises NoInstanceForModelError, with
    what the model is doing, if there is none.
    """
    excluded = _prefill_only(state)
    of_model = [
        instance
        for instance_id, instance in state.instances.items()
        if instance.shard_assignments.model_id == model and instance_id not in excluded
    ]
    if not of_model:
        # A deployment being placed again: waited for, as a model that isn't ready is
        if (
            is_launched(state, model)
            and (not_ready := model_readiness(state, model)) is not None
        ):
            raise NoInstanceForModelError(
                not_ready.message, not_ready.retry_after_seconds
            )
        raise NoInstanceForModelError(f"{NO_INSTANCE_FOR_MODEL}{model}")
    ready = [
        instance for instance in of_model if instance_is_ready(instance, state.runners)
    ]
    if not ready:
        not_ready = [
            status
            for instance in of_model
            if (status := instance_readiness(state, instance)) is not None
        ]
        if not not_ready:
            raise NoInstanceForModelError(INSTANCE_GONE_BEFORE_START)
        best = min(
            not_ready, key=lambda status: (status.failed, status.retry_after_seconds)
        )
        raise NoInstanceForModelError(best.message, best.retry_after_seconds)
    usable_ready = [instance for instance in ready if usable(instance)]

    def busy(instance: Instance) -> bool:
        return any(
            isinstance(state.runners.get(runner_id), RunnerRunning)
            for runner_id in instance.shard_assignments.runner_to_shard
        )

    # Ties in random order: API nodes that have nothing in flight would otherwise all pick the
    # same instance (this node's view of which runners are busy lags)
    return [
        instance.instance_id
        for instance in sorted(
            random.sample(usable_ready, len(usable_ready)),
            key=lambda instance: (
                in_flight.get(instance.instance_id, 0),
                busy(instance),
            ),
        )
    ]


def prefill_endpoint_for(
    state: State, decode_instance_id: InstanceId, in_flight: Mapping[InstanceId, int]
) -> str | None:
    """Where a decode instance's runner fetches a prompt's KV cache: the prefill server of the
    least busy prefill instance linked to it, at an address its rank 0 can reach."""
    decode = state.instances.get(decode_instance_id)
    if decode is None:
        return None
    decode_node = next(iter(decode.shard_assignments.node_to_runner.keys()), None)
    if decode_node is None:
        return None

    sources: set[InstanceId] = set()
    for link in state.instance_links.values():
        if decode_instance_id in link.decode_instances:
            sources.update(link.prefill_instances)
    sources.discard(decode_instance_id)

    for source_id in sorted(sources, key=lambda source: in_flight.get(source, 0)):
        instance = state.instances.get(source_id)
        if instance is None:
            continue
        for node_id, runner_id in instance.shard_assignments.node_to_runner.items():
            port = state.prefill_server_ports.get(runner_id)
            if port is None:
                continue
            # For moving a KV cache, as for a ring: the fastest link first
            ip = find_ip_prioritised(
                decode_node, node_id, state.topology, state.node_network, ring=True
            )
            if ip is None:
                continue
            return f"{ip}:{port}"
    return None


def _edge_addresses(state: State, source: NodeId, sink: NodeId) -> Iterator[str]:
    """Addresses of sink that source has verified it reaches (its HTTP probe of sink's API)"""
    for connection in state.topology.get_all_connections_between(source, sink):
        if isinstance(connection, SocketConnection):
            yield connection.sink_multiaddr.ip_address


# Which of a node's addresses to try first. Over a TCP stream any link works; Thunderbolt's
# link-local addresses come last, as only directly cabled nodes reach them.
_ADDRESS_PRIORITY: Final = {
    "ethernet": 0,
    "wifi": 1,
    "unknown": 2,
    "maybe_ethernet": 3,
    "thunderbolt": 4,
}


def node_urls(state: State, source: NodeId, sink: NodeId, api_port: int) -> list[str]:
    """Base URLs at which `source` can try to reach `sink`'s API, best first: the addresses
    source has verified (its reachability probe of sink's API), then sink's other addresses.
    IPv4 only: the API listens on IPv4 (exo.api.listening_socket). Every node of a cluster
    serves its API on the same port, as the reachability probe assumes."""
    interfaces = state.node_network.get(sink)
    types = (
        {iface.ip_address: iface.interface_type for iface in interfaces.interfaces}
        if interfaces is not None
        else {}
    )
    verified = list(dict.fromkeys(_edge_addresses(state, source, sink)))
    others = [address for address in types if address not in verified]

    def rank(address: str) -> int:
        return _ADDRESS_PRIORITY.get(types.get(address, "unknown"), 2)

    def usable(address: str) -> bool:
        return _is_ipv4(address) and not _is_loopback(address)

    addresses = sorted(filter(usable, verified), key=rank) + sorted(
        filter(usable, others), key=rank
    )
    return [f"http://{address}:{api_port}" for address in addresses]


def _is_ipv4(address: str) -> bool:
    try:
        return ipaddress.ip_address(address).version == 4
    except ValueError:
        return False


def _bare(address: str) -> str:
    """An address without an IPv6 zone (fe80::1%en0) or IPv4-mapped prefix"""
    address = address.split("%", 1)[0]
    return address.removeprefix("::ffff:")


def _is_loopback(address: str) -> bool:
    try:
        return ipaddress.ip_address(_bare(address)).is_loopback
    except ValueError:
        return False


def member_addresses(state: State) -> Set[str]:
    """Every address of every node in this node's view of the cluster"""
    members = set(state.topology.list_nodes())
    return {
        _bare(iface.ip_address)
        for node_id, network in state.node_network.items()
        if node_id in members
        for iface in network.interfaces
    }


def is_member_address(state: State, address: str | None) -> bool:
    """Whether a connection from this address may use the cluster's internal routes: loopback,
    or an address of a node in the cluster. Anyone else on the network could otherwise put a
    task straight into a runner."""
    if address is None:
        return False
    return _is_loopback(address) or _bare(address) in member_addresses(state)


class MembershipChanges(Protocol):
    """What ServingReachability reads of the membership (exo.routing.membership.Membership)"""

    def is_reachable(self, node: NodeId) -> bool: ...

    def changes(self) -> Receiver[MembershipChange]: ...


@final
class ServingReachability(Reachability):
    """Which nodes this node reaches, for serving requests: the membership's view, except for
    the nodes it loses while this node is itself Resuming. This node was stopped (frozen or
    asleep), so its links lapsed while the peers were fine: the membership says such a loss
    says nothing about them. Each is judged once this node is back, and reported lost only if
    it still isn't Alive then. Otherwise a node waking from a pause longer than the lease ended
    every request it served or streamed."""

    def __init__(self, node_id: NodeId, membership: MembershipChanges):
        self._node_id = node_id
        self._membership = membership
        self._watchers: list[Sender[NodeId]] = []
        self._resuming = False
        self._lost_while_resuming: set[NodeId] = set()

    def is_reachable(self, node: NodeId) -> bool:
        return self._membership.is_reachable(node)

    def watch(self) -> Receiver[NodeId]:
        send, receive = channel[NodeId]()
        self._watchers.append(send)
        return receive

    async def run(self) -> None:
        try:
            with self._membership.changes() as changes:
                async for change in changes:
                    self._follow(change)
        finally:
            for watcher in self._watchers:
                watcher.close()

    def _follow(self, change: MembershipChange) -> None:
        if change.node_id == self._node_id:
            self._resuming = change.current == "Resuming"
            if not self._resuming:
                lost, self._lost_while_resuming = self._lost_while_resuming, set()
                for node in lost:
                    if not self._membership.is_reachable(node):
                        self._lose(node)
        elif change.left_alive:
            if self._resuming:
                self._lost_while_resuming.add(change.node_id)
            else:
                self._lose(change.node_id)

    def _lose(self, node: NodeId) -> None:
        for watcher in list(self._watchers):
            try:
                watcher.send_nowait(node)
            except (ClosedResourceError, BrokenResourceError):
                self._watchers.remove(watcher)
