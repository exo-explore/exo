from datetime import datetime, timezone

from exo.routing.topics import STATE_SNAPSHOTS
from exo.shared.apply import apply
from exo.shared.topology import Topology
from exo.shared.types.common import CommandId, ModelId, NodeId, SessionId, SystemId
from exo.shared.types.events import (
    IndexedEvent,
    InstanceCreated,
    NodeDownloadProgress,
    NodeGatheredInfo,
    RunnerStatusUpdated,
    StateSnapshot,
    TaskCreated,
    TopologyEdgeCreated,
)
from exo.shared.types.memory import Memory
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.profiling import (
    MemoryUsage,
    NetworkInterfaceInfo,
    NodeNetworkInfo,
    ThunderboltBridgeStatus,
)
from exo.shared.types.state import State
from exo.shared.types.tasks import TaskId, TaskStatus, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.topology import Connection, SocketConnection
from exo.shared.types.worker.downloads import DownloadCompleted
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runners import RunnerId, RunnerReady
from exo.utils.info_gatherer.info_gatherer import MiscData, NodeNetworkInterfaces
from exo.worker.tests.unittests.conftest import (
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)


def test_state_serialization_roundtrip() -> None:
    """Verify that State → JSON → State round-trip preserves topology."""

    # --- build a simple state ------------------------------------------------
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")

    connection = Connection(
        source=node_a,
        sink=node_b,
        edge=SocketConnection(
            sink_multiaddr=Multiaddr(address="/ip4/127.0.0.1/tcp/10001"),
        ),
    )

    state = State()
    state.topology.add_connection(connection)

    json_repr = state.model_dump_json()
    restored_state = State.model_validate_json(json_repr)

    assert (
        state.topology.to_snapshot().nodes
        == restored_state.topology.to_snapshot().nodes
    )
    assert set(state.topology.to_snapshot().connections) == set(
        restored_state.topology.to_snapshot().connections
    )
    assert restored_state.model_dump_json() == json_repr


def test_state_snapshot_survives_the_wire() -> None:
    """A snapshot must rebuild exactly the state it was taken from."""
    node_a, node_b = NodeId("node-a"), NodeId("node-b")
    model = ModelId("mlx-community/test-model")
    runner = RunnerId()
    instance_id = InstanceId()
    shard = get_pipeline_shard_metadata(model, device_rank=0)
    task_id = TaskId()
    now = str(datetime.now(tz=timezone.utc))
    events = [
        NodeGatheredInfo(node_id=node_a, when=now, info=MiscData(friendly_name="a")),
        NodeGatheredInfo(
            node_id=node_a,
            when=now,
            info=MemoryUsage.from_bytes(
                ram_total=1000, ram_available=500, swap_total=0, swap_available=0
            ),
        ),
        NodeGatheredInfo(
            node_id=node_b,
            when=now,
            info=NodeNetworkInterfaces(
                ifaces=[NetworkInterfaceInfo(name="en0", ip_address="10.0.0.2")]
            ),
        ),
        TopologyEdgeCreated(
            conn=Connection(
                source=node_a,
                sink=node_b,
                edge=SocketConnection(
                    sink_multiaddr=Multiaddr(address="/ip4/10.0.0.2/tcp/52414")
                ),
            )
        ),
        InstanceCreated(
            instance=get_mlx_ring_instance(
                instance_id, model, {node_a: runner}, {runner: shard}
            )
        ),
        RunnerStatusUpdated(runner_id=runner, runner_status=RunnerReady()),
        NodeDownloadProgress(
            download_progress=DownloadCompleted(
                node_id=node_a,
                shard_metadata=shard,
                total=Memory.from_mb(100),
                model_directory="/models/test-model",
            )
        ),
        TaskCreated(
            task_id=task_id,
            task=TextGeneration(
                task_id=task_id,
                command_id=CommandId(),
                instance_id=instance_id,
                task_status=TaskStatus.Running,
                task_params=TextGenerationTaskParams(
                    model=model,
                    input=[
                        InputMessage(role="user", content=InputMessageContent("hi"))
                    ],
                ),
            ),
        ),
    ]
    state = State()
    for idx, event in enumerate(events):
        state = apply(state, IndexedEvent(idx=idx, event=event))

    snapshot = StateSnapshot(
        session=SessionId(master_node_id=node_a, election_clock=3),
        requester=SystemId(),
        state=state,
    )
    restored = STATE_SNAPSHOTS.deserialize(STATE_SNAPSHOTS.serialize(snapshot))

    assert restored.session == snapshot.session
    assert restored.requester == snapshot.requester
    assert restored.state.last_event_applied_idx == len(events) - 1
    assert restored.state.model_dump_json() == state.model_dump_json()


def test_topology_serialization_ignores_edge_order() -> None:
    """A topology restored from a snapshot and then updated can hold the same edges in a
    different order than one built from events; both must serialize the same."""
    node_a, node_b = NodeId("node-a"), NodeId("node-b")
    edges = [
        SocketConnection(sink_multiaddr=Multiaddr(address=f"/ip4/10.0.0.{i}/tcp/52415"))
        for i in range(3)
    ]
    forward, backward = State(), State()
    for edge in edges:
        forward.topology.add_connection(
            Connection(source=node_a, sink=node_b, edge=edge)
        )
    for edge in reversed(edges):
        backward.topology.add_connection(
            Connection(source=node_a, sink=node_b, edge=edge)
        )

    assert forward.model_dump_json() == backward.model_dump_json()


def test_topology_serialization_ignores_node_order() -> None:
    """The graph reuses a removed node's slot, so once a node leaves and another joins, a
    topology restored from a snapshot lists its nodes and connections in a different order
    than the master's; both must serialize the same."""
    node_a, node_b, node_c, node_d, node_e = (
        NodeId(f"node-{name}") for name in "abcde"
    )
    edge = SocketConnection(sink_multiaddr=Multiaddr(address="/ip4/10.0.0.1/tcp/52415"))
    master = State()
    for node in (node_a, node_b, node_c, node_d):
        master.topology.add_node(node)
    master.topology.add_connection(Connection(source=node_c, sink=node_d, edge=edge))
    master.topology.remove_node(node_b)

    restored = State.model_validate_json(master.model_dump_json())
    for state in (master, restored):
        state.topology.add_node(node_e)
        state.topology.add_connection(Connection(source=node_e, sink=node_a, edge=edge))

    assert master.model_dump_json() == restored.model_dump_json()


def test_thunderbolt_bridge_cycles_ignore_graph_order() -> None:
    """Every node computes the bridge cycles in apply(); nodes whose graphs hold the same
    Thunderbolt ring in a different order must store the same cycles."""
    nodes = [NodeId(f"node-{name}") for name in "abc"]
    address = {node: f"169.254.0.{i}" for i, node in enumerate(nodes)}
    network = {
        node: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="bridge0",
                    ip_address=address[node],
                    interface_type="thunderbolt",
                )
            ]
        )
        for node in nodes
    }
    bridges = {
        node: ThunderboltBridgeStatus(enabled=True, exists=True) for node in nodes
    }

    def ring(order: list[NodeId]) -> Topology:
        topology = Topology()
        for node in order:
            topology.add_node(node)
        for source in order:
            for sink in order:
                if source != sink:
                    topology.add_connection(
                        Connection(
                            source=source,
                            sink=sink,
                            edge=SocketConnection(
                                sink_multiaddr=Multiaddr(
                                    address=f"/ip4/{address[sink]}/tcp/52415"
                                )
                            ),
                        )
                    )
        return topology

    forward = ring(nodes).get_thunderbolt_bridge_cycles(bridges, network)
    backward = ring(nodes[::-1]).get_thunderbolt_bridge_cycles(bridges, network)

    assert len(forward) == 5  # three pairs and the ring in each direction
    assert forward == backward
