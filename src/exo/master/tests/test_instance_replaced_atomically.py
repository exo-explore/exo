from exo.master.placement import get_transition_events
from exo.master.placement_utils import (
    check_higher_priority_connection_upgrade,
    get_mlx_ring_hosts_by_node,
)
from exo.shared.apply import apply
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.common import CommandId, Host, NodeId
from exo.shared.types.events import (
    IndexedEvent,
    InstanceDeleted,
    InstanceReplacedAtomically,
    TaskStatusUpdated,
)
from exo.shared.types.instance_link import InstanceLink, InstanceLinkId
from exo.shared.types.memory import Memory
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.profiling import NetworkInterfaceInfo, NodeNetworkInfo
from exo.shared.types.state import State
from exo.shared.types.tasks import TaskId, TaskStatus, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.topology import Connection, Cycle, SocketConnection
from exo.shared.types.worker.instances import (
    InstanceId,
    MlxRingInstance,
)
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata


def _make_node_network(node_id: NodeId, wifi_ip: str, tb_ip: str) -> NodeNetworkInfo:
    return NodeNetworkInfo(
        interfaces=[
            NetworkInterfaceInfo(name="en0", ip_address=wifi_ip, interface_type="wifi"),
            NetworkInterfaceInfo(
                name="bridge0", ip_address=tb_ip, interface_type="thunderbolt"
            ),
        ]
    )


def _make_socket_conn(src: NodeId, sink: NodeId, ip: str) -> Connection:
    return Connection(
        source=src,
        sink=sink,
        edge=SocketConnection(sink_multiaddr=Multiaddr(address=f"/ip4/{ip}/tcp/50000")),
    )


from exo.shared.types.backends import Backend


def _make_two_node_ring_instance(
    node_a: NodeId, node_b: NodeId, hosts_by_node: dict[NodeId, list[Host]]
) -> MlxRingInstance:
    runner_a = RunnerId()
    runner_b = RunnerId()
    model_card = ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_kb(1000),
        n_layers=10,
        hidden_size=30,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )
    shard_a = PipelineShardMetadata(
        model_card=model_card,
        device_rank=0,
        world_size=2,
        start_layer=0,
        end_layer=5,
        n_layers=10,
    )
    shard_b = PipelineShardMetadata(
        model_card=model_card,
        device_rank=1,
        world_size=2,
        start_layer=5,
        end_layer=10,
        n_layers=10,
    )
    return MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=ModelId("test-model"),
            node_to_runner={node_a: runner_a, node_b: runner_b},
            runner_to_shard={runner_a: shard_a, runner_b: shard_b},
        ),
        hosts_by_node=hosts_by_node,
        ephemeral_port=50000,
    )


def test_check_higher_priority_connection_upgrade_from_wifi_to_thunderbolt() -> None:
    """When nodes are initially connected over WiFi and a Thunderbolt connection is later
    hotplugged, check_higher_priority_connection_upgrade detects the upgrade and generates
    a new MlxRingInstance with the high-speed connection.
    """
    node_a = NodeId()
    node_b = NodeId()

    wifi_ip_a = "192.168.1.10"
    wifi_ip_b = "192.168.1.20"
    tb_ip_a = "169.254.1.10"
    tb_ip_b = "169.254.1.20"

    node_network = {
        node_a: _make_node_network(node_a, wifi_ip_a, tb_ip_a),
        node_b: _make_node_network(node_b, wifi_ip_b, tb_ip_b),
    }

    # Initial topology: only WiFi connections exist
    topology = Topology()
    topology.add_connection(_make_socket_conn(node_a, node_b, wifi_ip_b))
    topology.add_connection(_make_socket_conn(node_b, node_a, wifi_ip_a))

    initial_hosts = get_mlx_ring_hosts_by_node(
        selected_cycle=Cycle(node_ids=[node_a, node_b]),
        cycle_digraph=topology,
        ephemeral_port=50000,
        node_network=node_network,
    )

    # Initial instance uses WiFi IPs
    instance = _make_two_node_ring_instance(node_a, node_b, initial_hosts)
    assert instance.hosts_by_node[node_a][1].ip == wifi_ip_b
    assert instance.hosts_by_node[node_b][0].ip == wifi_ip_a

    # No upgrade yet because topology hasn't changed
    assert (
        check_higher_priority_connection_upgrade(instance, topology, node_network)
        is None
    )

    # Hotplug event: Thunderbolt connection is established
    topology.add_connection(_make_socket_conn(node_a, node_b, tb_ip_b))
    topology.add_connection(_make_socket_conn(node_b, node_a, tb_ip_a))

    # Master detects that Thunderbolt (priority 0) is available!
    upgraded = check_higher_priority_connection_upgrade(
        instance, topology, node_network
    )
    assert upgraded is not None
    assert upgraded.instance_id != instance.instance_id
    assert isinstance(upgraded, MlxRingInstance)
    # The upgraded instance uses Thunderbolt IPs!
    assert upgraded.hosts_by_node[node_a][1].ip == tb_ip_b
    assert upgraded.hosts_by_node[node_b][0].ip == tb_ip_a


def test_get_transition_events_emits_instance_replaced_atomically_without_cancelling_tasks() -> (
    None
):
    """When an instance is upgraded to higher-priority connection, get_transition_events
    must emit InstanceReplacedAtomically rather than InstanceDeleted + InstanceCreated,
    and must NEVER cancel active/pending tasks.
    """
    node_a = NodeId()
    node_b = NodeId()

    initial_hosts = {
        node_a: [Host(ip="0.0.0.0", port=50000), Host(ip="192.168.1.20", port=50000)],
        node_b: [Host(ip="192.168.1.10", port=50000), Host(ip="0.0.0.0", port=50000)],
    }
    upgraded_hosts = {
        node_a: [Host(ip="0.0.0.0", port=50000), Host(ip="169.254.1.20", port=50000)],
        node_b: [Host(ip="169.254.1.10", port=50000), Host(ip="0.0.0.0", port=50000)],
    }

    old_instance = _make_two_node_ring_instance(node_a, node_b, initial_hosts)
    new_instance = old_instance.model_copy(
        update={
            "instance_id": InstanceId(),
            "hosts_by_node": upgraded_hosts,
        }
    )

    # Active task running on old_instance
    running_task = TextGeneration(
        task_id=TaskId(),
        command_id=CommandId(),
        instance_id=old_instance.instance_id,
        task_status=TaskStatus.Running,
        task_params=TextGenerationTaskParams(
            model=ModelId("test-model"),
            input=[InputMessage(role="user", content=InputMessageContent("hello"))],
        ),
    )

    current_instances = {old_instance.instance_id: old_instance}
    target_instances = {new_instance.instance_id: new_instance}
    tasks = {running_task.task_id: running_task}

    events = get_transition_events(current_instances, target_instances, tasks)

    # Assert exactly one atomic replacement event
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, InstanceReplacedAtomically)
    assert event.old_instance_id == old_instance.instance_id
    assert event.new_instance.instance_id == new_instance.instance_id

    # Crucial assertion: no task cancellation or instance deletion!
    assert not any(isinstance(e, TaskStatusUpdated) for e in events)
    assert not any(isinstance(e, InstanceDeleted) for e in events)


def test_apply_instance_replaced_atomically_updates_state_and_preserves_tasks() -> None:
    """Applying InstanceReplacedAtomically replaces the instance in state, preserves
    active runners, and seamlessly migrates tasks to the new instance ID.
    """
    node_a = NodeId()
    node_b = NodeId()

    initial_hosts = {
        node_a: [Host(ip="0.0.0.0", port=50000), Host(ip="192.168.1.20", port=50000)],
        node_b: [Host(ip="192.168.1.10", port=50000), Host(ip="0.0.0.0", port=50000)],
    }
    upgraded_hosts = {
        node_a: [Host(ip="0.0.0.0", port=50000), Host(ip="169.254.1.20", port=50000)],
        node_b: [Host(ip="169.254.1.10", port=50000), Host(ip="0.0.0.0", port=50000)],
    }

    old_instance = _make_two_node_ring_instance(node_a, node_b, initial_hosts)
    new_instance = old_instance.model_copy(
        update={
            "instance_id": InstanceId(),
            "hosts_by_node": upgraded_hosts,
        }
    )

    task = TextGeneration(
        task_id=TaskId(),
        command_id=CommandId(),
        instance_id=old_instance.instance_id,
        task_status=TaskStatus.Running,
        task_params=TextGenerationTaskParams(
            model=ModelId("test-model"),
            input=[InputMessage(role="user", content=InputMessageContent("hello"))],
        ),
    )

    link = InstanceLink(
        link_id=InstanceLinkId(),
        prefill_instances=[old_instance.instance_id],
        decode_instances=[old_instance.instance_id],
    )

    state = State(
        instances={old_instance.instance_id: old_instance},
        tasks={task.task_id: task},
        instance_links={link.link_id: link},
        last_event_applied_idx=0,
    )

    event = InstanceReplacedAtomically(
        old_instance_id=old_instance.instance_id,
        new_instance=new_instance,
    )

    new_state = apply(state, IndexedEvent(event=event, idx=1))

    # 1. Instances map updated
    assert old_instance.instance_id not in new_state.instances
    assert new_instance.instance_id in new_state.instances
    assert new_state.instances[new_instance.instance_id].hosts_by_node == upgraded_hosts

    # 2. Tasks seamlessly migrated without cancellation!
    migrated_task = new_state.tasks[task.task_id]
    assert migrated_task.instance_id == new_instance.instance_id
    assert migrated_task.task_status == TaskStatus.Running

    # 3. InstanceLinks re-pointed
    assert new_state.instance_links[link.link_id].prefill_instances == [
        new_instance.instance_id
    ]
    assert new_state.instance_links[link.link_id].decode_instances == [
        new_instance.instance_id
    ]
