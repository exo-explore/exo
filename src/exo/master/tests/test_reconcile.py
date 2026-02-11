from exo.master.reconcile import (
    find_unsatisfied_meta_instances,
    instance_connections_healthy,
    instance_runners_failed,
    instance_satisfies_meta_instance,
)
from exo.shared.apply import apply
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.common import Host, MetaInstanceId, NodeId
from exo.shared.types.events import (
    IndexedEvent,
    InstanceDeleted,
    MetaInstanceCreated,
    MetaInstanceDeleted,
)
from exo.shared.types.memory import Memory
from exo.shared.types.meta_instance import MetaInstance
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.state import State
from exo.shared.types.topology import Connection, SocketConnection
from exo.shared.types.worker.instances import (
    InstanceId,
    MlxRingInstance,
)
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerLoading,
    RunnerReady,
    RunnerShutdown,
    ShardAssignments,
)
from exo.shared.types.worker.shards import PipelineShardMetadata


def _model_card(model_id: str = "test-org/test-model") -> ModelCard:
    return ModelCard(
        model_id=ModelId(model_id),
        storage_size=Memory.from_kb(1000),
        n_layers=10,
        hidden_size=30,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )


def _topology(*node_ids: str, connect: bool = True) -> Topology:
    """Build a topology with nodes connected in a bidirectional ring with unique IPs.

    Node at index ``i`` gets IP ``10.0.0.{i+1}``. Edges go in both directions
    between consecutive nodes (including wrap-around).
    """
    t = Topology()
    nodes = [NodeId(n) for n in node_ids]
    for n in nodes:
        t.add_node(n)
    if connect and len(nodes) > 1:
        for i in range(len(nodes)):
            j = (i + 1) % len(nodes)
            t.add_connection(
                Connection(
                    source=nodes[i],
                    sink=nodes[j],
                    edge=SocketConnection(
                        sink_multiaddr=Multiaddr(
                            address=f"/ip4/10.0.0.{j + 1}/tcp/50000"
                        )
                    ),
                )
            )
            t.add_connection(
                Connection(
                    source=nodes[j],
                    sink=nodes[i],
                    edge=SocketConnection(
                        sink_multiaddr=Multiaddr(
                            address=f"/ip4/10.0.0.{i + 1}/tcp/50000"
                        )
                    ),
                )
            )
    return t


def _meta_instance(
    model_id: str = "test-org/test-model",
    *,
    min_nodes: int = 1,
    node_ids: list[NodeId] | None = None,
    meta_instance_id: MetaInstanceId | None = None,
) -> MetaInstance:
    return MetaInstance(
        meta_instance_id=meta_instance_id or MetaInstanceId(),
        model_id=ModelId(model_id),
        min_nodes=min_nodes,
        node_ids=node_ids,
    )


def _instance(
    model_id: str = "test-org/test-model",
    node_ids: list[str] | None = None,
    instance_id: InstanceId | None = None,
    meta_instance_id: MetaInstanceId | None = None,
) -> tuple[InstanceId, MlxRingInstance]:
    """Create a test instance with hosts_by_node matching ``_topology()`` IPs."""
    iid = instance_id or InstanceId()
    nodes = node_ids or ["node-a"]
    n = len(nodes)
    mc = _model_card(model_id)
    ephemeral_port = 50000
    node_to_runner = {NodeId(nd): RunnerId() for nd in nodes}
    runner_to_shard = {
        runner_id: PipelineShardMetadata(
            model_card=mc,
            device_rank=i,
            world_size=n,
            start_layer=0,
            end_layer=mc.n_layers,
            n_layers=mc.n_layers,
        )
        for i, runner_id in enumerate(node_to_runner.values())
    }
    # Build hosts_by_node with IPs matching _topology() convention:
    # node at index idx has IP 10.0.0.{idx+1}
    hosts_by_node: dict[NodeId, list[Host]] = {}
    for r, node_str in enumerate(nodes):
        hosts: list[Host] = []
        for idx in range(n):
            if idx == r:
                hosts.append(Host(ip="0.0.0.0", port=ephemeral_port))
            elif n > 1 and idx in ((r - 1) % n, (r + 1) % n):
                hosts.append(Host(ip=f"10.0.0.{idx + 1}", port=ephemeral_port))
            else:
                hosts.append(Host(ip="198.51.100.1", port=0))
        hosts_by_node[NodeId(node_str)] = hosts
    return iid, MlxRingInstance(
        instance_id=iid,
        shard_assignments=ShardAssignments(
            model_id=ModelId(model_id),
            runner_to_shard=runner_to_shard,
            node_to_runner=node_to_runner,
        ),
        hosts_by_node=hosts_by_node,
        ephemeral_port=ephemeral_port,
        meta_instance_id=meta_instance_id,
    )


# --- instance_satisfies_meta_instance (pure constraint matching) ---


def test_satisfies_matching_model():
    meta = _meta_instance()
    _, inst = _instance(node_ids=["node-a"])
    assert instance_satisfies_meta_instance(meta, inst) is True


def test_not_satisfies_wrong_model():
    meta = _meta_instance("test-org/model-a")
    _, inst = _instance("test-org/model-b")
    assert instance_satisfies_meta_instance(meta, inst) is False


def test_not_satisfies_missing_required_node():
    meta = _meta_instance(node_ids=[NodeId("node-c")])
    _, inst = _instance(node_ids=["node-a", "node-b"])
    assert instance_satisfies_meta_instance(meta, inst) is False


def test_not_satisfies_fewer_than_min_nodes():
    meta = _meta_instance(min_nodes=3)
    _, inst = _instance(node_ids=["node-a", "node-b"])
    assert instance_satisfies_meta_instance(meta, inst) is False


def test_satisfies_with_node_ids_specified():
    meta = _meta_instance(
        node_ids=[NodeId("node-a"), NodeId("node-b")], min_nodes=2
    )
    _, inst = _instance(node_ids=["node-a", "node-b", "node-c"])
    assert instance_satisfies_meta_instance(meta, inst) is True


# --- instance_connections_healthy ---


def test_healthy_single_node_present():
    _, inst = _instance(node_ids=["node-a"])
    topology = _topology("node-a")
    assert instance_connections_healthy(inst, topology) is True


def test_unhealthy_single_node_missing():
    _, inst = _instance(node_ids=["node-a"])
    topology = Topology()  # empty
    assert instance_connections_healthy(inst, topology) is False


def test_healthy_two_node_ring():
    _, inst = _instance(node_ids=["node-a", "node-b"])
    topology = _topology("node-a", "node-b")
    assert instance_connections_healthy(inst, topology) is True


def test_unhealthy_two_node_edge_removed():
    """Nodes present but edge removed — ring broken."""
    _, inst = _instance(node_ids=["node-a", "node-b"])
    topology = _topology("node-a", "node-b", connect=False)
    assert instance_connections_healthy(inst, topology) is False


def test_unhealthy_two_node_ip_changed():
    """Edge exists but with a different IP than instance was configured with."""
    _, inst = _instance(node_ids=["node-a", "node-b"])
    # Build topology with different IPs than _instance() expects
    topology = Topology()
    topology.add_node(NodeId("node-a"))
    topology.add_node(NodeId("node-b"))
    topology.add_connection(
        Connection(
            source=NodeId("node-a"),
            sink=NodeId("node-b"),
            edge=SocketConnection(
                sink_multiaddr=Multiaddr(address="/ip4/192.168.99.99/tcp/50000")
            ),
        )
    )
    topology.add_connection(
        Connection(
            source=NodeId("node-b"),
            sink=NodeId("node-a"),
            edge=SocketConnection(
                sink_multiaddr=Multiaddr(address="/ip4/192.168.99.98/tcp/50000")
            ),
        )
    )
    assert instance_connections_healthy(inst, topology) is False


def test_healthy_three_node_ring():
    _, inst = _instance(node_ids=["node-a", "node-b", "node-c"])
    topology = _topology("node-a", "node-b", "node-c")
    assert instance_connections_healthy(inst, topology) is True


def test_unhealthy_three_node_one_edge_removed():
    """Remove one edge from a three-node ring — instance unhealthy."""
    _, inst = _instance(node_ids=["node-a", "node-b", "node-c"])
    # Build topology with one direction of one edge missing
    topology = Topology()
    nodes = [NodeId("node-a"), NodeId("node-b"), NodeId("node-c")]
    for n in nodes:
        topology.add_node(n)
    # Add all edges except node-a → node-b
    topology.add_connection(
        Connection(
            source=nodes[1],
            sink=nodes[0],
            edge=SocketConnection(
                sink_multiaddr=Multiaddr(address="/ip4/10.0.0.1/tcp/50000")
            ),
        )
    )
    topology.add_connection(
        Connection(
            source=nodes[1],
            sink=nodes[2],
            edge=SocketConnection(
                sink_multiaddr=Multiaddr(address="/ip4/10.0.0.3/tcp/50000")
            ),
        )
    )
    topology.add_connection(
        Connection(
            source=nodes[2],
            sink=nodes[1],
            edge=SocketConnection(
                sink_multiaddr=Multiaddr(address="/ip4/10.0.0.2/tcp/50000")
            ),
        )
    )
    topology.add_connection(
        Connection(
            source=nodes[2],
            sink=nodes[0],
            edge=SocketConnection(
                sink_multiaddr=Multiaddr(address="/ip4/10.0.0.1/tcp/50000")
            ),
        )
    )
    topology.add_connection(
        Connection(
            source=nodes[0],
            sink=nodes[2],
            edge=SocketConnection(
                sink_multiaddr=Multiaddr(address="/ip4/10.0.0.3/tcp/50000")
            ),
        )
    )
    # Missing: node-a → node-b (ip 10.0.0.2)
    assert instance_connections_healthy(inst, topology) is False


def test_unhealthy_node_missing_from_topology():
    """Instance has a node that's not in the topology at all."""
    _, inst = _instance(node_ids=["node-a", "node-b"])
    topology = _topology("node-a")  # node-b not present
    assert instance_connections_healthy(inst, topology) is False


def test_healthy_extra_nodes_in_topology():
    """Extra nodes in topology don't affect instance health."""
    _, inst = _instance(node_ids=["node-a", "node-b"])
    topology = _topology("node-a", "node-b", "node-c")
    assert instance_connections_healthy(inst, topology) is True


# --- find_unsatisfied_meta_instances ---


def test_unsatisfied_no_meta_instances():
    result = find_unsatisfied_meta_instances({}, {}, Topology())
    assert list(result) == []


def test_unsatisfied_one_satisfied():
    meta = _meta_instance()
    id_a, inst_a = _instance(meta_instance_id=meta.meta_instance_id)
    topology = _topology("node-a")
    result = find_unsatisfied_meta_instances(
        {meta.meta_instance_id: meta},
        {id_a: inst_a},
        topology,
    )
    assert list(result) == []


def test_unsatisfied_one_not_satisfied():
    meta = _meta_instance("test-org/model-x")
    id_a, inst_a = _instance("test-org/model-y")
    topology = _topology("node-a")
    result = find_unsatisfied_meta_instances(
        {meta.meta_instance_id: meta}, {id_a: inst_a}, topology
    )
    assert list(result) == [meta]


def test_unsatisfied_mix():
    meta_satisfied = _meta_instance("test-org/model-a")
    meta_unsatisfied = _meta_instance("test-org/model-b")
    id_a, inst_a = _instance(
        "test-org/model-a", meta_instance_id=meta_satisfied.meta_instance_id
    )
    topology = _topology("node-a")
    result = find_unsatisfied_meta_instances(
        {
            meta_satisfied.meta_instance_id: meta_satisfied,
            meta_unsatisfied.meta_instance_id: meta_unsatisfied,
        },
        {id_a: inst_a},
        topology,
    )
    assert list(result) == [meta_unsatisfied]


def test_unsatisfied_node_disconnect():
    meta = _meta_instance()
    id_a, inst_a = _instance(
        node_ids=["node-a", "node-b"], meta_instance_id=meta.meta_instance_id
    )
    topology = _topology("node-a")  # node-b disconnected
    result = find_unsatisfied_meta_instances(
        {meta.meta_instance_id: meta},
        {id_a: inst_a},
        topology,
    )
    assert list(result) == [meta]


def test_unsatisfied_edge_break():
    """Instance exists but its connections broke — meta-instance becomes unsatisfied."""
    meta = _meta_instance()
    id_a, inst_a = _instance(
        node_ids=["node-a", "node-b"], meta_instance_id=meta.meta_instance_id
    )
    topology = _topology("node-a", "node-b", connect=False)  # nodes present, no edges
    result = find_unsatisfied_meta_instances(
        {meta.meta_instance_id: meta},
        {id_a: inst_a},
        topology,
    )
    assert list(result) == [meta]


def test_unsatisfied_idempotent():
    meta = _meta_instance("test-org/model-x")
    topology = _topology("node-a")
    meta_instances = {meta.meta_instance_id: meta}
    instances: dict[InstanceId, MlxRingInstance] = {}
    result_1 = list(
        find_unsatisfied_meta_instances(meta_instances, instances, topology)
    )
    result_2 = list(
        find_unsatisfied_meta_instances(meta_instances, instances, topology)
    )
    assert result_1 == result_2


def test_unsatisfied_exclusive_binding():
    """Two MetaInstances for the same model: one is bound via meta_instance_id, the other is unsatisfied."""
    meta_a = _meta_instance("test-org/model-x")
    meta_b = _meta_instance("test-org/model-x")
    id_inst, inst = _instance(
        "test-org/model-x", meta_instance_id=meta_a.meta_instance_id
    )
    topology = _topology("node-a")
    result = find_unsatisfied_meta_instances(
        {
            meta_a.meta_instance_id: meta_a,
            meta_b.meta_instance_id: meta_b,
        },
        {id_inst: inst},
        topology,
    )
    assert list(result) == [meta_b]


# --- apply handlers ---


def test_apply_meta_instance_created():
    state = State()
    meta = _meta_instance()
    event = MetaInstanceCreated(meta_instance=meta)
    new_state = apply(state, IndexedEvent(idx=0, event=event))
    assert meta.meta_instance_id in new_state.meta_instances
    assert new_state.meta_instances[meta.meta_instance_id] == meta


def test_apply_meta_instance_deleted():
    meta = _meta_instance()
    state = State(meta_instances={meta.meta_instance_id: meta})
    event = MetaInstanceDeleted(meta_instance_id=meta.meta_instance_id)
    new_state = apply(state, IndexedEvent(idx=0, event=event))
    assert meta.meta_instance_id not in new_state.meta_instances


def test_apply_meta_instance_deleted_clears_failure_info():
    meta = _meta_instance().model_copy(
        update={"consecutive_failures": 2, "last_failure_error": "OOM"}
    )
    state = State(meta_instances={meta.meta_instance_id: meta})
    event = MetaInstanceDeleted(meta_instance_id=meta.meta_instance_id)
    new_state = apply(state, IndexedEvent(idx=0, event=event))
    assert meta.meta_instance_id not in new_state.meta_instances


# --- instance_runners_failed ---


def test_runners_failed_all_failed():
    """All runners in RunnerFailed -> instance is failed."""
    _, inst = _instance(node_ids=["node-a", "node-b"])
    runners = {
        rid: RunnerFailed(error_message="OOM")
        for rid in inst.shard_assignments.node_to_runner.values()
    }
    is_failed, error = instance_runners_failed(inst, runners, {})
    assert is_failed is True
    assert error is not None
    assert "OOM" in error


def test_runners_failed_mixed_failed_shutdown():
    """One Failed + one Shutdown = failed."""
    _, inst = _instance(node_ids=["node-a", "node-b"])
    runner_ids = list(inst.shard_assignments.node_to_runner.values())
    runners = {
        runner_ids[0]: RunnerFailed(error_message="crash"),
        runner_ids[1]: RunnerShutdown(),
    }
    is_failed, error = instance_runners_failed(inst, runners, {})
    assert is_failed is True
    assert error is not None
    assert "crash" in error


def test_runners_not_failed_all_shutdown():
    """All Shutdown (graceful) = not a failure."""
    _, inst = _instance(node_ids=["node-a"])
    runners = {
        rid: RunnerShutdown()
        for rid in inst.shard_assignments.node_to_runner.values()
    }
    is_failed, _ = instance_runners_failed(inst, runners, {})
    assert is_failed is False


def test_runners_not_failed_still_active():
    """Some runners still active = not failed yet."""
    _, inst = _instance(node_ids=["node-a", "node-b"])
    runner_ids = list(inst.shard_assignments.node_to_runner.values())
    runners = {
        runner_ids[0]: RunnerFailed(error_message="OOM"),
        runner_ids[1]: RunnerLoading(),
    }
    is_failed, _ = instance_runners_failed(inst, runners, {})
    assert is_failed is False


def test_runners_not_failed_no_status():
    """Runner not yet reported = not failed."""
    _, inst = _instance(node_ids=["node-a"])
    is_failed, _ = instance_runners_failed(inst, {}, {})
    assert is_failed is False


def test_runners_not_failed_healthy():
    """Runners in Ready state = not failed."""
    _, inst = _instance(node_ids=["node-a"])
    runners = {
        rid: RunnerReady()
        for rid in inst.shard_assignments.node_to_runner.values()
    }
    is_failed, _ = instance_runners_failed(inst, runners, {})
    assert is_failed is False


# --- failure tracking in apply_instance_deleted ---


def test_apply_instance_deleted_tracks_failure():
    """InstanceDeleted with failure_error increments meta instance failure count."""
    meta = _meta_instance()
    iid, inst = _instance(
        node_ids=["node-a"], meta_instance_id=meta.meta_instance_id
    )
    state = State(
        meta_instances={meta.meta_instance_id: meta},
        instances={iid: inst},
    )
    event = InstanceDeleted(instance_id=iid, failure_error="Runner OOM")
    new_state = apply(state, IndexedEvent(idx=0, event=event))
    mi = new_state.meta_instances[meta.meta_instance_id]
    assert mi.consecutive_failures == 1
    assert mi.last_failure_error == "Runner OOM"
    assert mi.last_failure_at is not None


def test_apply_instance_deleted_increments_failure():
    """Subsequent failures increment the counter."""
    meta = _meta_instance().model_copy(
        update={"consecutive_failures": 2, "last_failure_error": "previous error"}
    )
    iid, inst = _instance(
        node_ids=["node-a"], meta_instance_id=meta.meta_instance_id
    )
    state = State(
        meta_instances={meta.meta_instance_id: meta},
        instances={iid: inst},
    )
    event = InstanceDeleted(instance_id=iid, failure_error="new error")
    new_state = apply(state, IndexedEvent(idx=0, event=event))
    mi = new_state.meta_instances[meta.meta_instance_id]
    assert mi.consecutive_failures == 3
    assert mi.last_failure_error == "new error"


def test_apply_instance_deleted_no_failure_no_tracking():
    """InstanceDeleted without failure_error does not track."""
    meta = _meta_instance()
    iid, inst = _instance(
        node_ids=["node-a"], meta_instance_id=meta.meta_instance_id
    )
    state = State(
        meta_instances={meta.meta_instance_id: meta},
        instances={iid: inst},
    )
    event = InstanceDeleted(instance_id=iid)
    new_state = apply(state, IndexedEvent(idx=0, event=event))
    mi = new_state.meta_instances[meta.meta_instance_id]
    assert mi.consecutive_failures == 0


def test_apply_instance_deleted_orphan_no_tracking():
    """InstanceDeleted for orphan instance (no meta_instance_id) does not track."""
    iid, inst = _instance(node_ids=["node-a"])
    state = State(instances={iid: inst})
    event = InstanceDeleted(instance_id=iid, failure_error="crash")
    new_state = apply(state, IndexedEvent(idx=0, event=event))
    assert len(new_state.meta_instances) == 0
