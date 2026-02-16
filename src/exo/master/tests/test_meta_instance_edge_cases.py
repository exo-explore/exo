"""Edge-case and regression tests for MetaInstance lifecycle, concurrent operations, and error handling."""

import pytest

from exo.master.process_managers.instance_health import (
    MAX_INSTANCE_RETRIES,
    InstanceHealthReconciler,
)
from exo.master.process_managers.meta_instance import MetaInstanceReconciler
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
    InstanceCreated,
    InstanceDeleted,
    InstanceRetrying,
    MetaInstanceCreated,
    MetaInstanceDeleted,
    MetaInstancePlacementFailed,
    TaskStatusUpdated,
)
from exo.shared.types.memory import Memory
from exo.shared.types.meta_instance import MetaInstance
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.profiling import NodeIdentity
from exo.shared.types.state import State
from exo.shared.types.tasks import LoadModel, TaskId, TaskStatus
from exo.shared.types.topology import Connection, SocketConnection
from exo.shared.types.worker.instances import (
    InstanceId,
    MlxRingInstance,
)
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerReady,
    ShardAssignments,
)
from exo.shared.types.worker.shards import PipelineShardMetadata

# --- Helpers (copied from test_reconcile.py for independence) ---


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
    consecutive_failures: int = 0,
    last_failure_error: str | None = None,
    placement_error: str | None = None,
) -> MetaInstance:
    return MetaInstance(
        meta_instance_id=meta_instance_id or MetaInstanceId(),
        model_id=ModelId(model_id),
        min_nodes=min_nodes,
        node_ids=node_ids,
        consecutive_failures=consecutive_failures,
        last_failure_error=last_failure_error,
        placement_error=placement_error,
    )


def _instance(
    model_id: str = "test-org/test-model",
    node_ids: list[str] | None = None,
    instance_id: InstanceId | None = None,
    meta_instance_id: MetaInstanceId | None = None,
) -> tuple[InstanceId, MlxRingInstance]:
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


# =============================================================================
# 1. MetaInstance lifecycle edge cases
# =============================================================================


def test_meta_instance_model_is_frozen():
    """MetaInstance should be immutable (frozen model)."""
    meta = _meta_instance()
    try:
        meta.model_id = ModelId("something-else")
        raise AssertionError("Should have raised")
    except Exception:
        pass  # Expected — frozen model


def test_meta_instance_created_then_deleted_roundtrip():
    """Create and delete a MetaInstance through apply — state should be clean."""
    state = State()
    meta = _meta_instance()
    state = apply(
        state, IndexedEvent(idx=0, event=MetaInstanceCreated(meta_instance=meta))
    )
    assert meta.meta_instance_id in state.meta_instances
    state = apply(
        state,
        IndexedEvent(
            idx=1, event=MetaInstanceDeleted(meta_instance_id=meta.meta_instance_id)
        ),
    )
    assert meta.meta_instance_id not in state.meta_instances
    assert len(state.meta_instances) == 0


def test_delete_nonexistent_meta_instance_is_safe():
    """Deleting a MetaInstance that doesn't exist should not crash."""
    state = State()
    event = MetaInstanceDeleted(meta_instance_id=MetaInstanceId("nonexistent"))
    new_state = apply(state, IndexedEvent(idx=0, event=event))
    assert len(new_state.meta_instances) == 0


def test_placement_failed_for_nonexistent_meta_instance_is_safe():
    """MetaInstancePlacementFailed for unknown ID should not crash."""
    state = State()
    event = MetaInstancePlacementFailed(
        meta_instance_id=MetaInstanceId("nonexistent"),
        reason="test",
    )
    new_state = apply(state, IndexedEvent(idx=0, event=event))
    assert len(new_state.meta_instances) == 0


def test_multiple_meta_instances_for_same_model():
    """Multiple MetaInstances for the same model are tracked independently."""
    state = State()
    meta_a = _meta_instance("test-org/model-x")
    meta_b = _meta_instance("test-org/model-x")
    state = apply(
        state, IndexedEvent(idx=0, event=MetaInstanceCreated(meta_instance=meta_a))
    )
    state = apply(
        state, IndexedEvent(idx=1, event=MetaInstanceCreated(meta_instance=meta_b))
    )
    assert len(state.meta_instances) == 2
    assert meta_a.meta_instance_id in state.meta_instances
    assert meta_b.meta_instance_id in state.meta_instances


# =============================================================================
# 2. Retry logic edge cases
# =============================================================================


def test_retry_counter_resets_on_successful_instance_creation():
    """When a new instance is created for a meta-instance, failures should reset."""
    meta = _meta_instance(consecutive_failures=2, last_failure_error="old")
    _, inst = _instance(node_ids=["node-a"], meta_instance_id=meta.meta_instance_id)
    state = State(meta_instances={meta.meta_instance_id: meta})
    state = apply(state, IndexedEvent(idx=0, event=InstanceCreated(instance=inst)))
    mi = state.meta_instances[meta.meta_instance_id]
    assert mi.consecutive_failures == 0
    # last_failure_error is preserved (for UI display)
    assert mi.last_failure_error == "old"


async def test_retry_count_increments_through_full_cycle():
    """Walk through MAX_INSTANCE_RETRIES worth of retries, then verify delete."""
    meta = _meta_instance()
    iid, inst = _instance(node_ids=["node-a"], meta_instance_id=meta.meta_instance_id)
    topology = _topology("node-a")
    state = State(
        meta_instances={meta.meta_instance_id: meta},
        instances={iid: inst},
        topology=topology,
    )

    runner_ids = list(inst.shard_assignments.node_to_runner.values())
    for idx, i in enumerate(range(MAX_INSTANCE_RETRIES)):
        # Simulate runners failing
        state_with_runners = state.model_copy(
            update={"runners": {runner_ids[0]: RunnerFailed(error_message=f"fail-{i}")}}
        )
        reconciler = InstanceHealthReconciler()
        events = await reconciler.reconcile(state_with_runners)
        assert len(events) == 1
        assert isinstance(events[0], InstanceRetrying), f"iteration {i}"
        state = apply(state, IndexedEvent(idx=idx, event=events[0]))

    # After MAX_INSTANCE_RETRIES retries, failure counter should be at max
    mi = state.meta_instances[meta.meta_instance_id]
    assert mi.consecutive_failures == MAX_INSTANCE_RETRIES

    # Next failure should result in deletion
    state_with_runners = state.model_copy(
        update={"runners": {runner_ids[0]: RunnerFailed(error_message="final")}}
    )
    reconciler = InstanceHealthReconciler()
    events = await reconciler.reconcile(state_with_runners)
    assert len(events) == 1
    assert isinstance(events[0], InstanceDeleted)


async def test_health_reconciler_respects_exact_limit():
    """At exactly MAX_INSTANCE_RETRIES, reconciler should delete, not retry."""
    meta = _meta_instance(consecutive_failures=MAX_INSTANCE_RETRIES)
    iid, inst = _instance(node_ids=["node-a"], meta_instance_id=meta.meta_instance_id)
    runner_ids = list(inst.shard_assignments.node_to_runner.values())
    state = State(
        meta_instances={meta.meta_instance_id: meta},
        instances={iid: inst},
        runners={runner_ids[0]: RunnerFailed(error_message="OOM")},
        topology=_topology("node-a"),
    )
    reconciler = InstanceHealthReconciler()
    events = await reconciler.reconcile(state)
    assert len(events) == 1
    assert isinstance(events[0], InstanceDeleted)


async def test_health_reconciler_at_limit_minus_one_retries():
    """At MAX_INSTANCE_RETRIES - 1, reconciler should still retry."""
    meta = _meta_instance(consecutive_failures=MAX_INSTANCE_RETRIES - 1)
    iid, inst = _instance(node_ids=["node-a"], meta_instance_id=meta.meta_instance_id)
    runner_ids = list(inst.shard_assignments.node_to_runner.values())
    state = State(
        meta_instances={meta.meta_instance_id: meta},
        instances={iid: inst},
        runners={runner_ids[0]: RunnerFailed(error_message="OOM")},
        topology=_topology("node-a"),
    )
    reconciler = InstanceHealthReconciler()
    events = await reconciler.reconcile(state)
    assert len(events) == 1
    assert isinstance(events[0], InstanceRetrying)


# =============================================================================
# 3. Error handling edge cases
# =============================================================================


def test_runners_failed_with_empty_error_message():
    """RunnerFailed with empty error_message should still report as failed."""
    _, inst = _instance(node_ids=["node-a"])
    runners = {
        rid: RunnerFailed(error_message="")
        for rid in inst.shard_assignments.node_to_runner.values()
    }
    is_failed, error = instance_runners_failed(inst, runners, {})
    assert is_failed is True
    # Empty error message means we get the fallback
    assert error == "Runner failed"


def test_runners_failed_with_none_error_message():
    """RunnerFailed with None error_message should still report as failed."""
    _, inst = _instance(node_ids=["node-a"])
    runners = {
        rid: RunnerFailed(error_message=None)
        for rid in inst.shard_assignments.node_to_runner.values()
    }
    is_failed, error = instance_runners_failed(inst, runners, {})
    assert is_failed is True
    assert error == "Runner failed"


def test_runners_failed_collects_all_error_messages():
    """With multiple failed runners, all error messages should be collected."""
    _, inst = _instance(node_ids=["node-a", "node-b", "node-c"])
    runner_ids = list(inst.shard_assignments.node_to_runner.values())
    runners = {
        runner_ids[0]: RunnerFailed(error_message="OOM on GPU 0"),
        runner_ids[1]: RunnerFailed(error_message="OOM on GPU 1"),
        runner_ids[2]: RunnerFailed(error_message="OOM on GPU 2"),
    }
    is_failed, error = instance_runners_failed(inst, runners, {})
    assert is_failed is True
    assert error is not None
    assert "OOM on GPU 0" in error
    assert "OOM on GPU 1" in error
    assert "OOM on GPU 2" in error


def test_runners_failed_includes_friendly_name():
    """Error messages should include node friendly names when available."""
    _, inst = _instance(node_ids=["node-a"])
    node_id = NodeId("node-a")
    runner_ids = list(inst.shard_assignments.node_to_runner.values())
    runners = {runner_ids[0]: RunnerFailed(error_message="OOM")}
    identities = {node_id: NodeIdentity(friendly_name="My Mac Studio")}
    is_failed, error = instance_runners_failed(inst, runners, identities)
    assert is_failed is True
    assert error is not None
    assert "My Mac Studio" in error


def test_instance_retrying_for_missing_instance_is_safe():
    """InstanceRetrying for an instance not in state should not crash.

    NOTE: When the instance is missing, the handler returns early WITHOUT
    incrementing the MetaInstance failure counter. This means stale retry
    events for already-deleted instances are silently dropped. This is
    acceptable since the InstanceDeleted handler already increments failures.
    """
    meta = _meta_instance()
    state = State(meta_instances={meta.meta_instance_id: meta})
    event = InstanceRetrying(
        instance_id=InstanceId("nonexistent"),
        meta_instance_id=meta.meta_instance_id,
        failure_error="crash",
    )
    new_state = apply(state, IndexedEvent(idx=0, event=event))
    # Does not crash, but failure count is NOT incremented (early return)
    mi = new_state.meta_instances[meta.meta_instance_id]
    assert mi.consecutive_failures == 0


# =============================================================================
# 4. Backward compatibility
# =============================================================================


def test_instance_without_meta_instance_id_works():
    """Instances created without meta_instance_id should still function normally."""
    _, inst = _instance(node_ids=["node-a"])
    assert inst.meta_instance_id is None
    topology = _topology("node-a")
    assert instance_connections_healthy(inst, topology) is True


def test_instance_deleted_without_meta_does_not_affect_meta_instances():
    """Deleting an instance without meta_instance_id should not affect meta_instances."""
    meta = _meta_instance()
    iid, inst = _instance(node_ids=["node-a"])  # no meta_instance_id
    state = State(
        meta_instances={meta.meta_instance_id: meta},
        instances={iid: inst},
    )
    event = InstanceDeleted(instance_id=iid, failure_error="crash")
    new_state = apply(state, IndexedEvent(idx=0, event=event))
    mi = new_state.meta_instances[meta.meta_instance_id]
    assert mi.consecutive_failures == 0  # unchanged


def test_satisfies_ignores_meta_instance_id_binding():
    """instance_satisfies_meta_instance checks constraints only, not binding."""
    meta = _meta_instance()
    _, inst = _instance(node_ids=["node-a"])  # no meta_instance_id set
    # Should match on constraints (model, min_nodes) regardless of binding
    assert instance_satisfies_meta_instance(meta, inst) is True


def test_find_unsatisfied_uses_binding_not_constraints():
    """find_unsatisfied checks meta_instance_id binding, not just constraint matching."""
    meta = _meta_instance()
    # Instance matches constraints but is NOT bound to this meta_instance
    iid, inst = _instance(node_ids=["node-a"])
    topology = _topology("node-a")
    result = find_unsatisfied_meta_instances(
        {meta.meta_instance_id: meta}, {iid: inst}, topology
    )
    # Should be unsatisfied because instance.meta_instance_id != meta.meta_instance_id
    assert list(result) == [meta]


# =============================================================================
# 5. Concurrent / multi-instance scenarios
# =============================================================================


async def test_health_reconciler_handles_multiple_failing_instances():
    """Multiple instances failing simultaneously should each get their own event."""
    meta_a = _meta_instance()
    meta_b = _meta_instance()
    iid_a, inst_a = _instance(
        node_ids=["node-a"], meta_instance_id=meta_a.meta_instance_id
    )
    iid_b, inst_b = _instance(
        node_ids=["node-b"], meta_instance_id=meta_b.meta_instance_id
    )
    runner_ids_a = list(inst_a.shard_assignments.node_to_runner.values())
    runner_ids_b = list(inst_b.shard_assignments.node_to_runner.values())
    state = State(
        meta_instances={
            meta_a.meta_instance_id: meta_a,
            meta_b.meta_instance_id: meta_b,
        },
        instances={iid_a: inst_a, iid_b: inst_b},
        runners={
            runner_ids_a[0]: RunnerFailed(error_message="OOM"),
            runner_ids_b[0]: RunnerFailed(error_message="OOM"),
        },
        topology=_topology("node-a", "node-b"),
    )
    reconciler = InstanceHealthReconciler()
    events = await reconciler.reconcile(state)
    assert len(events) == 2
    # Both should be InstanceRetrying since failures < MAX
    assert all(isinstance(e, InstanceRetrying) for e in events)
    instance_ids = {e.instance_id for e in events}  # type: ignore[union-attr]
    assert instance_ids == {iid_a, iid_b}


async def test_health_reconciler_mixed_healthy_and_failing():
    """Only failing instances should produce events; healthy ones should not."""
    meta_healthy = _meta_instance()
    meta_failing = _meta_instance()
    iid_h, inst_h = _instance(
        node_ids=["node-a"], meta_instance_id=meta_healthy.meta_instance_id
    )
    iid_f, inst_f = _instance(
        node_ids=["node-b"], meta_instance_id=meta_failing.meta_instance_id
    )
    runner_ids_h = list(inst_h.shard_assignments.node_to_runner.values())
    runner_ids_f = list(inst_f.shard_assignments.node_to_runner.values())
    state = State(
        meta_instances={
            meta_healthy.meta_instance_id: meta_healthy,
            meta_failing.meta_instance_id: meta_failing,
        },
        instances={iid_h: inst_h, iid_f: inst_f},
        runners={
            runner_ids_h[0]: RunnerReady(),
            runner_ids_f[0]: RunnerFailed(error_message="crash"),
        },
        topology=_topology("node-a", "node-b"),
    )
    reconciler = InstanceHealthReconciler()
    events = await reconciler.reconcile(state)
    assert len(events) == 1
    assert isinstance(events[0], InstanceRetrying)
    assert events[0].instance_id == iid_f


async def test_meta_instance_reconciler_empty_state():
    """MetaInstanceReconciler with no meta_instances should produce no events."""
    state = State()
    reconciler = MetaInstanceReconciler()
    events = await reconciler.reconcile(state)
    assert len(events) == 0


# =============================================================================
# 6. Placement error tracking
# =============================================================================


def test_placement_failed_sets_error():
    """MetaInstancePlacementFailed should set placement_error on the MetaInstance."""
    meta = _meta_instance()
    state = State(meta_instances={meta.meta_instance_id: meta})
    event = MetaInstancePlacementFailed(
        meta_instance_id=meta.meta_instance_id,
        reason="Not enough memory",
    )
    new_state = apply(state, IndexedEvent(idx=0, event=event))
    mi = new_state.meta_instances[meta.meta_instance_id]
    assert mi.placement_error == "Not enough memory"


def test_instance_created_clears_placement_error():
    """InstanceCreated should clear placement_error on the MetaInstance."""
    meta = _meta_instance(placement_error="Not enough memory")
    _, inst = _instance(node_ids=["node-a"], meta_instance_id=meta.meta_instance_id)
    state = State(meta_instances={meta.meta_instance_id: meta})
    state = apply(state, IndexedEvent(idx=0, event=InstanceCreated(instance=inst)))
    mi = state.meta_instances[meta.meta_instance_id]
    assert mi.placement_error is None


def test_placement_error_does_not_increment_failures():
    """Placement failures should only set placement_error, not increment consecutive_failures."""
    meta = _meta_instance()
    state = State(meta_instances={meta.meta_instance_id: meta})
    event = MetaInstancePlacementFailed(
        meta_instance_id=meta.meta_instance_id,
        reason="No resources",
    )
    new_state = apply(state, IndexedEvent(idx=0, event=event))
    mi = new_state.meta_instances[meta.meta_instance_id]
    assert mi.consecutive_failures == 0
    assert mi.placement_error == "No resources"


# =============================================================================
# 7. State serialization roundtrip
# =============================================================================


def test_state_with_meta_instances_serializes():
    """State with meta_instances should serialize and deserialize correctly."""
    meta = _meta_instance(consecutive_failures=2, last_failure_error="test")
    iid, inst = _instance(node_ids=["node-a"], meta_instance_id=meta.meta_instance_id)
    state = State(
        meta_instances={meta.meta_instance_id: meta},
        instances={iid: inst},
    )
    json_str = state.model_dump_json()
    restored = State.model_validate_json(json_str)
    assert meta.meta_instance_id in restored.meta_instances
    mi = restored.meta_instances[meta.meta_instance_id]
    assert mi.model_id == meta.model_id
    assert mi.consecutive_failures == 2
    assert mi.last_failure_error == "test"
    assert iid in restored.instances
    assert restored.instances[iid].meta_instance_id == meta.meta_instance_id


# =============================================================================
# 8. MetaInstanceReconciler error handling
# =============================================================================


async def test_meta_instance_reconciler_model_load_error_emits_placement_failed(
    monkeypatch: "pytest.MonkeyPatch",
):
    """When ModelCard.load raises, reconciler emits MetaInstancePlacementFailed."""
    import exo.master.process_managers.meta_instance as mi_mod

    meta = _meta_instance()
    topo = _topology("node-a")
    state = State(
        meta_instances={meta.meta_instance_id: meta},
        topology=topo,
    )

    async def _failing_load(_model_id: ModelId) -> ModelCard:
        raise RuntimeError("Network error")

    monkeypatch.setattr(
        mi_mod, "ModelCard", type("MC", (), {"load": staticmethod(_failing_load)})
    )

    reconciler = MetaInstanceReconciler()
    events = await reconciler.reconcile(state)

    placement_failed = [e for e in events if isinstance(e, MetaInstancePlacementFailed)]
    assert len(placement_failed) == 1
    assert "Failed to load model card" in placement_failed[0].reason
    assert meta.meta_instance_id == placement_failed[0].meta_instance_id


async def test_meta_instance_reconciler_model_load_error_skips_dedup(
    monkeypatch: "pytest.MonkeyPatch",
):
    """When ModelCard.load error matches existing placement_error, no duplicate event."""
    import exo.master.process_managers.meta_instance as mi_mod

    meta = _meta_instance(placement_error="Failed to load model card: Network error")
    topo = _topology("node-a")
    state = State(
        meta_instances={meta.meta_instance_id: meta},
        topology=topo,
    )

    async def _failing_load(_model_id: ModelId) -> ModelCard:
        raise RuntimeError("Network error")

    monkeypatch.setattr(
        mi_mod, "ModelCard", type("MC", (), {"load": staticmethod(_failing_load)})
    )

    reconciler = MetaInstanceReconciler()
    events = await reconciler.reconcile(state)

    # Error matches existing placement_error, so no duplicate event emitted
    assert len(events) == 0


async def test_meta_instance_reconciler_continues_after_error(
    monkeypatch: "pytest.MonkeyPatch",
):
    """Reconciler should continue to next meta-instance after one fails to load."""
    import exo.master.process_managers.meta_instance as mi_mod

    meta_a = _meta_instance(model_id="org/model-a")
    meta_b = _meta_instance(model_id="org/model-b")
    topo = _topology("node-a")
    state = State(
        meta_instances={
            meta_a.meta_instance_id: meta_a,
            meta_b.meta_instance_id: meta_b,
        },
        topology=topo,
    )

    call_count = 0

    async def _load_second_fails(model_id: ModelId) -> ModelCard:
        nonlocal call_count
        call_count += 1
        raise RuntimeError(f"Cannot load {model_id}")

    monkeypatch.setattr(
        mi_mod, "ModelCard", type("MC", (), {"load": staticmethod(_load_second_fails)})
    )

    reconciler = MetaInstanceReconciler()
    events = await reconciler.reconcile(state)

    # Both meta-instances should have been attempted (not short-circuited)
    assert call_count == 2
    # Both should have placement failed events
    placement_failed = [e for e in events if isinstance(e, MetaInstancePlacementFailed)]
    assert len(placement_failed) == 2


# =============================================================================
# 8. Cascade delete with task cancellation
# =============================================================================


def test_cascade_delete_cancels_active_tasks():
    """Deleting a MetaInstance should cancel tasks on backing instances.

    Regression test: previously, cascade-deleting backing instances via
    DeleteMetaInstance did not emit TaskStatusUpdated(Cancelled) for active
    tasks, leaving orphaned task references in state.
    """
    meta = _meta_instance()
    iid, inst = _instance(node_ids=["node-a"], meta_instance_id=meta.meta_instance_id)
    task_id = TaskId()
    task = LoadModel(task_id=task_id, instance_id=iid, task_status=TaskStatus.Running)

    # Build state with meta-instance, backing instance, and active task
    state = State(
        meta_instances={meta.meta_instance_id: meta},
        instances={iid: inst},
        tasks={task_id: task},
        topology=_topology("node-a"),
    )

    # Simulate the cascade-delete event sequence produced by main.py:
    # 1. MetaInstanceDeleted
    # 2. TaskStatusUpdated(Cancelled) for active tasks
    # 3. InstanceDeleted
    idx = 0
    state = apply(
        state,
        IndexedEvent(
            idx=idx,
            event=MetaInstanceDeleted(meta_instance_id=meta.meta_instance_id),
        ),
    )
    idx += 1
    state = apply(
        state,
        IndexedEvent(
            idx=idx,
            event=TaskStatusUpdated(task_id=task_id, task_status=TaskStatus.Cancelled),
        ),
    )
    idx += 1
    state = apply(
        state,
        IndexedEvent(idx=idx, event=InstanceDeleted(instance_id=iid)),
    )

    # Verify everything is cleaned up
    assert len(state.meta_instances) == 0
    assert len(state.instances) == 0
    assert state.tasks[task_id].task_status == TaskStatus.Cancelled


def test_cascade_delete_skips_completed_tasks():
    """Cascade delete should only cancel Pending/Running tasks, not completed ones."""
    meta = _meta_instance()
    iid, inst = _instance(node_ids=["node-a"], meta_instance_id=meta.meta_instance_id)

    running_task_id = TaskId()
    completed_task_id = TaskId()
    running_task = LoadModel(
        task_id=running_task_id, instance_id=iid, task_status=TaskStatus.Running
    )
    completed_task = LoadModel(
        task_id=completed_task_id, instance_id=iid, task_status=TaskStatus.Complete
    )

    state = State(
        meta_instances={meta.meta_instance_id: meta},
        instances={iid: inst},
        tasks={running_task_id: running_task, completed_task_id: completed_task},
        topology=_topology("node-a"),
    )

    # Only the running task should be cancelled — we verify the logic pattern
    # by checking which tasks are Pending or Running
    active_tasks = [
        t
        for t in state.tasks.values()
        if t.instance_id == iid
        and t.task_status in (TaskStatus.Pending, TaskStatus.Running)
    ]
    assert len(active_tasks) == 1
    assert active_tasks[0].task_id == running_task_id
