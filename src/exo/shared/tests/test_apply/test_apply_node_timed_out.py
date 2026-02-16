from exo.shared.apply import apply_node_timed_out
from exo.shared.models.model_cards import ModelId
from exo.shared.tests.conftest import get_pipeline_shard_metadata
from exo.shared.types.common import NodeId
from exo.shared.types.events import NodeTimedOut
from exo.shared.types.state import State
from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import (
    RunnerId,
    RunnerReady,
    RunnerShutdown,
    ShardAssignments,
)
from exo.shared.types.worker.shards import ShardMetadata
from exo.worker.tests.constants import (
    INSTANCE_1_ID,
    INSTANCE_2_ID,
    MODEL_A_ID,
    MODEL_B_ID,
    NODE_A,
    NODE_B,
    RUNNER_1_ID,
    RUNNER_2_ID,
    RUNNER_3_ID,
)


def _make_instance(
    instance_id: InstanceId,
    model_id: ModelId,
    node_to_runner: dict[NodeId, RunnerId],
    runner_to_shard: dict[RunnerId, ShardMetadata],
) -> MlxRingInstance:
    return MlxRingInstance(
        instance_id=instance_id,
        shard_assignments=ShardAssignments(
            model_id=model_id,
            node_to_runner=node_to_runner,
            runner_to_shard=runner_to_shard,
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )


def test_node_timed_out_removes_its_runners():
    """When a node times out, its runners should be removed from state."""
    shard = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0)
    instance = _make_instance(
        INSTANCE_1_ID,
        MODEL_A_ID,
        {NODE_A: RUNNER_1_ID},
        {RUNNER_1_ID: shard},
    )
    state = State(
        instances={INSTANCE_1_ID: instance},
        runners={RUNNER_1_ID: RunnerShutdown()},
    )

    new_state = apply_node_timed_out(NodeTimedOut(node_id=NODE_A), state)

    assert RUNNER_1_ID not in new_state.runners


def test_node_timed_out_preserves_other_nodes_runners():
    """Timing out one node must not remove runners belonging to another node."""
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard2 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    instance = _make_instance(
        INSTANCE_1_ID,
        MODEL_A_ID,
        {NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        {RUNNER_1_ID: shard1, RUNNER_2_ID: shard2},
    )
    state = State(
        instances={INSTANCE_1_ID: instance},
        runners={RUNNER_1_ID: RunnerShutdown(), RUNNER_2_ID: RunnerReady()},
    )

    new_state = apply_node_timed_out(NodeTimedOut(node_id=NODE_A), state)

    assert RUNNER_1_ID not in new_state.runners
    assert RUNNER_2_ID in new_state.runners


def test_node_timed_out_cleans_runners_across_instances():
    """A node may have runners in multiple instances; all should be removed."""
    shard_a = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0)
    shard_b = get_pipeline_shard_metadata(MODEL_B_ID, device_rank=0)
    instance_1 = _make_instance(
        INSTANCE_1_ID,
        MODEL_A_ID,
        {NODE_A: RUNNER_1_ID},
        {RUNNER_1_ID: shard_a},
    )
    instance_2 = _make_instance(
        INSTANCE_2_ID,
        MODEL_B_ID,
        {NODE_A: RUNNER_2_ID},
        {RUNNER_2_ID: shard_b},
    )
    state = State(
        instances={INSTANCE_1_ID: instance_1, INSTANCE_2_ID: instance_2},
        runners={RUNNER_1_ID: RunnerShutdown(), RUNNER_2_ID: RunnerShutdown()},
    )

    new_state = apply_node_timed_out(NodeTimedOut(node_id=NODE_A), state)

    assert RUNNER_1_ID not in new_state.runners
    assert RUNNER_2_ID not in new_state.runners
    assert len(new_state.runners) == 0


def test_node_timed_out_no_runners_is_noop():
    """Timing out a node with no runners should not affect state.runners."""
    shard = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0)
    instance = _make_instance(
        INSTANCE_1_ID,
        MODEL_A_ID,
        {NODE_B: RUNNER_1_ID},
        {RUNNER_1_ID: shard},
    )
    state = State(
        instances={INSTANCE_1_ID: instance},
        runners={RUNNER_1_ID: RunnerReady()},
    )

    new_state = apply_node_timed_out(NodeTimedOut(node_id=NODE_A), state)

    assert RUNNER_1_ID in new_state.runners


def test_node_timed_out_removes_orphaned_runners_from_old_peer():
    """
    Simulate the stale runner bug: node restarts with a new peer ID,
    old peer ID times out. Runners from the old peer's instance should
    be cleaned up even if they're in RunnerShutdown state.
    """
    old_peer = NodeId("old-peer-id-that-restarted")
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard2 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    instance = _make_instance(
        INSTANCE_1_ID,
        MODEL_A_ID,
        {old_peer: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        {RUNNER_1_ID: shard1, RUNNER_2_ID: shard2},
    )
    # Runner 3 belongs to a different instance on the live node
    shard3 = get_pipeline_shard_metadata(MODEL_B_ID, device_rank=0)
    instance_2 = _make_instance(
        INSTANCE_2_ID,
        MODEL_B_ID,
        {NODE_B: RUNNER_3_ID},
        {RUNNER_3_ID: shard3},
    )
    state = State(
        instances={INSTANCE_1_ID: instance, INSTANCE_2_ID: instance_2},
        runners={
            RUNNER_1_ID: RunnerShutdown(),
            RUNNER_2_ID: RunnerReady(),
            RUNNER_3_ID: RunnerReady(),
        },
    )

    new_state = apply_node_timed_out(NodeTimedOut(node_id=old_peer), state)

    # Old peer's runner should be gone
    assert RUNNER_1_ID not in new_state.runners
    # Other node's runners should remain
    assert RUNNER_2_ID in new_state.runners
    assert RUNNER_3_ID in new_state.runners
