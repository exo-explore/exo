from exo.shared.apply import apply_instance_deleted
from exo.shared.models.model_cards import ModelId
from exo.shared.tests.conftest import get_pipeline_shard_metadata
from exo.shared.types.common import NodeId
from exo.shared.types.events import InstanceDeleted
from exo.shared.types.state import State
from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import (
    RunnerId,
    RunnerReady,
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


def test_instance_deleted_removes_runners():
    """Deleting an instance must also remove its runner entries from state."""
    shard = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0)
    instance = _make_instance(
        INSTANCE_1_ID,
        MODEL_A_ID,
        {NODE_A: RUNNER_1_ID},
        {RUNNER_1_ID: shard},
    )
    state = State(
        instances={INSTANCE_1_ID: instance},
        runners={RUNNER_1_ID: RunnerReady()},
    )

    new_state = apply_instance_deleted(
        InstanceDeleted(instance_id=INSTANCE_1_ID), state
    )

    assert INSTANCE_1_ID not in new_state.instances
    assert RUNNER_1_ID not in new_state.runners


def test_instance_deleted_removes_only_its_runners():
    """Deleting one instance must not remove runners belonging to another."""
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
        {NODE_B: RUNNER_2_ID},
        {RUNNER_2_ID: shard_b},
    )
    state = State(
        instances={INSTANCE_1_ID: instance_1, INSTANCE_2_ID: instance_2},
        runners={RUNNER_1_ID: RunnerReady(), RUNNER_2_ID: RunnerReady()},
    )

    new_state = apply_instance_deleted(
        InstanceDeleted(instance_id=INSTANCE_1_ID), state
    )

    assert INSTANCE_1_ID not in new_state.instances
    assert RUNNER_1_ID not in new_state.runners
    # Instance 2 and its runner must remain
    assert INSTANCE_2_ID in new_state.instances
    assert RUNNER_2_ID in new_state.runners


def test_instance_deleted_multi_node_removes_all_runners():
    """Deleting a multi-node instance removes all of its runners."""
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
        runners={RUNNER_1_ID: RunnerReady(), RUNNER_2_ID: RunnerReady()},
    )

    new_state = apply_instance_deleted(
        InstanceDeleted(instance_id=INSTANCE_1_ID), state
    )

    assert INSTANCE_1_ID not in new_state.instances
    assert RUNNER_1_ID not in new_state.runners
    assert RUNNER_2_ID not in new_state.runners
    assert len(new_state.runners) == 0


def test_instance_deleted_unknown_id_is_noop_for_runners():
    """Deleting a non-existent instance should not affect runners."""
    shard = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0)
    instance = _make_instance(
        INSTANCE_1_ID,
        MODEL_A_ID,
        {NODE_A: RUNNER_1_ID},
        {RUNNER_1_ID: shard},
    )
    unknown_id = InstanceId("99999999-9999-4999-8999-999999999999")
    state = State(
        instances={INSTANCE_1_ID: instance},
        runners={RUNNER_1_ID: RunnerReady()},
    )

    new_state = apply_instance_deleted(InstanceDeleted(instance_id=unknown_id), state)

    # Everything should remain untouched
    assert INSTANCE_1_ID in new_state.instances
    assert RUNNER_1_ID in new_state.runners
