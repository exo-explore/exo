# pyright: reportPrivateUsage=false

from exo.shared.types.common import ModelId, NodeId
from exo.shared.types.state import State
from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import ShardAssignments
from exo.utils.keyed_backoff import KeyedBackoff
from exo.worker.main import Worker


def _make_instance(instance_id: InstanceId) -> MlxRingInstance:
    return MlxRingInstance(
        instance_id=instance_id,
        shard_assignments=ShardAssignments(
            model_id=ModelId("test-model"),
            node_to_runner={},
            runner_to_shard={},
        ),
        hosts_by_node={NodeId("node-1"): []},
        ephemeral_port=1,
    )


def test_worker_reconciles_instance_backoff_from_state() -> None:
    live_instance_id = InstanceId("inst-live")
    deleted_instance_id = InstanceId("inst-deleted")
    worker = object.__new__(Worker)
    worker.state = State(instances={live_instance_id: _make_instance(live_instance_id)})
    worker._instance_backoff = KeyedBackoff[InstanceId]()
    worker._instance_backoff.record_attempt(live_instance_id)
    worker._instance_backoff.record_attempt(deleted_instance_id)

    worker._reconcile_instance_backoff_once()

    assert worker._instance_backoff.attempts(live_instance_id) == 1
    assert worker._instance_backoff.attempts(deleted_instance_id) == 0
