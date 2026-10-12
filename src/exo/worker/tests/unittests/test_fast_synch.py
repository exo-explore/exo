"""MLX fast synch is used where it helps (RDMA instances), unless overridden."""

import pytest

from exo.shared.types.common import NodeId
from exo.shared.types.worker.instances import Instance, InstanceId, MlxJacclInstance
from exo.shared.types.worker.runners import RunnerId
from exo.worker.runner.bootstrap import use_fast_synch
from exo.worker.tests.constants import MODEL_A_ID
from exo.worker.tests.unittests.conftest import (
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
    get_shard_assignments,
)

RUNNER = RunnerId()
NODE = NodeId("node")


def ring() -> Instance:
    return get_mlx_ring_instance(
        instance_id=InstanceId(),
        model_id=MODEL_A_ID,
        node_to_runner={NODE: RUNNER},
        runner_to_shard={
            RUNNER: get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0)
        },
    )


def rdma() -> Instance:
    return MlxJacclInstance(
        instance_id=InstanceId(),
        shard_assignments=get_shard_assignments(
            MODEL_A_ID,
            {NODE: RUNNER},
            {RUNNER: get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0)},
        ),
        jaccl_devices=[[None]],
        jaccl_coordinators={NODE: "127.0.0.1:5000"},
    )


@pytest.mark.parametrize(
    ("instance", "override", "expected"),
    [
        (ring(), None, False),
        (rdma(), None, True),
        (ring(), "true", True),
        (rdma(), "false", False),
        (ring(), "false", False),
        (rdma(), "true", True),
    ],
)
def test_fast_synch_choice(
    instance: Instance, override: str | None, expected: bool
) -> None:
    assert use_fast_synch(instance, override) is expected
