"""MLX fast synch is used where it helps (RDMA instances), unless a model family turns it
off or it is overridden."""

import pytest

from exo.shared.types.common import NodeId
from exo.shared.types.worker.instances import Instance, InstanceId, MlxJacclInstance
from exo.shared.types.worker.runners import RunnerId
from exo.worker.engines.mlx.models import spec_registry
from exo.worker.engines.mlx.models.spec import FamilySpec
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


@pytest.mark.parametrize(
    ("family_fast_synch", "override", "expected"),
    [
        (False, None, False),
        (True, None, True),
        (None, None, True),
        (False, "true", True),
    ],
)
def test_a_family_can_turn_fast_synch_off(
    monkeypatch: pytest.MonkeyPatch,
    family_fast_synch: bool | None,
    override: str | None,
    expected: bool,
) -> None:
    family = (
        None
        if family_fast_synch is None
        else FamilySpec(
            name="family",
            model_types=frozenset({"family"}),
            fast_synch=family_fast_synch,
        )
    )

    def lookup(_model_type: str | None) -> FamilySpec | None:
        return family

    monkeypatch.setattr(spec_registry, "family_spec_for_model_type", lookup)
    assert use_fast_synch(rdma(), override) is expected
