"""The runner keeps its wired weights resident across idle periods, unless told otherwise."""

import pytest

from exo.shared.types.common import NodeId
from exo.shared.types.worker.instances import Instance, InstanceId
from exo.shared.types.worker.runners import RunnerId
from exo.worker.runner.bootstrap import (
    RESIDENCY_REFRESH_DEFAULT_MS,
    RESIDENCY_REFRESH_ENV,
    configure_mlx_environment,
    residency_refresh_interval_ms,
)
from exo.worker.tests.constants import MODEL_A_ID
from exo.worker.tests.unittests.conftest import (
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)


def ring() -> Instance:
    runner, node = RunnerId(), NodeId("node")
    return get_mlx_ring_instance(
        instance_id=InstanceId(),
        model_id=MODEL_A_ID,
        node_to_runner={node: runner},
        runner_to_shard={
            runner: get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0)
        },
    )


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        (None, str(RESIDENCY_REFRESH_DEFAULT_MS)),
        ("", str(RESIDENCY_REFRESH_DEFAULT_MS)),
        ("0", "0"),
        ("250", "250"),
    ],
)
def test_residency_refresh_interval(override: str | None, expected: str) -> None:
    assert residency_refresh_interval_ms(override) == expected


def test_runner_environment_sets_the_refresh_interval() -> None:
    environ: dict[str, str] = {}
    configure_mlx_environment(ring(), environ)
    assert environ[RESIDENCY_REFRESH_ENV] == str(RESIDENCY_REFRESH_DEFAULT_MS)
    assert environ["MLX_METAL_FAST_SYNCH"] == "0"


def test_runner_environment_keeps_a_user_interval() -> None:
    environ = {RESIDENCY_REFRESH_ENV: "0", "EXO_FAST_SYNCH": "true"}
    configure_mlx_environment(ring(), environ)
    assert environ[RESIDENCY_REFRESH_ENV] == "0"
    assert environ["MLX_METAL_FAST_SYNCH"] == "1"
