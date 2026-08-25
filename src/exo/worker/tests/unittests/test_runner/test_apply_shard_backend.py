import mlx.core as mx
import pytest

from exo.shared.types.backends import Backend
from exo.worker.runner.bootstrap import apply_shard_backend


@pytest.fixture(autouse=True)
def restore_default_device():
    device_before = mx.default_device()
    yield
    mx.set_default_device(device_before)


def test_apply_shard_backend_cpu_sets_default_device():
    apply_shard_backend(Backend.MlxCpu)

    assert mx.default_device() == mx.Device(mx.cpu)


def test_apply_shard_backend_unassigned_keeps_default_device():
    device_before = mx.default_device()

    apply_shard_backend(None)

    assert mx.default_device() == device_before


def test_apply_shard_backend_non_mlx_backend_keeps_default_device():
    device_before = mx.default_device()

    apply_shard_backend(Backend.Vllm)

    assert mx.default_device() == device_before
