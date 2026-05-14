# pyright: reportPrivateUsage=false

import pytest

from exo.worker.engines.mlx.utils_mlx import (
    _available_memory_from_macmon_output,
    get_mlx_force_oom_size,
)

_MLX_FORCE_OOM_DTYPE_BYTES = 4
_MLX_FORCE_OOM_LIVE_MATRICES = 3


def _target_live_bytes(available_ram: int) -> int:
    return -(-(available_ram * 11) // 10)


def _allocation_bytes(size: int) -> int:
    return _MLX_FORCE_OOM_LIVE_MATRICES * size * size * _MLX_FORCE_OOM_DTYPE_BYTES


def test_get_mlx_force_oom_size_targets_slightly_above_available_memory() -> None:
    available_ram = 16 * 1024**3

    size = get_mlx_force_oom_size(available_ram)

    assert _allocation_bytes(size) >= _target_live_bytes(available_ram)
    assert _allocation_bytes(size - 1) < _target_live_bytes(available_ram)
    assert 39_000 <= size <= 41_000


def test_get_mlx_force_oom_size_scales_to_max_studio_memory() -> None:
    available_ram = 400 * 1024**3

    size = get_mlx_force_oom_size(available_ram)

    assert 195_000 <= size <= 205_000


@pytest.mark.parametrize("available_ram", [0, -1])
def test_get_mlx_force_oom_size_requires_positive_available_ram(
    available_ram: int,
) -> None:
    with pytest.raises(ValueError, match="available_ram must be positive"):
        get_mlx_force_oom_size(available_ram)


def test_available_memory_from_macmon_output_uses_first_sample() -> None:
    output = """
{"timestamp":"2026-05-14T12:00:00Z","temp":{"cpu_temp_avg":45.0,"gpu_temp_avg":46.0},"memory":{"ram_total":1000,"ram_usage":275,"swap_total":500,"swap_usage":125},"ecpu_usage":[1000,1.0],"pcpu_usage":[2000,2.0],"gpu_usage":[1200,3.0],"all_power":1.0,"ane_power":2.0,"cpu_power":3.0,"gpu_power":4.0,"gpu_ram_power":5.0,"ram_power":6.0,"sys_power":7.0}
{"timestamp":"2026-05-14T12:00:01Z","temp":{"cpu_temp_avg":45.0,"gpu_temp_avg":46.0},"memory":{"ram_total":1000,"ram_usage":999,"swap_total":500,"swap_usage":125},"ecpu_usage":[1000,1.0],"pcpu_usage":[2000,2.0],"gpu_usage":[1200,3.0],"all_power":1.0,"ane_power":2.0,"cpu_power":3.0,"gpu_power":4.0,"gpu_ram_power":5.0,"ram_power":6.0,"sys_power":7.0}
"""

    assert _available_memory_from_macmon_output(output) == 725


def test_available_memory_from_macmon_output_rejects_missing_sample() -> None:
    assert _available_memory_from_macmon_output("") is None
