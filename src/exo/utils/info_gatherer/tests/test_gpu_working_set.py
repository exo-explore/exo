import json

import psutil
import pytest

from exo.shared.types.memory import Memory
from exo.utils.info_gatherer.macmon import MacmonMetrics
from exo.utils.info_gatherer.system_info import get_gpu_working_set

GB = 1024**3


def _macmon_line(ram_total: int, ram_usage: int) -> str:
    return json.dumps(
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "temp": {"cpu_temp_avg": 40.0, "gpu_temp_avg": 35.0},
            "memory": {
                "ram_total": ram_total,
                "ram_usage": ram_usage,
                "swap_total": 0,
                "swap_usage": 0,
            },
            "ecpu_usage": [1000, 0.1],
            "pcpu_usage": [3000, 0.2],
            "gpu_usage": [1000, 0.0],
            "all_power": 10.0,
            "ane_power": 0.0,
            "cpu_power": 5.0,
            "gpu_power": 1.0,
            "gpu_ram_power": 0.0,
            "ram_power": 0.5,
            "sys_power": 12.0,
        }
    )


def test_memory_reports_carry_the_gpus_working_set() -> None:
    working_set = Memory.from_bytes(12_713_115_648)

    metrics = MacmonMetrics.from_raw_json(_macmon_line(16 * GB, 4 * GB), working_set)

    assert metrics.memory.ram_available == Memory.from_bytes(12 * GB)
    assert metrics.memory.gpu_working_set == working_set


def test_the_gpus_working_set_is_most_of_the_ram() -> None:
    working_set = get_gpu_working_set()
    if working_set is None:
        pytest.skip("Needs a Mac with Metal")

    ram = psutil.virtual_memory().total
    assert 0.5 * ram < working_set.in_bytes < ram
