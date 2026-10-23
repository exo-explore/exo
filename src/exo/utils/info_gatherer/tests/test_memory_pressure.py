from unittest.mock import patch

from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryPressureLevel, MemoryUsage
from exo.utils.info_gatherer.macmon import MacmonMetrics
from exo.utils.info_gatherer.memory_pressure import (
    _get_darwin_vm_stats,
    probe_memory_pressure,
)


def test_memory_pressure_enum_and_model():
    mem = MemoryUsage.from_bytes(
        ram_total=64 * 1024 * 1024 * 1024,
        ram_available=40 * 1024 * 1024 * 1024,
        swap_total=0,
        swap_available=0,
        pressure_level=MemoryPressureLevel.NORMAL,
        pressure_pct=25.5,
    )
    assert mem.pressure_level == MemoryPressureLevel.NORMAL
    assert mem.pressure_pct == 25.5
    assert mem.ram_total.in_gb == 64.0

    # Serialization test
    dumped = mem.model_dump(by_alias=True)
    assert dumped["pressureLevel"] == "normal"
    assert dumped["pressurePct"] == 25.5


def test_darwin_vm_stat_parsing():
    sample_vm_stat = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                              40000.
Pages active:                           400000.
Pages inactive:                         300000.
Pages speculative:                        5000.
Pages throttled:                             0.
Pages wired down:                       200000.
Pages purgeable:                         25000.
"Translation faults":                442183842.
"""
    with (
        patch("subprocess.check_output", return_value=sample_vm_stat),
        patch("sys.platform", "darwin"),
        patch("shutil.which", return_value="/usr/bin/vm_stat"),
        patch("os.path.exists", return_value=True),
    ):
        wired, reclaimable = _get_darwin_vm_stats()
        assert wired == 200000 * 16384
        assert reclaimable == (25000 + 300000) * 16384


def test_probe_memory_pressure_darwin_normal():
    ram_total = 64 * 1024 * 1024 * 1024  # 64 GB
    psutil_available = 4 * 1024 * 1024 * 1024  # 4 GB (due to file caches)
    wired_bytes = 10 * 1024 * 1024 * 1024  # 10 GB wired

    with (
        patch("sys.platform", "darwin"),
        patch(
            "exo.utils.info_gatherer.memory_pressure._get_darwin_sysctl_pressure",
            return_value=(MemoryPressureLevel.NORMAL, 20.0),
        ),
        patch(
            "exo.utils.info_gatherer.memory_pressure._get_darwin_vm_stats",
            return_value=(wired_bytes, 30 * 1024 * 1024 * 1024),
        ),
    ):
        info = probe_memory_pressure(ram_total, psutil_available)
        assert info.level == MemoryPressureLevel.NORMAL
        assert info.pressure_pct == 20.0
        # True available should account for unified memory (much higher than 4 GB naive)
        assert info.effective_available_bytes is not None
        assert info.effective_available_bytes > psutil_available
        assert info.effective_available_bytes <= int(ram_total * 0.85)


def test_probe_memory_pressure_darwin_critical():
    ram_total = 64 * 1024 * 1024 * 1024
    psutil_available = 1 * 1024 * 1024 * 1024

    with (
        patch("sys.platform", "darwin"),
        patch(
            "exo.utils.info_gatherer.memory_pressure._get_darwin_sysctl_pressure",
            return_value=(MemoryPressureLevel.CRITICAL, 95.0),
        ),
        patch(
            "exo.utils.info_gatherer.memory_pressure._get_darwin_vm_stats",
            return_value=(40 * 1024 * 1024 * 1024, 0),
        ),
    ):
        info = probe_memory_pressure(ram_total, psutil_available)
        assert info.level == MemoryPressureLevel.CRITICAL
        # Critical pressure throttles effective availability to 0
        assert info.effective_available_bytes == 0


def test_macmon_metrics_with_pressure():
    raw_json = """{
        "timestamp": "2026-10-23T16:00:00Z",
        "temp": {"cpu_temp_avg": 45.0, "gpu_temp_avg": 48.0},
        "memory": {
            "ram_total": 68719476736,
            "ram_usage": 60129542144,
            "swap_total": 0,
            "swap_usage": 0
        },
        "ecpu_usage": [1000, 10.0],
        "pcpu_usage": [2000, 20.0],
        "gpu_usage": [1000, 15.0],
        "all_power": 12.5,
        "ane_power": 0.0,
        "cpu_power": 5.0,
        "gpu_power": 7.5,
        "gpu_ram_power": 2.0,
        "ram_power": 1.5,
        "sys_power": 16.0
    }"""
    metrics = MacmonMetrics.from_raw_json(raw_json)
    assert metrics.memory.ram_total == Memory.from_bytes(68719476736)
    assert metrics.memory.pressure_level in (
        MemoryPressureLevel.NORMAL,
        MemoryPressureLevel.WARNING,
        MemoryPressureLevel.CRITICAL,
    )
