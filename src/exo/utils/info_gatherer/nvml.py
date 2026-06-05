import contextlib
from typing import Protocol, cast

from exo.shared.types.profiling import SystemPerformanceProfile
from exo.utils.pydantic_ext import TaggedModel

try:
    import pynvml as nvml
except ImportError:
    nvml = None


class _NvmlMemoryInfo(Protocol):
    total: int
    free: int


_CPU_POWER_IDLE = 20.0
_CPU_POWER_MAX = 100.0

_DEVICE_GPU_MAX_WATTS: list[tuple[str, float]] = [
    ("GB10", 100.0),
]


def _device_gpu_max_watts(name: str) -> float:
    for substring, gpu_max_watts in _DEVICE_GPU_MAX_WATTS:
        if substring in name:
            return gpu_max_watts
    return 0.0


_UNIFIED_MEMORY_DEVICES: tuple[str, ...] = ("GB10",)


class NvmlMetrics(TaggedModel):
    system_profile: SystemPerformanceProfile
    provides_sys_power: bool


def has_nvml() -> bool:
    if nvml is None:
        return False
    try:
        nvml.nvmlInit()
        count = nvml.nvmlDeviceGetCount()
        nvml.nvmlShutdown()
        return count > 0
    except Exception:
        return False


def gather_gpu_memory() -> tuple[int, int] | None:
    """Total and free GPU VRAM in bytes, summed across all NVIDIA devices.

    Returns None when NVML is unavailable, no device is present, or the device
    reports no usable VRAM — notably unified-memory devices (GB10/DGX Spark),
    where NVML/nvidia-smi report no discrete memory pool (`memory.total` shows
    [N/A]). Callers fall back to psutil host RAM in that case, which on a
    unified device is the same pool and is the correct number anyway.

    On discrete GPUs (e.g. B300) this is the meaningful capacity for model
    load — host RAM under-reports/misleads there.
    """
    if nvml is None:
        return None
    is_init = False
    try:
        nvml.nvmlInit()
        is_init = True
        count = nvml.nvmlDeviceGetCount()
        if count == 0:
            return None
        total = 0
        free = 0
        for i in range(count):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            name = cast(str | bytes, nvml.nvmlDeviceGetName(handle))  # pyright: ignore[reportUnknownMemberType]
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            if any(substring in name for substring in _UNIFIED_MEMORY_DEVICES):
                # Unified-memory device — host RAM (psutil) is the right number.
                return None
            mem = cast(_NvmlMemoryInfo, nvml.nvmlDeviceGetMemoryInfo(handle))  # pyright: ignore[reportUnknownMemberType]
            total += int(mem.total)
            free += int(mem.free)
        if total <= 0:
            return None
        return total, free
    except Exception:
        return None
    finally:
        if is_init:
            nvml.nvmlShutdown()


def gather_nvidia_metrics(*, provides_sys_power: bool) -> NvmlMetrics | None:
    if nvml is None:
        return None

    is_init = False
    try:
        nvml.nvmlInit()
        is_init = True
        count = nvml.nvmlDeviceGetCount()
        if count == 0:
            return None

        total_gpu_util = 0.0
        total_temp = 0.0
        total_gpu_power = 0.0
        total_power_limit = 0.0
        total_gpu_max = 0.0
        for i in range(count):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            total_gpu_util += float(util.gpu)
            total_temp += float(
                nvml.nvmlDeviceGetTemperatureV(handle, nvml.NVML_TEMPERATURE_GPU)
            )
            total_gpu_power += float(nvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
            with contextlib.suppress(Exception):
                total_power_limit += (
                    float(nvml.nvmlDeviceGetPowerManagementLimit(handle)) / 1000.0
                )
            with contextlib.suppress(Exception):
                name = cast(str | bytes, nvml.nvmlDeviceGetName(handle))  # pyright: ignore[reportUnknownMemberType]
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="replace")
                total_gpu_max += _device_gpu_max_watts(name)

        # Denominator for the GPU load fraction: prefer NVML's reported power
        # limit; fall back to the known device GPU-max when NVML reports none
        # (e.g. GB10, which exposes no power-management limit).
        power_denominator = (
            total_power_limit if total_power_limit > 0 else total_gpu_max
        )
        gpu_load_fraction = (
            min(total_gpu_power / power_denominator, 1.0)
            if power_denominator > 0
            else 0.0
        )
        estimated_cpu_power = (
            _CPU_POWER_IDLE + (_CPU_POWER_MAX - _CPU_POWER_IDLE) * gpu_load_fraction
        )

        return NvmlMetrics(
            system_profile=SystemPerformanceProfile(
                gpu_usage=total_gpu_util / count / 100.0,
                temp=total_temp / count,
                sys_power=total_gpu_power + estimated_cpu_power,
            ),
            provides_sys_power=provides_sys_power,
        )
    except Exception:
        return None
    finally:
        if is_init:
            nvml.nvmlShutdown()
