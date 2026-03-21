# pyright: reportMissingImports=false
from exo_core.models import TaggedModel

from exo.shared.types.profiling import SystemPerformanceProfile

try:
    from pynvml import (
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetPowerUsage,
        nvmlDeviceGetTemperature,
        nvmlDeviceGetUtilizationRates,
        nvmlInit,
        nvmlShutdown,
    )
except ImportError:
    nvmlDeviceGetCount = None  # noqa: N816
    nvmlDeviceGetHandleByIndex = None  # noqa: N816
    nvmlDeviceGetPowerUsage = None  # noqa: N816
    nvmlDeviceGetTemperature = None  # noqa: N816
    nvmlDeviceGetUtilizationRates = None  # noqa: N816
    nvmlInit = None  # noqa: N816
    nvmlShutdown = None  # noqa: N816

_CPU_POWER_IDLE = 20.0
_CPU_POWER_MAX = 100.0
_GPU_POWER_MAX = 120.0


class NvmlMetrics(TaggedModel):
    system_profile: SystemPerformanceProfile


def has_nvml() -> bool:
    if nvmlInit is None:
        return False
    try:
        nvmlInit()
        count = nvmlDeviceGetCount()  # type: ignore[reportOptionalCall]
        nvmlShutdown()  # type: ignore[reportOptionalCall]
        return count > 0
    except Exception:
        return False


def gather_nvidia_metrics() -> NvmlMetrics | None:
    if nvmlInit is None or nvmlDeviceGetCount is None or nvmlShutdown is None:
        return None
    if nvmlDeviceGetHandleByIndex is None or nvmlDeviceGetUtilizationRates is None:
        return None
    if nvmlDeviceGetTemperature is None or nvmlDeviceGetPowerUsage is None:
        return None

    try:
        nvmlInit()
        count = nvmlDeviceGetCount()
        if count == 0:
            nvmlShutdown()
            return None

        total_gpu_util = 0.0
        total_temp = 0.0
        total_gpu_power = 0.0
        for i in range(count):
            handle = nvmlDeviceGetHandleByIndex(i)
            util = nvmlDeviceGetUtilizationRates(handle)
            total_gpu_util += float(util.gpu)
            total_temp += float(nvmlDeviceGetTemperature(handle, 0))
            total_gpu_power += float(nvmlDeviceGetPowerUsage(handle)) / 1000.0

        nvmlShutdown()

        gpu_load_fraction = min(total_gpu_power / _GPU_POWER_MAX, 1.0)
        estimated_cpu_power = (
            _CPU_POWER_IDLE + (_CPU_POWER_MAX - _CPU_POWER_IDLE) * gpu_load_fraction
        )

        return NvmlMetrics(
            system_profile=SystemPerformanceProfile(
                gpu_usage=total_gpu_util / count / 100.0,
                temp=total_temp / count,
                sys_power=total_gpu_power + estimated_cpu_power,
            ),
        )
    except Exception:
        return None
