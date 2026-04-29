from exo.shared.types.profiling import SystemPerformanceProfile
from exo.utils.pydantic_ext import TaggedModel

try:
    import pynvml as nvml
except ImportError:
    nvml = None

_CPU_POWER_IDLE = 20.0
_CPU_POWER_MAX = 100.0
_GPU_POWER_MAX = 120.0


class NvmlMetrics(TaggedModel):
    system_profile: SystemPerformanceProfile


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


def gather_nvidia_metrics() -> NvmlMetrics | None:
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
        for i in range(count):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            total_gpu_util += float(util.gpu)
            total_temp += float(
                nvml.nvmlDeviceGetTemperatureV(handle, nvml.NVML_TEMPERATURE_GPU)
            )
            total_gpu_power += float(nvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0

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
    finally:
        if is_init:
            nvml.nvmlShutdown()
