import shutil
from dataclasses import dataclass
from datetime import datetime

from anyio import run_process
from loguru import logger

from exo.shared.types.profiling import MemoryUsage, SystemPerformanceProfile
from exo.utils.info_gatherer.macmon import MacmonMetrics

# Conversion constant
MIB_TO_BYTES = 1024 * 1024


@dataclass
class LinuxGpuMetrics:
    """Clean dataclass for Linux GPU metrics from nvidia-smi."""

    gpu_utilization: float  # percentage 0-100
    gpu_power_watts: float
    gpu_temp_celsius: float
    vram_total_bytes: int
    vram_free_bytes: int
    timestamp: str


def _safe_parse_float(value: str, default: float = 0.0) -> float:
    """Safely parse a float from nvidia-smi output, handling [N/A] and other edge cases."""
    value = value.strip()
    if not value or value.startswith("[") or value.lower() in ("n/a", "not supported"):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _safe_parse_int_from_mib(value: str, default: int = 0) -> int:
    """Safely parse MiB value to bytes from nvidia-smi output."""
    parsed = _safe_parse_float(value, float(default) / MIB_TO_BYTES)
    return int(parsed * MIB_TO_BYTES)


async def get_linux_gpu_metrics() -> LinuxGpuMetrics:
    """Collects GPU metrics for Linux via nvidia-smi. Returns clean dataclass."""

    # Defaults
    gpu_util = 0.0
    gpu_power = 0.0
    gpu_temp = 0.0
    vram_total = 0
    vram_free = 0
    timestamp = str(int(datetime.now().timestamp() * 1000))

    if shutil.which("nvidia-smi"):
        try:
            # query: utilization.gpu [%], power.draw [W], temperature.gpu [C], memory.total [MiB], memory.free [MiB]
            result = await run_process(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,power.draw,temperature.gpu,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ]
            )
            output = result.stdout.decode().strip()
            if output:
                lines = output.split("\n")
                # Aggregate across all GPUs: sum VRAM, average utilization, max temp, sum power
                total_vram = 0
                total_free = 0
                total_power = 0.0
                max_temp = 0.0
                total_util = 0.0
                gpu_count = 0
                for line in lines:
                    parts = line.split(",")
                    if len(parts) >= 5:
                        total_util += _safe_parse_float(parts[0])
                        total_power += _safe_parse_float(parts[1])
                        max_temp = max(max_temp, _safe_parse_float(parts[2]))
                        total_vram += _safe_parse_int_from_mib(parts[3])
                        total_free += _safe_parse_int_from_mib(parts[4])
                        gpu_count += 1
                if gpu_count > 0:
                    gpu_util = total_util / gpu_count
                    gpu_power = total_power
                    gpu_temp = max_temp
                    vram_total = total_vram
                    vram_free = total_free
        except Exception as e:
            logger.warning(f"Failed to query nvidia-smi: {e}")

    return LinuxGpuMetrics(
        gpu_utilization=gpu_util,
        gpu_power_watts=gpu_power,
        gpu_temp_celsius=gpu_temp,
        vram_total_bytes=vram_total,
        vram_free_bytes=vram_free,
        timestamp=timestamp,
    )


async def get_linux_metrics_async() -> MacmonMetrics:
    """Collects metrics for Linux (specifically NVIDIA GPUs via nvidia-smi).

    Returns a MacmonMetrics object for compatibility with the GatheredInfo interface.
    MacmonMetrics wraps SystemPerformanceProfile + MemoryUsage which are generic types;
    Mac-specific fields (pcpu_usage, ecpu_usage) are set to 0 on Linux.
    Note: Uses VRAM as memory metrics for Linux GPU systems.
    """
    gpu_metrics = await get_linux_gpu_metrics()

    # Convert GPU utilization from percentage (0-100) to decimal (0-1)
    gpu_util_decimal = (
        gpu_metrics.gpu_utilization / 100.0 if gpu_metrics.gpu_utilization > 0 else 0.0
    )

    return MacmonMetrics(
        system_profile=SystemPerformanceProfile(
            gpu_usage=gpu_util_decimal,
            temp=gpu_metrics.gpu_temp_celsius,
            sys_power=gpu_metrics.gpu_power_watts,
            pcpu_usage=0.0,
            ecpu_usage=0.0,
        ),
        memory=MemoryUsage.from_bytes(
            ram_total=gpu_metrics.vram_total_bytes,
            ram_available=gpu_metrics.vram_free_bytes,
            swap_total=0,
            swap_available=0,
        ),
    )
