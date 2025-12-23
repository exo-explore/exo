import asyncio
import os
import platform
from typing import Any, Callable, Coroutine

import anyio
import psutil
from loguru import logger

from exo.shared.types.memory import Memory
from exo.shared.types.profiling import (
    MemoryPerformanceProfile,
    MemoryPressureLevel,
    NodePerformanceProfile,
    PSIMetrics,
    SystemPerformanceProfile,
)

from .macmon import (
    MacMonError,
    Metrics,
)
from .macmon import (
    get_metrics_async as macmon_get_metrics_async,
)
from .memory_pressure import get_memory_pressure_sync
from .system_info import (
    get_friendly_name,
    get_model_and_chip,
    get_network_interfaces,
)


async def get_metrics_async() -> Metrics | None:
    """Return detailed Metrics on macOS or a minimal fallback elsewhere."""

    if platform.system().lower() == "darwin":
        return await macmon_get_metrics_async()


def get_memory_profile(*, include_pressure: bool = True) -> MemoryPerformanceProfile:
    """Construct a MemoryPerformanceProfile using psutil and pressure metrics.

    Args:
        include_pressure: If True (default), includes memory pressure detection.
                         Set to False for faster polling without pressure checks.
    """
    override_memory_env = os.getenv("OVERRIDE_MEMORY_MB")
    override_memory: int | None = (
        Memory.from_mb(int(override_memory_env)).in_bytes
        if override_memory_env
        else None
    )

    vm = psutil.virtual_memory()
    sm = psutil.swap_memory()

    # Get memory pressure metrics
    pressure_level = MemoryPressureLevel.NORMAL
    pressure_pct = 100.0
    psi: PSIMetrics | None = None

    if include_pressure:
        try:
            pressure_level, pressure_pct, psi = get_memory_pressure_sync()
        except Exception as e:
            logger.debug(f"Failed to get memory pressure: {e}")

    return MemoryPerformanceProfile.from_bytes(
        ram_total=vm.total,
        ram_available=vm.available if override_memory is None else override_memory,
        swap_total=sm.total,
        swap_available=sm.free,
        pressure_level=pressure_level,
        pressure_pct=pressure_pct,
        psi=psi,
    )


async def start_polling_memory_metrics(
    callback: Callable[[MemoryPerformanceProfile], Coroutine[Any, Any, None]],
    *,
    poll_interval_s: float = 0.5,
    pressure_check_interval: int = 10,
) -> None:
    """Continuously poll and emit memory-only metrics at a faster cadence.

    Parameters
    - callback: coroutine called with a fresh MemoryPerformanceProfile each tick
    - poll_interval_s: interval between polls for basic memory stats
    - pressure_check_interval: check pressure every N polls (reduces subprocess calls)
    """
    poll_count = 0
    while True:
        try:
            # Only check pressure every N polls to reduce subprocess overhead
            include_pressure = (poll_count % pressure_check_interval) == 0
            mem = get_memory_profile(include_pressure=include_pressure)
            await callback(mem)
            poll_count += 1
        except MacMonError as e:
            logger.opt(exception=e).error("Memory Monitor encountered error")
        finally:
            await anyio.sleep(poll_interval_s)


async def start_polling_node_metrics(
    callback: Callable[[NodePerformanceProfile], Coroutine[Any, Any, None]],
):
    poll_interval_s = 1.0
    while True:
        try:
            metrics = await get_metrics_async()
            if metrics is None:
                return

            network_interfaces = get_network_interfaces()
            # these awaits could be joined but realistically they should be cached
            model_id, chip_id = await get_model_and_chip()
            friendly_name = await get_friendly_name()

            # do the memory profile last to get a fresh reading to not conflict with the other memory profiling loop
            memory_profile = get_memory_profile()

            await callback(
                NodePerformanceProfile(
                    model_id=model_id,
                    chip_id=chip_id,
                    friendly_name=friendly_name,
                    network_interfaces=network_interfaces,
                    memory=memory_profile,
                    system=SystemPerformanceProfile(
                        gpu_usage=metrics.gpu_usage[1],
                        temp=metrics.temp.gpu_temp_avg,
                        sys_power=metrics.sys_power,
                        pcpu_usage=metrics.pcpu_usage[1],
                        ecpu_usage=metrics.ecpu_usage[1],
                        ane_power=metrics.ane_power,
                    ),
                )
            )

        except asyncio.TimeoutError:
            logger.warning(
                "[resource_monitor] Operation timed out after 30s, skipping this cycle."
            )
        except MacMonError as e:
            logger.opt(exception=e).error("Resource Monitor encountered error")
            return
        finally:
            await anyio.sleep(poll_interval_s)
