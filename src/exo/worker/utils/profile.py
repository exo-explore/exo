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
    NodePerformanceProfile,
    SystemPerformanceProfile,
)
from exo.worker.utils.macmon.macmon import (
    Metrics,
)
from exo.worker.utils.macmon.macmon import (
    get_metrics_async as macmon_get_metrics_async,
)
from exo.worker.utils.system_info import (
    get_mac_friendly_name_async,
    get_mac_system_info_async,
    get_network_interface_info_async,
)


async def get_metrics_async() -> Metrics:
    """Return detailed Metrics on macOS or a minimal fallback elsewhere.

    The *Metrics* schema comes from ``utils.macmon.macmon``; on non-macOS systems we
    fill only the ``memory`` sub-structure so downstream code can still access
    ``metrics.memory.ram_total`` & ``ram_usage``.
    """

    if platform.system().lower() == "darwin":
        return await macmon_get_metrics_async()
    return Metrics()


async def get_memory_profile_async() -> MemoryPerformanceProfile:
    """Return MemoryPerformanceProfile using psutil (fast, cross-platform).

    Uses synchronous psutil calls in a worker thread to avoid blocking the event loop.
    """

    def _read_psutil() -> MemoryPerformanceProfile:
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()

        override_memory_env = os.getenv("OVERRIDE_MEMORY_MB")
        override_memory: int | None = (
            Memory.from_mb(int(override_memory_env)).in_bytes
            if override_memory_env
            else None
        )

        return MemoryPerformanceProfile.from_bytes(
            ram_total=int(vm.total),
            ram_available=int(override_memory)
            if override_memory
            else int(vm.available),
            swap_total=int(sm.total),
            swap_available=int(sm.free),
        )

    return await asyncio.to_thread(_read_psutil)


async def start_polling_memory_metrics(
    callback: Callable[[MemoryPerformanceProfile], Coroutine[Any, Any, None]],
    *,
    poll_interval_s: float = 0.5,
) -> None:
    """Continuously poll and emit memory-only metrics at a faster cadence.

    Parameters
    - callback: coroutine called with a fresh MemoryPerformanceProfile each tick
    - poll_interval_s: interval between polls
    """
    while True:
        try:
            mem = await get_memory_profile_async()
            await callback(mem)
        except Exception as e:
            logger.opt(exception=e).error("Memory Monitor encountered error")
        finally:
            await anyio.sleep(poll_interval_s)


async def start_polling_node_metrics(
    callback: Callable[[NodePerformanceProfile], Coroutine[Any, Any, None]],
):
    poll_interval_s = 1.0
    while True:
        try:
            # Gather metrics & system info with a timeout on each call
            metrics = await get_metrics_async()

            (
                system_info,
                network_interfaces,
                mac_friendly_name,
            ) = await asyncio.gather(
                get_mac_system_info_async(),
                get_network_interface_info_async(),
                get_mac_friendly_name_async(),
            )

            # do the memory profile last to get a fresh reading to not conflict with the other memory profiling loop
            memory_profile = await get_memory_profile_async()

            await callback(
                NodePerformanceProfile(
                    model_id=system_info.model_id,
                    chip_id=system_info.chip_id,
                    friendly_name=mac_friendly_name or "Unknown",
                    network_interfaces=network_interfaces,
                    memory=memory_profile,
                    system=SystemPerformanceProfile(
                        flops_fp16=0,
                        gpu_usage=metrics.gpu_usage[1]
                        if metrics.gpu_usage is not None
                        else 0,
                        temp=metrics.temp.gpu_temp_avg
                        if metrics.temp is not None
                        and metrics.temp.gpu_temp_avg is not None
                        else 0,
                        sys_power=metrics.sys_power
                        if metrics.sys_power is not None
                        else 0,
                        pcpu_usage=metrics.pcpu_usage[1]
                        if metrics.pcpu_usage is not None
                        else 0,
                        ecpu_usage=metrics.ecpu_usage[1]
                        if metrics.ecpu_usage is not None
                        else 0,
                        ane_power=metrics.ane_power
                        if metrics.ane_power is not None
                        else 0,
                    ),
                )
            )

        except asyncio.TimeoutError:
            # One of the operations took too long; skip this iteration but keep the loop alive.
            logger.warning(
                "[resource_monitor] Operation timed out after 30s, skipping this cycle."
            )
        except Exception as e:
            # Catch-all to ensure the monitor keeps running.
            logger.opt(exception=e).error("Resource Monitor encountered error")
        finally:
            await anyio.sleep(poll_interval_s)
