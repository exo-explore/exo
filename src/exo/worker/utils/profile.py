import asyncio
import os
import platform
from typing import Any, Callable, Coroutine

import anyio
from anyio import to_thread
from loguru import logger

from exo.shared.types.memory import Memory
from exo.shared.types.profiling import (
    MemoryPerformanceProfile,
    NodePerformanceProfile,
    SystemPerformanceProfile,
)

from .macmon import (
    MacMonError,
    Metrics,
    get_metrics_async as macmon_get_metrics_async,
)
from .system_info import (
    get_friendly_name,
    get_model_and_chip,
    get_network_interfaces,
    profile_memory_bandwidth,
)


async def get_memory_bandwidth() -> int | None:
    """Profile memory bandwidth with retries. Returns None if all attempts fail."""
    for attempt in range(1, 6):
        try:
            return await to_thread.run_sync(profile_memory_bandwidth)
        except Exception as e:
            if attempt < 5:
                logger.warning(f"Memory bandwidth profiling attempt {attempt} failed: {e}")
                await anyio.sleep(1)
            else:
                logger.error(f"Memory bandwidth profiling failed after 5 attempts: {e}")
    return None


async def get_metrics_async() -> Metrics | None:
    if platform.system().lower() == "darwin":
        return await macmon_get_metrics_async()
    return None


def get_memory_profile() -> MemoryPerformanceProfile:
    override_memory_env = os.getenv("OVERRIDE_MEMORY_MB")
    override_memory: int | None = (
        Memory.from_mb(int(override_memory_env)).in_bytes
        if override_memory_env
        else None
    )
    return MemoryPerformanceProfile.from_psutil(override_memory=override_memory)


async def start_polling_memory_metrics(
    callback: Callable[[MemoryPerformanceProfile], Coroutine[Any, Any, None]],
    *,
    poll_interval_s: float = 0.5,
) -> None:
    while True:
        try:
            await callback(get_memory_profile())
        except MacMonError as e:
            logger.opt(exception=e).error("Memory Monitor encountered error")
        finally:
            await anyio.sleep(poll_interval_s)


async def start_polling_node_metrics(
    callback: Callable[[NodePerformanceProfile], Coroutine[Any, Any, None]],
) -> None:
    poll_interval_s = 1.0
    memory_bandwidth = await get_memory_bandwidth()
    if memory_bandwidth:
        logger.info(f"Memory bandwidth: {memory_bandwidth / 1e9:.1f} GB/s")

    while True:
        try:
            metrics = await get_metrics_async()
            if metrics is None:
                return

            model_id, chip_id = await get_model_and_chip()

            await callback(
                NodePerformanceProfile(
                    model_id=model_id,
                    chip_id=chip_id,
                    friendly_name=await get_friendly_name(),
                    network_interfaces=get_network_interfaces(),
                    memory=get_memory_profile(),
                    memory_bandwidth=memory_bandwidth,
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
            logger.warning("[resource_monitor] Operation timed out after 30s")
        except MacMonError as e:
            logger.opt(exception=e).error("Resource Monitor encountered error")
            return
        finally:
            await anyio.sleep(poll_interval_s)
