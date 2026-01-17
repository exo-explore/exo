import asyncio
import os
import platform
from dataclasses import dataclass
from typing import Any, Callable, Coroutine

import anyio
from loguru import logger

from exo.shared.types.memory import Memory
from exo.shared.types.profiling import (
    MemoryPerformanceProfile,
    NetworkInterfaceInfo,
    SystemPerformanceProfile,
)

from .macmon import (
    MacMonError,
    Metrics,
)
from .macmon import (
    get_metrics_async as macmon_get_metrics_async,
)
from .system_info import (
    get_friendly_name,
    get_model_and_chip,
    get_network_interfaces,
)


@dataclass(frozen=True)
class IdentityMetrics:
    model_id: str
    chip_id: str
    friendly_name: str


async def get_metrics_async() -> Metrics | None:
    """Return detailed Metrics on macOS or a minimal fallback elsewhere."""

    if platform.system().lower() == "darwin":
        return await macmon_get_metrics_async()


def get_memory_profile() -> MemoryPerformanceProfile:
    """Construct a MemoryPerformanceProfile using psutil"""
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
    """Continuously poll and emit memory-only metrics at a faster cadence.

    Parameters
    - callback: coroutine called with a fresh MemoryPerformanceProfile each tick
    - poll_interval_s: interval between polls
    """
    while True:
        try:
            mem = get_memory_profile()
            await callback(mem)
        except MacMonError as e:
            logger.opt(exception=e).error("Memory Monitor encountered error")
        finally:
            await anyio.sleep(poll_interval_s)


async def start_polling_identity_metrics(
    callback: Callable[[IdentityMetrics], Coroutine[Any, Any, None]],
    *,
    poll_interval_s: float = 30.0,
) -> None:
    """Continuously poll and emit identity metrics at 30s intervals."""
    while True:
        try:
            model_id, chip_id = await get_model_and_chip()
            friendly_name = await get_friendly_name()
            await callback(
                IdentityMetrics(
                    model_id=model_id,
                    chip_id=chip_id,
                    friendly_name=friendly_name,
                )
            )
        except Exception as e:
            logger.opt(exception=e).error("Failed to emit identity metrics")
        finally:
            await anyio.sleep(poll_interval_s)


async def start_polling_system_metrics(
    callback: Callable[[SystemPerformanceProfile], Coroutine[Any, Any, None]],
    *,
    poll_interval_s: float = 1.0,
) -> None:
    """Continuously poll and emit system metrics (GPU, temp, power) at 1s intervals."""
    while True:
        try:
            metrics = await get_metrics_async()
            if metrics is None:
                return

            await callback(
                SystemPerformanceProfile(
                    gpu_usage=metrics.gpu_usage[1],
                    temp=metrics.temp.gpu_temp_avg,
                    sys_power=metrics.sys_power,
                    pcpu_usage=metrics.pcpu_usage[1],
                    ecpu_usage=metrics.ecpu_usage[1],
                    ane_power=metrics.ane_power,
                )
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[system_monitor] Operation timed out after 30s, skipping this cycle."
            )
        except MacMonError as e:
            logger.opt(exception=e).error("System Monitor encountered error")
            return
        finally:
            await anyio.sleep(poll_interval_s)


async def start_polling_network_metrics(
    callback: Callable[[list[NetworkInterfaceInfo]], Coroutine[Any, Any, None]],
    *,
    poll_interval_s: float = 30.0,
) -> None:
    """Continuously poll and emit network interface info at 30s intervals."""
    while True:
        try:
            network_interfaces = get_network_interfaces()
            await callback(network_interfaces)
        except Exception as e:
            logger.opt(exception=e).error("Network Monitor encountered error")
        finally:
            await anyio.sleep(poll_interval_s)
