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
)
from .macmon import (
    get_metrics_async as macmon_get_metrics_async,
)
from .system_info import (
    get_friendly_name,
    get_model_and_chip,
    get_network_interfaces,
    profile_memory_bandwidth,
)

# Module-level cache for memory bandwidth (doesn't change at runtime)
_cached_bandwidth: int | None = None
_bandwidth_profiled: bool = False
_bandwidth_profiling_task: asyncio.Task[int | None] | None = None


async def profile_bandwidth_once() -> int | None:
    """Profile bandwidth once in a background thread and cache the result.

    This function is non-blocking - it runs the profiling in a thread pool.
    Subsequent calls return the cached result immediately.
    """
    global _cached_bandwidth, _bandwidth_profiled, _bandwidth_profiling_task

    # Already profiled, return cached value
    if _bandwidth_profiled:
        return _cached_bandwidth

    # Profiling already in progress, wait for it
    if _bandwidth_profiling_task is not None:
        return await _bandwidth_profiling_task

    # Start profiling in background thread
    async def _do_profile() -> int | None:
        global _cached_bandwidth, _bandwidth_profiled
        try:
            logger.info("Starting memory bandwidth profiling in background thread...")
            bandwidth = await to_thread.run_sync(profile_memory_bandwidth, cancellable=True)
            _cached_bandwidth = bandwidth
            _bandwidth_profiled = True
            if bandwidth:
                logger.info(f"Memory bandwidth profiled: {bandwidth / 1e9:.1f} GB/s")
            else:
                logger.warning("Memory bandwidth profiling returned None")
            return bandwidth
        except Exception as e:
            logger.opt(exception=e).error("Memory bandwidth profiling failed")
            _bandwidth_profiled = True  # Mark as done to avoid retrying
            return None

    _bandwidth_profiling_task = asyncio.create_task(_do_profile())
    return await _bandwidth_profiling_task


def get_memory_bandwidth_cached() -> int | None:
    """Return cached bandwidth or None if not yet profiled.

    This is a non-blocking synchronous function that returns immediately.
    Call profile_bandwidth_once() first to trigger profiling.
    """
    return _cached_bandwidth if _bandwidth_profiled else None


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


async def start_polling_node_metrics(
    callback: Callable[[NodePerformanceProfile], Coroutine[Any, Any, None]],
):
    poll_interval_s = 1.0
    bandwidth_profile_started = False

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

            # Start bandwidth profiling in background on first poll (non-blocking)
            if not bandwidth_profile_started:
                bandwidth_profile_started = True
                # Fire and forget - don't await, let it run in background
                asyncio.create_task(profile_bandwidth_once())

            # Use cached bandwidth (None until profiling completes)
            memory_bandwidth = get_memory_bandwidth_cached()

            await callback(
                NodePerformanceProfile(
                    model_id=model_id,
                    chip_id=chip_id,
                    friendly_name=friendly_name,
                    network_interfaces=network_interfaces,
                    memory=memory_profile,
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
            logger.warning(
                "[resource_monitor] Operation timed out after 30s, skipping this cycle."
            )
        except MacMonError as e:
            logger.opt(exception=e).error("Resource Monitor encountered error")
            return
        finally:
            await anyio.sleep(poll_interval_s)
