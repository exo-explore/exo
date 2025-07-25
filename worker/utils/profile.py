import asyncio
import platform
from typing import Any, Callable, Coroutine

from shared.types.profiling import (
    MemoryPerformanceProfile,
    NodePerformanceProfile,
    SystemPerformanceProfile,
)
from worker.utils.macmon.macmon import (
    Metrics,
)
from worker.utils.macmon.macmon import (
    get_metrics_async as macmon_get_metrics_async,
)

# from exo.infra.event_log import EventLog
# from exo.app.config import ResourceMonitorConfig
# from exo.utils.mlx.mlx_utils import profile_flops_fp16


async def get_metrics_async() -> Metrics:
    """Return detailed Metrics on macOS or a minimal fallback elsewhere.

    The *Metrics* schema comes from ``utils.macmon.macmon``; on non-macOS systems we
    fill only the ``memory`` sub-structure so downstream code can still access
    ``metrics.memory.ram_total`` & ``ram_usage``.
    """

    if platform.system().lower() == "darwin":
        return await macmon_get_metrics_async()
    return Metrics()


async def start_polling_node_metrics(
    callback: Callable[[NodePerformanceProfile], Coroutine[Any, Any, None]],
):
    poll_interval_s = 1.0
    while True:
        try:
            # Gather metrics & system info with a timeout on each call
            metrics = await get_metrics_async()

            # Extract memory totals from metrics
            total_mem = (
                metrics.memory.ram_total
                if metrics.memory is not None and metrics.memory.ram_total is not None
                else 0
            )
            used_mem = (
                metrics.memory.ram_usage
                if metrics.memory is not None and metrics.memory.ram_usage is not None
                else 0
            )

            # Run heavy FLOPs profiling only if enough time has elapsed

            await callback(
                NodePerformanceProfile(
                    model_id=platform.machine(),
                    chip_id=platform.processor(),
                    memory=MemoryPerformanceProfile(
                        ram_total=total_mem,
                        ram_available=total_mem - used_mem,
                        swap_total=metrics.memory.swap_total
                        if metrics.memory is not None
                        and metrics.memory.swap_total is not None
                        else 0,
                        swap_available=metrics.memory.swap_total
                        - metrics.memory.swap_usage
                        if metrics.memory is not None
                        and metrics.memory.swap_usage is not None
                        and metrics.memory.swap_total is not None
                        else 0,
                    ),
                    network_interfaces=[],
                    system=SystemPerformanceProfile(
                        flops_fp16=0,
                    ),
                )
            )

        except asyncio.TimeoutError:
            # One of the operations took too long; skip this iteration but keep the loop alive.
            print(
                "[resource_monitor] Operation timed out after 30s, skipping this cycle."
            )
        except Exception as e:
            # Catch-all to ensure the monitor keeps running.
            print(f"[resource_monitor] Encountered error: {e}")
        finally:
            await asyncio.sleep(poll_interval_s)
