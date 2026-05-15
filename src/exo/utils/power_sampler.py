import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import final

import anyio

from exo.api.types import NodePowerStats, PowerUsage
from exo.shared.types.common import NodeId
from exo.shared.types.profiling import SystemPerformanceProfile


@final
class PowerSampler:
    def __init__(
        self,
        get_node_system: Callable[[], Mapping[NodeId, SystemPerformanceProfile]],
        interval: float = 1.0,
    ):
        self._get_node_system = get_node_system
        self._interval = interval
        self._samples: defaultdict[
            NodeId, list[tuple[float, SystemPerformanceProfile]]
        ] = defaultdict(list)
        self._start_time: float | None = None
        self._stopped = False

    def _take_sample(self, t_rel: float | None = None) -> None:
        assert self._start_time is not None
        ts = t_rel if t_rel is not None else time.perf_counter() - self._start_time
        for node_id, profile in self._get_node_system().items():
            self._samples[node_id].append((ts, profile))

    async def run(self) -> None:
        self._start_time = time.perf_counter()
        self._take_sample(t_rel=0.0)
        while not self._stopped:
            await anyio.sleep(self._interval)
            self._take_sample()

    def result(self) -> PowerUsage:
        self._stopped = True
        assert self._start_time is not None, "result() called before run()"
        elapsed = time.perf_counter() - self._start_time
        self._take_sample(t_rel=elapsed)

        node_stats: list[NodePowerStats] = []
        total_energy_j = 0.0
        for node_id, ts_profiles in self._samples.items():
            n = len(ts_profiles)
            if n == 0:
                continue
            node_energy_j = trapezoidal_energy(ts_profiles, elapsed)
            avg_power_w = node_energy_j / elapsed if elapsed > 0 else 0.0
            total_energy_j += node_energy_j
            node_stats.append(
                NodePowerStats(
                    node_id=node_id,
                    samples=n,
                    avg_sys_power=avg_power_w,
                )
            )

        total_avg_sys_w = total_energy_j / elapsed if elapsed > 0 else 0.0
        return PowerUsage(
            elapsed_seconds=elapsed,
            nodes=node_stats,
            total_avg_sys_power_watts=total_avg_sys_w,
            total_energy_joules=total_energy_j,
        )


def trapezoidal_energy(
    ts_profiles: list[tuple[float, SystemPerformanceProfile]],
    elapsed: float,
) -> float:
    """Integrate sys_power(t) over the sample window using the trapezoidal rule.
    First sample is anchored at t=0 and last at t=elapsed (set by `run` /
    `result`), so the integral spans the full request interval. Falls back to
    power * elapsed when only one sample exists (constant-power assumption)."""
    if len(ts_profiles) == 1:
        return ts_profiles[0][1].sys_power * elapsed
    energy_j = 0.0
    for i in range(1, len(ts_profiles)):
        t_prev, p_prev = ts_profiles[i - 1]
        t_cur, p_cur = ts_profiles[i]
        dt = t_cur - t_prev
        if dt <= 0:
            continue
        energy_j += (p_prev.sys_power + p_cur.sys_power) / 2.0 * dt
    return energy_j
