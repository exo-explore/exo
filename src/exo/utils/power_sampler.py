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
        self._samples: defaultdict[NodeId, list[SystemPerformanceProfile]] = (
            defaultdict(list)
        )
        self._start_time: float | None = None
        self._stopped = False

    def _take_sample(self) -> None:
        for node_id, profile in self._get_node_system().items():
            self._samples[node_id].append(profile)

    async def run(self) -> None:
        self._start_time = time.perf_counter()
        self._take_sample()
        while not self._stopped:
            await anyio.sleep(self._interval)
            self._take_sample()

    def result(self) -> PowerUsage:
        self._stopped = True
        assert self._start_time is not None, "result() called before run()"
        self._take_sample()
        elapsed = time.perf_counter() - self._start_time

        node_stats: list[NodePowerStats] = []
        for node_id, profiles in self._samples.items():
            n = len(profiles)
            if n == 0:
                continue
            node_stats.append(
                NodePowerStats(
                    node_id=node_id,
                    samples=n,
                    avg_sys_power=sum(p.sys_power for p in profiles) / n,
                )
            )

        total_avg_sys = sum(ns.avg_sys_power for ns in node_stats)
        return PowerUsage(
            elapsed_seconds=elapsed,
            nodes=node_stats,
            total_avg_sys_power_watts=total_avg_sys,
            total_energy_joules=total_avg_sys * elapsed,
        )
