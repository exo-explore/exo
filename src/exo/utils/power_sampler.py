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
        self._prefill_done_at: float | None = None

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

    def mark_prefill_done(self) -> None:
        """Anchor the prefill→generation boundary on a fresh sample.
        Idempotent. Safe to call before `run()`; boundary then lands at t=0.
        """
        if self._prefill_done_at is not None:
            return
        if self._start_time is None:
            self._prefill_done_at = 0.0
            return
        t_rel = time.perf_counter() - self._start_time
        self._take_sample(t_rel=t_rel)
        self._prefill_done_at = t_rel

    def result(self) -> PowerUsage:
        self._stopped = True
        assert self._start_time is not None, "result() called before run()"
        elapsed = time.perf_counter() - self._start_time
        self._take_sample(t_rel=elapsed)

        # Clamp the split point to [0, elapsed] in case timing is weird (e.g.
        # mark called after result, or sampler ran for < the prefill window).
        split = self._prefill_done_at
        if split is not None:
            split = max(0.0, min(elapsed, split))

        node_stats: list[NodePowerStats] = []
        total_energy_j = 0.0
        total_prefill_energy_j = 0.0
        total_generation_energy_j = 0.0
        for node_id, ts_profiles in self._samples.items():
            n = len(ts_profiles)
            if n == 0:
                continue
            node_energy_j = trapezoidal_energy(ts_profiles, elapsed)
            avg_power_w = node_energy_j / elapsed if elapsed > 0 else 0.0
            total_energy_j += node_energy_j

            prefill_e: float | None = None
            generation_e: float | None = None
            prefill_avg: float | None = None
            generation_avg: float | None = None
            if split is not None:
                prefill_e = trapezoidal_energy_range(ts_profiles, 0.0, split)
                generation_e = trapezoidal_energy_range(ts_profiles, split, elapsed)
                total_prefill_energy_j += prefill_e
                total_generation_energy_j += generation_e
                prefill_dt = split
                generation_dt = elapsed - split
                prefill_avg = prefill_e / prefill_dt if prefill_dt > 0 else 0.0
                generation_avg = (
                    generation_e / generation_dt if generation_dt > 0 else 0.0
                )

            node_stats.append(
                NodePowerStats(
                    node_id=node_id,
                    samples=n,
                    avg_sys_power=avg_power_w,
                    prefill_avg_sys_power=prefill_avg,
                    generation_avg_sys_power=generation_avg,
                    prefill_energy_joules=prefill_e,
                    generation_energy_joules=generation_e,
                )
            )

        total_avg_sys_w = total_energy_j / elapsed if elapsed > 0 else 0.0

        prefill_seconds: float | None = None
        generation_seconds: float | None = None
        prefill_energy_joules: float | None = None
        generation_energy_joules: float | None = None
        prefill_avg_w: float | None = None
        generation_avg_w: float | None = None
        if split is not None:
            prefill_seconds = split
            generation_seconds = elapsed - split
            prefill_energy_joules = total_prefill_energy_j
            generation_energy_joules = total_generation_energy_j
            prefill_avg_w = (
                total_prefill_energy_j / prefill_seconds if prefill_seconds > 0 else 0.0
            )
            generation_avg_w = (
                total_generation_energy_j / generation_seconds
                if generation_seconds > 0
                else 0.0
            )

        return PowerUsage(
            elapsed_seconds=elapsed,
            nodes=node_stats,
            total_avg_sys_power_watts=total_avg_sys_w,
            total_energy_joules=total_energy_j,
            prefill_seconds=prefill_seconds,
            generation_seconds=generation_seconds,
            prefill_energy_joules=prefill_energy_joules,
            generation_energy_joules=generation_energy_joules,
            prefill_avg_sys_power_watts=prefill_avg_w,
            generation_avg_sys_power_watts=generation_avg_w,
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


def trapezoidal_energy_range(
    ts_profiles: list[tuple[float, SystemPerformanceProfile]],
    t_start: float,
    t_end: float,
) -> float:
    """Integrate sys_power(t) over [t_start, t_end] using the trapezoidal rule.

    Linearly interpolates power at the endpoints when they fall between
    existing samples, so callers can integrate over arbitrary sub-windows
    (e.g. the prefill segment) without losing accuracy. Returns 0 for an
    empty or zero-length window. Falls back to constant-power assumption
    when only one sample exists.
    """
    if t_end <= t_start:
        return 0.0
    if len(ts_profiles) == 0:
        return 0.0
    if len(ts_profiles) == 1:
        return ts_profiles[0][1].sys_power * (t_end - t_start)

    def power_at(t: float) -> float:
        if t <= ts_profiles[0][0]:
            return ts_profiles[0][1].sys_power
        if t >= ts_profiles[-1][0]:
            return ts_profiles[-1][1].sys_power
        for i in range(1, len(ts_profiles)):
            t_cur, p_cur = ts_profiles[i]
            if t_cur >= t:
                t_prev, p_prev = ts_profiles[i - 1]
                span = t_cur - t_prev
                if span <= 0:
                    return p_cur.sys_power
                frac = (t - t_prev) / span
                return p_prev.sys_power + frac * (p_cur.sys_power - p_prev.sys_power)
        return ts_profiles[-1][1].sys_power

    p_start = power_at(t_start)
    p_end = power_at(t_end)
    in_range: list[tuple[float, float]] = [
        (t, profile.sys_power) for t, profile in ts_profiles if t_start < t < t_end
    ]
    seq: list[tuple[float, float]] = [(t_start, p_start)] + in_range + [(t_end, p_end)]

    energy_j = 0.0
    for i in range(1, len(seq)):
        dt = seq[i][0] - seq[i - 1][0]
        if dt <= 0:
            continue
        energy_j += (seq[i - 1][1] + seq[i][1]) / 2.0 * dt
    return energy_j
