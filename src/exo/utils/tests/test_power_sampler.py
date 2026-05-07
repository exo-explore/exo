from collections.abc import Mapping

import anyio
import pytest

from exo.api.types import PowerUsage
from exo.shared.types.common import NodeId
from exo.shared.types.profiling import SystemPerformanceProfile
from exo.utils.power_sampler import PowerSampler


def _make_profile(sys_power: float) -> SystemPerformanceProfile:
    return SystemPerformanceProfile(sys_power=sys_power)


NODE_A = NodeId("node-a")
NODE_B = NodeId("node-b")


@pytest.fixture
def single_node_sampler() -> PowerSampler:
    state: dict[NodeId, SystemPerformanceProfile] = {
        NODE_A: _make_profile(10.0),
    }
    return PowerSampler(get_node_system=lambda: state)


@pytest.fixture
def multi_node_state() -> dict[NodeId, SystemPerformanceProfile]:
    return {
        NODE_A: _make_profile(10.0),
        NODE_B: _make_profile(20.0),
    }


async def test_single_sample(single_node_sampler: PowerSampler) -> None:
    """A sampler that runs briefly should capture at least the initial sample."""
    async with anyio.create_task_group() as tg:
        tg.start_soon(single_node_sampler.run)
        await anyio.sleep(0.05)
        tg.cancel_scope.cancel()

    result = single_node_sampler.result()
    assert len(result.nodes) == 1
    assert result.nodes[0].node_id == NODE_A
    assert result.nodes[0].avg_sys_power == 10.0
    assert result.nodes[0].samples >= 1
    assert result.elapsed_seconds > 0


async def test_multi_node_averaging(
    multi_node_state: dict[NodeId, SystemPerformanceProfile],
) -> None:
    """Power from multiple nodes should be summed for total cluster power."""
    sampler = PowerSampler(get_node_system=lambda: multi_node_state)
    async with anyio.create_task_group() as tg:
        tg.start_soon(sampler.run)
        await anyio.sleep(0.05)
        tg.cancel_scope.cancel()

    result = sampler.result()
    assert len(result.nodes) == 2
    assert result.total_avg_sys_power_watts == 30.0


async def test_energy_calculation(single_node_sampler: PowerSampler) -> None:
    """Energy (joules) should be avg_power * elapsed_seconds."""
    async with anyio.create_task_group() as tg:
        tg.start_soon(single_node_sampler.run)
        await anyio.sleep(0.3)
        tg.cancel_scope.cancel()

    result = single_node_sampler.result()
    expected_energy = result.total_avg_sys_power_watts * result.elapsed_seconds
    assert result.total_energy_joules == expected_energy


async def test_changing_power_is_averaged() -> None:
    """When power changes mid-sampling, the result should be the average."""
    state: dict[NodeId, SystemPerformanceProfile] = {
        NODE_A: _make_profile(10.0),
    }
    sampler = PowerSampler(get_node_system=lambda: state, interval=0.05)

    async with anyio.create_task_group() as tg:
        tg.start_soon(sampler.run)
        await anyio.sleep(0.15)
        state[NODE_A] = _make_profile(20.0)
        await anyio.sleep(0.15)
        tg.cancel_scope.cancel()

    result = sampler.result()
    avg = result.nodes[0].avg_sys_power
    # Should be between 10 and 20, not exactly either
    assert 10.0 < avg < 20.0


async def test_empty_state() -> None:
    """A sampler with no nodes should return an empty result."""
    empty: Mapping[NodeId, SystemPerformanceProfile] = {}
    sampler = PowerSampler(get_node_system=lambda: empty)

    async with anyio.create_task_group() as tg:
        tg.start_soon(sampler.run)
        await anyio.sleep(0.05)
        tg.cancel_scope.cancel()

    result = sampler.result()
    assert len(result.nodes) == 0
    assert result.total_avg_sys_power_watts == 0.0
    assert result.total_energy_joules == 0.0


def test_trapezoidal_unit_dt_weighting() -> None:
    """Pure unit test on the integration helper. Crafted samples where the
    arithmetic mean is wildly wrong vs the time-weighted result."""
    from exo.utils.power_sampler import trapezoidal_energy

    # 5 s window. Power = 10 W for the first 4.9 s, then 100 W for the last 0.1 s.
    # Three samples: t=0 W=10, t=4.9 W=10, t=5.0 W=100.
    samples = [
        (0.0, _make_profile(10.0)),
        (4.9, _make_profile(10.0)),
        (5.0, _make_profile(100.0)),
    ]
    energy = trapezoidal_energy(samples, elapsed=5.0)
    # (10+10)/2 * 4.9 + (10+100)/2 * 0.1 = 49 + 5.5 = 54.5 J
    assert abs(energy - 54.5) < 1e-9
    avg = energy / 5.0  # 10.9 W
    # Arithmetic mean of the three samples would be (10+10+100)/3 ≈ 40 W.
    # Trapezoidal correctly weights each segment by its dt.
    assert abs(avg - 10.9) < 1e-9


def test_trapezoidal_unit_single_sample() -> None:
    """One sample: no window to integrate over, so fall back to constant power
    over the elapsed duration."""
    from exo.utils.power_sampler import trapezoidal_energy

    samples = [(0.0, _make_profile(42.0))]
    assert trapezoidal_energy(samples, elapsed=3.0) == 42.0 * 3.0


async def test_result_stops_sampling() -> None:
    """Calling result() should stop the sampler's run loop."""
    state: dict[NodeId, SystemPerformanceProfile] = {
        NODE_A: _make_profile(10.0),
    }
    sampler = PowerSampler(get_node_system=lambda: state, interval=0.02)

    result: PowerUsage | None = None
    async with anyio.create_task_group() as tg:
        tg.start_soon(sampler.run)
        await anyio.sleep(0.1)
        result = sampler.result()
        # run() should exit on its own since _stopped is True
        await anyio.sleep(0.1)
        tg.cancel_scope.cancel()

    assert result is not None
    assert result.nodes[0].samples >= 2
