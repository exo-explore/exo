from collections.abc import Mapping

import anyio
import pytest
from exo_core.types.common import NodeId

from exo.api.types import PowerUsage
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
