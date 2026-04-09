"""Tests for compute_wired_limit — the per-runner wired memory budget logic.

This is a pure function: how much device memory should each runner ask
Metal to wire so N sibling runners can coexist on the same machine
without each one nominally claiming the full device working set?
"""

from exo.shared.types.memory import Memory
from exo.worker.engines.mlx.utils_mlx import compute_wired_limit

# Plausible Mac Studio M4 Ultra working-set ceiling.
DEVICE_96_GB = Memory.from_bytes(96 * 1024 * 1024 * 1024)


def test_single_runner_gets_full_device_budget():
    """When max_runners_per_node=1 (the default), behavior matches the
    pre-fix code: each runner can wire the full device budget."""
    model = Memory.from_bytes(22 * 1024 * 1024 * 1024)  # 22 GB
    result = compute_wired_limit(model, DEVICE_96_GB, max_runners_per_node=1)
    assert result == DEVICE_96_GB


def test_three_sibling_runners_get_third_each():
    """With three siblings on a 96 GB device, each gets ~32 GB budget."""
    model = Memory.from_bytes(22 * 1024 * 1024 * 1024)
    result = compute_wired_limit(model, DEVICE_96_GB, max_runners_per_node=3)
    expected = DEVICE_96_GB / 3
    assert result == expected
    # Sanity: per-runner budget is bigger than the model itself.
    assert result > model * 1.1


def test_floor_at_model_size_when_too_many_siblings():
    """If the divided budget would be smaller than model_size * 1.1, the
    floor kicks in so the model can always be wired."""
    model = Memory.from_bytes(40 * 1024 * 1024 * 1024)  # 40 GB
    # 96 / 4 = 24 GB, which is less than 40 * 1.1 = 44 GB → floor.
    result = compute_wired_limit(model, DEVICE_96_GB, max_runners_per_node=4)
    assert result == model * 1.1


def test_floor_caps_at_device_max_when_model_huge():
    """If model_size * 1.1 exceeds the device working set entirely, we
    can only return the device max — we won't claim more than the device
    has."""
    model = Memory.from_bytes(100 * 1024 * 1024 * 1024)  # 100 GB > 96 GB
    result = compute_wired_limit(model, DEVICE_96_GB, max_runners_per_node=2)
    assert result == DEVICE_96_GB


def test_zero_or_negative_runners_treated_as_one():
    """Defensive: bad inputs should not crash; treat as single-runner."""
    model = Memory.from_bytes(10 * 1024 * 1024 * 1024)
    assert (
        compute_wired_limit(model, DEVICE_96_GB, max_runners_per_node=0) == DEVICE_96_GB
    )
    assert (
        compute_wired_limit(model, DEVICE_96_GB, max_runners_per_node=-5)
        == DEVICE_96_GB
    )


def test_two_siblings_half_each():
    """Sanity check the divide for the common 2-up case."""
    model = Memory.from_bytes(20 * 1024 * 1024 * 1024)
    result = compute_wired_limit(model, DEVICE_96_GB, max_runners_per_node=2)
    assert result == DEVICE_96_GB / 2
