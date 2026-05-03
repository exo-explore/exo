import pytest

from exo.utils.profilers.ane_profiler import (
    AnePrecisionProfile,
    AneProfile,
    ane_available,
)


@pytest.mark.skipif(
    not ane_available(),
    reason="ANE profile requires AppleNeuralEngine.framework",
)
async def test_ane_profile_returns_plausible_numbers():
    profile = await AneProfile.measure()
    assert profile is not None
    assert profile.engine == "ane"
    precision_by_bits = {p.precision_bits: p for p in profile.precision_profiles}
    assert set(precision_by_bits) == {32, 16, 8, 4}
    for bits in (16, 8, 4):
        precision = precision_by_bits[bits]
        assert precision.supported
        assert precision.compute_tops is not None
        assert precision.compute_tops > 0
        assert precision.compute_tops < 1000
        assert precision.memory_bandwidth_gbps is not None
        assert precision.memory_bandwidth_gbps > 0
        assert precision.memory_bandwidth_gbps < 100_000

    fp32 = precision_by_bits[32]
    if fp32.supported:
        assert fp32.compute_tops is not None
        assert fp32.compute_tops > 0
    else:
        assert fp32.error


def test_ane_profile_serializes_with_class_tag():
    profile = AneProfile(
        engine="ane",
        precision_profiles=(
            AnePrecisionProfile(
                precision_bits=16,
                weight_bits=16,
                activation_bits=16,
                supported=True,
                compute_tops=12.3,
                memory_bandwidth_gbps=80.0,
            ),
        ),
    )
    dumped = profile.model_dump()
    assert "AneProfile" in dumped
    assert dumped["AneProfile"]["precision_profiles"][0]["precision_bits"] == 16
    assert dumped["AneProfile"]["precision_profiles"][0]["compute_tops"] == 12.3
