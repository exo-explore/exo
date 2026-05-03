import pytest

from exo.utils.profilers.ane_profiler import AneProfile, ane_available


@pytest.mark.skipif(
    not ane_available(),
    reason="ANE profile requires AppleNeuralEngine.framework",
)
async def test_ane_profile_returns_plausible_numbers():
    profile = await AneProfile.measure()
    assert profile is not None
    assert profile.engine == "ane"
    assert profile.tflops_fp16 > 0
    assert profile.tflops_fp16 < 1000
    assert profile.memory_bandwidth_gbps > 0
    assert profile.memory_bandwidth_gbps < 100_000


def test_ane_profile_serializes_with_class_tag():
    profile = AneProfile(
        engine="ane",
        tflops_fp16=12.3,
        memory_bandwidth_gbps=80.0,
    )
    dumped = profile.model_dump()
    assert "AneProfile" in dumped
    assert dumped["AneProfile"]["tflops_fp16"] == 12.3
