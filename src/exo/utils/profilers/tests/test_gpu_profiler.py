import mlx.core as mx
import pytest

from exo.utils.profilers.gpu_profiler import GpuProfile


@pytest.mark.skipif(
    not mx.metal.is_available(),
    reason="GPU profile requires Metal — skip on Linux/CPU",
)
async def test_gpu_profile_returns_plausible_numbers():
    """End-to-end: actually run the matmul + GEMV on the local GPU.

    We only assert that the numbers are positive and within a sane range.
    Hard-coding "expect X TFLOPS on chip Y" would just create flaky tests.
    """
    profile = await GpuProfile.measure()
    assert profile is not None
    assert profile.engine == "mlx"
    assert profile.tflops_fp16 > 0
    # 1000 TFLOPS would mean we're miscomputing; nothing on the market touches it.
    assert profile.tflops_fp16 < 1000
    assert profile.memory_bandwidth_gbps > 0
    assert profile.memory_bandwidth_gbps < 100_000


def test_gpu_profile_serializes_with_class_tag():
    """Wire format includes the class name as the discriminator (TaggedModel)."""
    profile = GpuProfile(
        engine="mlx",
        tflops_fp16=12.3,
        memory_bandwidth_gbps=400.0,
    )
    dumped = profile.model_dump()
    assert "GpuProfile" in dumped
    assert dumped["GpuProfile"]["tflops_fp16"] == 12.3
