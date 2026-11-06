
from exo.shared.types.cpu_jit import (
    CPUCapabilityInfo,
)
from exo.worker.engines.cpu_jit import (
    CacheTiledGEMMEngine,
    CacheTilingPlanner,
    CacheTopologyProber,
    CoreAffinityManager,
    CPUIDInstructionProber,
    VectorizedJITDispatcher,
)


def test_cpuid_prober_hardware_and_override():
    """Verify hardware detection and simulated legacy CPU degradation."""
    info = CPUIDInstructionProber.probe()
    assert info.physical_cores >= 1
    assert info.active_level in (
        "AVX512",
        "AVX2",
        "AVX",
        "SSE4_2",
        "SSE4_1",
        "ARM_NEON",
        "SCALAR",
    )

    # Simulate legacy Core 2 Duo (only SSE4.1)
    legacy_info = CPUIDInstructionProber.probe(override_flags={"sse4_1", "sse2"})
    assert legacy_info.active_level == "SSE4_1"
    assert not legacy_info.has_avx2
    assert legacy_info.has_sse4_1

    # Simulate legacy Nehalem (SSE4.2)
    sse42_info = CPUIDInstructionProber.probe(override_flags={"sse4_2"})
    assert sse42_info.active_level == "SSE4_2"
    assert sse42_info.has_sse4_2

    # Simulate modern AVX2 machine
    avx2_info = CPUIDInstructionProber.probe(override_flags={"avx2", "fma"})
    assert avx2_info.active_level == "AVX2"
    assert avx2_info.has_avx2

    # Simulate AVX512 workstation
    avx512_info = CPUIDInstructionProber.probe(override_flags={"avx512f"})
    assert avx512_info.active_level == "AVX512"
    assert avx512_info.has_avx512

    # Simulate ARM Cortex / Apple Silicon
    arm_info = CPUIDInstructionProber.probe(override_flags={"neon"})
    assert arm_info.active_level == "ARM_NEON"
    assert arm_info.has_neon

    # Simulate ultra-legacy 386/Pentium without vector flags -> Graceful fallback
    scalar_info = CPUIDInstructionProber.probe(override_flags=set())
    assert scalar_info.active_level == "SCALAR"
    assert not scalar_info.has_avx2


def test_cache_topology_prober():
    """Verify cache topology geometry detection and custom overrides."""
    geom = CacheTopologyProber.get_cache_geometry()
    assert geom.validate()
    assert geom.l1d_cache_bytes >= 16 * 1024
    assert geom.l2_cache_bytes >= geom.l1d_cache_bytes
    assert geom.cache_line_bytes == 64

    custom = CacheTopologyProber.get_cache_geometry(
        l1d_override=16 * 1024,
        l2_override=128 * 1024,
        l3_override=2 * 1024 * 1024,
    )
    assert custom.l1d_cache_bytes == 16 * 1024
    assert custom.l2_cache_bytes == 128 * 1024
    assert custom.validate()


def test_cache_tiling_planner():
    """Verify cache tiling planner produces valid tile bounds and cache hit rate estimates."""
    plan_small = CacheTilingPlanner.plan(m=8, k=16, n=12)
    assert plan_small.tile_m <= 8
    assert plan_small.tile_k <= 16
    assert plan_small.tile_n <= 12
    assert plan_small.estimated_cache_hit_rate > 0.50

    # Large matrix tiling test
    plan_large = CacheTilingPlanner.plan(m=512, k=1024, n=512)
    assert plan_large.tile_m > 0
    assert plan_large.tile_k > 0
    assert plan_large.tile_n > 0
    assert plan_large.l1_working_set_bytes <= 32 * 1024  # fits L1 budget
    assert plan_large.l2_working_set_bytes <= 256 * 1024  # fits L2 budget


def test_vector_jit_dispatcher_kernel_outputs():
    """Verify numeric equivalence across scalar, SSE4 unrolled, and AVX2 unrolled kernels."""
    m, k, n = 4, 8, 12
    a = [float(i + 1) * 0.5 for i in range(m * k)]
    b = [float(j + 2) * 0.25 for j in range(k * n)]

    c_scalar, _ = VectorizedJITDispatcher._kernel_scalar_safe(a, b, m, k, n)
    c_sse4, _ = VectorizedJITDispatcher._kernel_sse4_unrolled(a, b, m, k, n)
    c_avx2, _ = VectorizedJITDispatcher._kernel_avx2_unrolled(a, b, m, k, n)

    assert len(c_scalar) == m * n
    assert len(c_sse4) == m * n
    assert len(c_avx2) == m * n

    for s, e, v in zip(c_scalar, c_sse4, c_avx2, strict=True):
        assert abs(s - e) < 1e-4, f"Scalar {s} != SSE4 {e}"
        assert abs(s - v) < 1e-4, f"Scalar {s} != AVX2 {v}"


def test_ternary_weights_unpacking():
    """Verify unpacking of packed 1.58-bit ternary {-1, 0, +1} weights."""
    # Packed byte: 0b10010000 -> codes: 00(0), 00(0), 01(+1), 10(-1)
    # in little-endian bit pairs:
    # shift 0: 00 -> 0.0
    # shift 2: 00 -> 0.0
    # shift 4: 01 -> +1.5 (scale 1.5)
    # shift 6: 10 -> -1.5
    packed = bytes([0b10010000])
    unpacked = VectorizedJITDispatcher.unpack_ternary_weights(
        packed, num_weights=4, scale=1.5
    )
    assert len(unpacked) == 4
    assert unpacked[0] == 0.0
    assert unpacked[1] == 0.0
    assert unpacked[2] == 1.5
    assert unpacked[3] == -1.5


def test_cache_tiled_gemm_engine_execution():
    """Verify end-to-end CacheTiledGEMMEngine computation correctness."""
    m, k, n = 8, 16, 12
    a = [float(i) for i in range(m * k)]
    b = [float(j) * 0.1 for j in range(k * n)]

    engine = CacheTiledGEMMEngine()
    c, report = engine.execute_gemm(a, b, m, k, n)

    assert len(c) == m * n
    assert report.m == m
    assert report.k == k
    assert report.n == n
    assert report.elapsed_ms >= 0.0
    assert report.gflops >= 0.0

    # Ground truth reference check for c[0, 0] = sum(a[0, p] * b[p, 0])
    expected_c00 = sum(a[0 * k + p] * b[p * n + 0] for p in range(k))
    assert abs(c[0] - expected_c00) < 1e-4


def test_legacy_cpu_fallback_guarantee():
    """Verify no Illegal Instruction crash on simulated ancient CPU (Scalar / SSE4 fallback)."""
    legacy_cap = CPUCapabilityInfo(
        architecture="x86_64",
        vendor="GenuineIntel",
        active_level="SCALAR",
        physical_cores=1,
        logical_cores=1,
    )
    legacy_engine = CacheTiledGEMMEngine(capability=legacy_cap)

    m, k, n = 16, 32, 16
    a = [1.0] * (m * k)
    b = [2.0] * (k * n)

    c, report = legacy_engine.execute_gemm(a, b, m, k, n)
    assert report.is_fallback
    assert report.instruction_level == "SCALAR"
    # Each element of C should be k * 1.0 * 2.0 = 64.0
    assert all(abs(val - 64.0) < 1e-4 for val in c)


def test_core_affinity_manager_safe():
    """Verify CoreAffinityManager does not throw unhandled exceptions."""
    # Attempt pinning to core 0 (or empty list)
    res_empty = CoreAffinityManager.pin_to_cores([])
    assert not res_empty

    # Safe call with core 0 (returns bool, never raises exception)
    res = CoreAffinityManager.pin_to_cores([0])
    assert isinstance(res, bool)
