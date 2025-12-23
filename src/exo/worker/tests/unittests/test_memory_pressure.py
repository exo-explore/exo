"""Tests for memory pressure detection and integration."""

from exo.shared.types.profiling import (
    MemoryPerformanceProfile,
    MemoryPressureLevel,
    PSIMetrics,
)
from exo.worker.utils.memory_pressure import (
    _parse_psi_content,
    get_memory_pressure_sync,
)


class TestPSIMetrics:
    """Tests for PSI metrics parsing and pressure level conversion."""

    def test_parse_psi_content_normal(self):
        """Test parsing PSI content with normal pressure."""
        content = """some avg10=0.00 avg60=0.00 avg300=0.00 total=0
full avg10=0.00 avg60=0.00 avg300=0.00 total=0"""
        metrics = _parse_psi_content(content)

        assert metrics.some_avg10 == 0.0
        assert metrics.some_avg60 == 0.0
        assert metrics.full_avg10 == 0.0
        assert metrics.to_pressure_level() == MemoryPressureLevel.NORMAL

    def test_parse_psi_content_warning(self):
        """Test parsing PSI content with warning-level pressure."""
        content = """some avg10=15.50 avg60=8.20 avg300=3.10 total=123456789
full avg10=5.00 avg60=2.00 avg300=1.00 total=987654321"""
        metrics = _parse_psi_content(content)

        assert metrics.some_avg10 == 15.50
        assert metrics.some_avg60 == 8.20
        assert metrics.some_avg300 == 3.10
        assert metrics.full_avg10 == 5.00
        assert metrics.to_pressure_level() == MemoryPressureLevel.WARN

    def test_parse_psi_content_critical_from_some(self):
        """Test PSI critical level triggered by high 'some' value."""
        content = """some avg10=30.00 avg60=20.00 avg300=10.00 total=123456789
full avg10=5.00 avg60=2.00 avg300=1.00 total=987654321"""
        metrics = _parse_psi_content(content)

        assert metrics.some_avg10 == 30.00
        assert metrics.to_pressure_level() == MemoryPressureLevel.CRITICAL

    def test_parse_psi_content_critical_from_full(self):
        """Test PSI critical level triggered by high 'full' value."""
        content = """some avg10=15.00 avg60=10.00 avg300=5.00 total=123456789
full avg10=15.00 avg60=8.00 avg300=4.00 total=987654321"""
        metrics = _parse_psi_content(content)

        assert metrics.full_avg10 == 15.00
        assert metrics.to_pressure_level() == MemoryPressureLevel.CRITICAL

    def test_parse_psi_content_empty_lines(self):
        """Test parsing PSI content with empty lines."""
        content = """
some avg10=5.00 avg60=2.00 avg300=1.00 total=100

full avg10=1.00 avg60=0.50 avg300=0.25 total=50
"""
        metrics = _parse_psi_content(content)

        assert metrics.some_avg10 == 5.00
        assert metrics.full_avg10 == 1.00

    def test_psi_pressure_thresholds(self):
        """Test exact threshold boundaries for pressure levels."""
        # Just below warning threshold (some_avg10 <= 10)
        normal = PSIMetrics(some_avg10=10.0, full_avg10=0.0)
        assert normal.to_pressure_level() == MemoryPressureLevel.NORMAL

        # Just above warning threshold (some_avg10 > 10)
        warn = PSIMetrics(some_avg10=10.1, full_avg10=0.0)
        assert warn.to_pressure_level() == MemoryPressureLevel.WARN

        # Just below critical threshold (some_avg10 <= 25, full_avg10 <= 10)
        still_warn = PSIMetrics(some_avg10=25.0, full_avg10=10.0)
        assert still_warn.to_pressure_level() == MemoryPressureLevel.WARN

        # Just above critical threshold (some_avg10 > 25)
        critical_some = PSIMetrics(some_avg10=25.1, full_avg10=0.0)
        assert critical_some.to_pressure_level() == MemoryPressureLevel.CRITICAL

        # Just above critical threshold (full_avg10 > 10)
        critical_full = PSIMetrics(some_avg10=0.0, full_avg10=10.1)
        assert critical_full.to_pressure_level() == MemoryPressureLevel.CRITICAL


class TestMemoryPerformanceProfile:
    """Tests for MemoryPerformanceProfile with pressure awareness."""

    def test_effective_available_normal(self):
        """Test effective_available returns full memory under normal pressure."""
        profile = MemoryPerformanceProfile.from_bytes(
            ram_total=16 * 1024**3,
            ram_available=8 * 1024**3,
            swap_total=4 * 1024**3,
            swap_available=4 * 1024**3,
            pressure_level=MemoryPressureLevel.NORMAL,
        )

        assert profile.effective_available.in_bytes == 8 * 1024**3

    def test_effective_available_warn(self):
        """Test effective_available returns half memory under warning pressure."""
        profile = MemoryPerformanceProfile.from_bytes(
            ram_total=16 * 1024**3,
            ram_available=8 * 1024**3,
            swap_total=4 * 1024**3,
            swap_available=4 * 1024**3,
            pressure_level=MemoryPressureLevel.WARN,
        )

        assert profile.effective_available.in_bytes == 4 * 1024**3

    def test_effective_available_critical(self):
        """Test effective_available returns zero under critical pressure."""
        profile = MemoryPerformanceProfile.from_bytes(
            ram_total=16 * 1024**3,
            ram_available=8 * 1024**3,
            swap_total=4 * 1024**3,
            swap_available=4 * 1024**3,
            pressure_level=MemoryPressureLevel.CRITICAL,
        )

        assert profile.effective_available.in_bytes == 0

    def test_from_bytes_with_psi(self):
        """Test creating profile with PSI metrics."""
        psi = PSIMetrics(some_avg10=5.0, full_avg10=1.0)
        profile = MemoryPerformanceProfile.from_bytes(
            ram_total=16 * 1024**3,
            ram_available=8 * 1024**3,
            swap_total=4 * 1024**3,
            swap_available=4 * 1024**3,
            pressure_level=MemoryPressureLevel.NORMAL,
            pressure_pct=75.0,
            psi=psi,
        )

        assert profile.psi is not None
        assert profile.psi.some_avg10 == 5.0
        assert profile.pressure_pct == 75.0

    def test_from_psutil_defaults(self):
        """Test from_psutil creates profile with default pressure values."""
        profile = MemoryPerformanceProfile.from_psutil(override_memory=None)

        # Should have default pressure level
        assert profile.pressure_level == MemoryPressureLevel.NORMAL
        assert profile.pressure_pct == 0.0
        assert profile.psi is None


class TestGetMemoryPressureSync:
    """Tests for synchronous memory pressure detection."""

    def test_returns_tuple(self):
        """Test that get_memory_pressure_sync returns correct tuple structure."""
        level, free_pct, psi = get_memory_pressure_sync()

        assert isinstance(level, MemoryPressureLevel)
        assert isinstance(free_pct, float)
        assert 0.0 <= free_pct <= 100.0
        # psi can be None (macOS) or PSIMetrics (Linux)
        assert psi is None or isinstance(psi, PSIMetrics)

    def test_level_is_valid(self):
        """Test that returned level is a valid MemoryPressureLevel."""
        level, _, _ = get_memory_pressure_sync()

        assert level in [
            MemoryPressureLevel.NORMAL,
            MemoryPressureLevel.WARN,
            MemoryPressureLevel.CRITICAL,
        ]
