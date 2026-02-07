"""Tests for Linux GPU metrics collection from nvidia-smi."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exo.utils.info_gatherer.linux_metrics import (
    MIB_TO_BYTES,
    _safe_parse_float,
    _safe_parse_int_from_mib,
    get_linux_gpu_metrics,
    get_linux_metrics_async,
)


class TestSafeParseFloat:
    """Tests for the _safe_parse_float helper function."""

    def test_valid_float(self):
        assert _safe_parse_float("42.5") == 42.5

    def test_valid_int_as_float(self):
        assert _safe_parse_float("100") == 100.0

    def test_with_whitespace(self):
        assert _safe_parse_float("  75.5  ") == 75.5

    def test_na_value(self):
        assert _safe_parse_float("[N/A]") == 0.0

    def test_not_supported(self):
        assert _safe_parse_float("[Not Supported]") == 0.0

    def test_lowercase_na(self):
        assert _safe_parse_float("n/a") == 0.0

    def test_empty_string(self):
        assert _safe_parse_float("") == 0.0

    def test_custom_default(self):
        assert _safe_parse_float("[N/A]", default=99.9) == 99.9

    def test_invalid_string(self):
        assert _safe_parse_float("invalid") == 0.0


class TestSafeParseIntFromMib:
    """Tests for the _safe_parse_int_from_mib helper function."""

    def test_valid_mib(self):
        assert _safe_parse_int_from_mib("1024") == 1024 * MIB_TO_BYTES

    def test_na_value(self):
        assert _safe_parse_int_from_mib("[N/A]") == 0

    def test_with_whitespace(self):
        assert _safe_parse_int_from_mib("  2048  ") == 2048 * MIB_TO_BYTES


class TestGetLinuxGpuMetrics:
    """Tests for get_linux_gpu_metrics function."""

    @pytest.mark.asyncio
    async def test_nvidia_smi_not_found(self):
        """Test graceful handling when nvidia-smi is not installed."""
        with patch(
            "exo.utils.info_gatherer.linux_metrics.shutil.which", return_value=None
        ):
            metrics = await get_linux_gpu_metrics()
            assert metrics.gpu_utilization == 0.0
            assert metrics.vram_total_bytes == 0
            assert metrics.vram_free_bytes == 0

    @pytest.mark.asyncio
    async def test_nvidia_smi_normal_output(self):
        """Test parsing normal nvidia-smi output."""
        mock_result = MagicMock()
        mock_result.stdout = b"75, 150.5, 65, 8192, 4096\n"

        with (
            patch(
                "exo.utils.info_gatherer.linux_metrics.shutil.which",
                return_value="/usr/bin/nvidia-smi",
            ),
            patch(
                "exo.utils.info_gatherer.linux_metrics.run_process",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
        ):
            metrics = await get_linux_gpu_metrics()
            assert metrics.gpu_utilization == 75.0
            assert metrics.gpu_power_watts == 150.5
            assert metrics.gpu_temp_celsius == 65.0
            assert metrics.vram_total_bytes == 8192 * MIB_TO_BYTES
            assert metrics.vram_free_bytes == 4096 * MIB_TO_BYTES

    @pytest.mark.asyncio
    async def test_nvidia_smi_with_na_values(self):
        """Test parsing nvidia-smi output with [N/A] values (e.g., datacenter GPUs)."""
        mock_result = MagicMock()
        mock_result.stdout = b"80, [N/A], 70, 16384, 8192\n"

        with (
            patch(
                "exo.utils.info_gatherer.linux_metrics.shutil.which",
                return_value="/usr/bin/nvidia-smi",
            ),
            patch(
                "exo.utils.info_gatherer.linux_metrics.run_process",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
        ):
            metrics = await get_linux_gpu_metrics()
            assert metrics.gpu_utilization == 80.0
            assert metrics.gpu_power_watts == 0.0  # N/A defaults to 0
            assert metrics.gpu_temp_celsius == 70.0
            assert metrics.vram_total_bytes == 16384 * MIB_TO_BYTES

    @pytest.mark.asyncio
    async def test_nvidia_smi_all_na(self):
        """Test parsing nvidia-smi output with all [N/A] values."""
        mock_result = MagicMock()
        mock_result.stdout = b"[N/A], [N/A], [N/A], [N/A], [N/A]\n"

        with (
            patch(
                "exo.utils.info_gatherer.linux_metrics.shutil.which",
                return_value="/usr/bin/nvidia-smi",
            ),
            patch(
                "exo.utils.info_gatherer.linux_metrics.run_process",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
        ):
            metrics = await get_linux_gpu_metrics()
            assert metrics.gpu_utilization == 0.0
            assert metrics.gpu_power_watts == 0.0
            assert metrics.vram_total_bytes == 0

    @pytest.mark.asyncio
    async def test_nvidia_smi_exception(self):
        """Test graceful handling when nvidia-smi throws an exception."""
        with (
            patch(
                "exo.utils.info_gatherer.linux_metrics.shutil.which",
                return_value="/usr/bin/nvidia-smi",
            ),
            patch(
                "exo.utils.info_gatherer.linux_metrics.run_process",
                new_callable=AsyncMock,
                side_effect=Exception("Driver error"),
            ),
        ):
            metrics = await get_linux_gpu_metrics()
            assert metrics.gpu_utilization == 0.0
            assert metrics.vram_total_bytes == 0

    @pytest.mark.asyncio
    async def test_nvidia_smi_empty_output(self):
        """Test handling of empty nvidia-smi output."""
        mock_result = MagicMock()
        mock_result.stdout = b""

        with (
            patch(
                "exo.utils.info_gatherer.linux_metrics.shutil.which",
                return_value="/usr/bin/nvidia-smi",
            ),
            patch(
                "exo.utils.info_gatherer.linux_metrics.run_process",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
        ):
            metrics = await get_linux_gpu_metrics()
            assert metrics.gpu_utilization == 0.0
            assert metrics.vram_total_bytes == 0

    @pytest.mark.asyncio
    async def test_nvidia_smi_multi_gpu_takes_first(self):
        """Test that multi-GPU output uses the first GPU."""
        mock_result = MagicMock()
        mock_result.stdout = b"50, 100, 60, 8192, 4096\n75, 200, 70, 16384, 8192\n"

        with (
            patch(
                "exo.utils.info_gatherer.linux_metrics.shutil.which",
                return_value="/usr/bin/nvidia-smi",
            ),
            patch(
                "exo.utils.info_gatherer.linux_metrics.run_process",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
        ):
            metrics = await get_linux_gpu_metrics()
            # Should use first GPU's values
            assert metrics.gpu_utilization == 50.0
            assert metrics.gpu_power_watts == 100.0


class TestGetLinuxMetricsAsync:
    """Tests for get_linux_metrics_async compatibility wrapper."""

    @pytest.mark.asyncio
    async def test_returns_macmon_metrics_object(self):
        """Test that it returns a valid MacmonMetrics object."""
        with patch(
            "exo.utils.info_gatherer.linux_metrics.shutil.which", return_value=None
        ):
            metrics = await get_linux_metrics_async()
            # Should return a MacmonMetrics object with system_profile and memory
            assert hasattr(metrics, "system_profile")
            assert hasattr(metrics, "memory")
            assert hasattr(metrics.system_profile, "gpu_usage")
            assert hasattr(metrics.system_profile, "temp")
