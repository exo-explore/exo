"""Tests for macmon error handling.

These tests verify that MacMon errors are handled gracefully without
crashing the application or spamming logs.
"""

import platform
from subprocess import CalledProcessError
from unittest.mock import AsyncMock, patch

import pytest

from exo.worker.utils.macmon import MacMonError, get_metrics_async


@pytest.mark.skipif(
    platform.system().lower() != "darwin" or "arm" not in platform.machine().lower(),
    reason="MacMon only supports macOS with Apple Silicon",
)
class TestMacMonErrorHandling:
    """Test MacMon error handling."""

    async def test_called_process_error_wrapped_as_macmon_error(self) -> None:
        """CalledProcessError should be wrapped as MacMonError."""
        mock_error = CalledProcessError(
            returncode=1,
            cmd=["macmon", "pipe", "-s", "1"],
            stderr=b"some error message",
        )

        with (
            patch(
                "exo.worker.utils.macmon.shutil.which", return_value="/usr/bin/macmon"
            ),
            patch(
                "exo.worker.utils.macmon.run_process", new_callable=AsyncMock
            ) as mock_run,
        ):
            mock_run.side_effect = mock_error

            with pytest.raises(MacMonError) as exc_info:
                await get_metrics_async()

            assert "MacMon failed with return code 1" in str(exc_info.value)
            assert "some error message" in str(exc_info.value)

    async def test_called_process_error_with_no_stderr(self) -> None:
        """CalledProcessError with no stderr should be handled gracefully."""
        mock_error = CalledProcessError(
            returncode=1,
            cmd=["macmon", "pipe", "-s", "1"],
            stderr=None,
        )

        with (
            patch(
                "exo.worker.utils.macmon.shutil.which", return_value="/usr/bin/macmon"
            ),
            patch(
                "exo.worker.utils.macmon.run_process", new_callable=AsyncMock
            ) as mock_run,
        ):
            mock_run.side_effect = mock_error

            with pytest.raises(MacMonError) as exc_info:
                await get_metrics_async()

            assert "MacMon failed with return code 1" in str(exc_info.value)
            assert "no stderr" in str(exc_info.value)

    async def test_macmon_not_found_raises_macmon_error(self) -> None:
        """When macmon is not found in PATH, MacMonError should be raised."""
        with patch("exo.worker.utils.macmon.shutil.which", return_value=None):
            with pytest.raises(MacMonError) as exc_info:
                await get_metrics_async()

            assert "MacMon not found in PATH" in str(exc_info.value)
