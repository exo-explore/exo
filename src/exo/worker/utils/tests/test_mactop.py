"""Tests for mactop error handling.

These tests verify that Mactop errors are handled gracefully without
crashing the application or spamming logs.
"""

import platform
from subprocess import CalledProcessError, CompletedProcess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exo.worker.utils.mactop import MactopError, get_metrics_async


def _mock_version_check() -> MagicMock:
    """Create a mock for subprocess.run that returns a valid version."""
    mock = MagicMock()
    mock.return_value = CompletedProcess(
        args=["mactop", "-v"], returncode=0, stdout="mactop version: v2.0.5", stderr=""
    )
    return mock


@pytest.mark.skipif(
    platform.system().lower() != "darwin" or "arm" not in platform.machine().lower(),
    reason="Mactop only supports macOS with Apple Silicon",
)
class TestMactopErrorHandling:
    """Test Mactop error handling."""

    async def test_called_process_error_wrapped_as_mactop_error(self) -> None:
        """CalledProcessError should be wrapped as MactopError."""
        mock_error = CalledProcessError(
            returncode=1,
            cmd=["mactop", "--headless", "--count", "1"],
            stderr=b"some error message",
        )

        with (
            patch(
                "exo.worker.utils.mactop.shutil.which", return_value="/usr/bin/mactop"
            ),
            patch("exo.worker.utils.mactop.subprocess.run", _mock_version_check()),
            patch(
                "exo.worker.utils.mactop.run_process", new_callable=AsyncMock
            ) as mock_run,
        ):
            mock_run.side_effect = mock_error

            with pytest.raises(MactopError) as exc_info:
                await get_metrics_async()

            assert "Mactop failed with return code 1" in str(exc_info.value)
            assert "some error message" in str(exc_info.value)

    async def test_called_process_error_with_no_stderr(self) -> None:
        """CalledProcessError with no stderr should be handled gracefully."""
        mock_error = CalledProcessError(
            returncode=1,
            cmd=["mactop", "--headless", "--count", "1"],
            stderr=None,
        )

        with (
            patch(
                "exo.worker.utils.mactop.shutil.which", return_value="/usr/bin/mactop"
            ),
            patch("exo.worker.utils.mactop.subprocess.run", _mock_version_check()),
            patch(
                "exo.worker.utils.mactop.run_process", new_callable=AsyncMock
            ) as mock_run,
        ):
            mock_run.side_effect = mock_error

            with pytest.raises(MactopError) as exc_info:
                await get_metrics_async()

            assert "Mactop failed with return code 1" in str(exc_info.value)
            assert "no stderr" in str(exc_info.value)

    async def test_mactop_not_found_raises_mactop_error(self) -> None:
        """When mactop is not found in PATH, MactopError should be raised."""
        with patch("exo.worker.utils.mactop.shutil.which", return_value=None):
            with pytest.raises(MactopError) as exc_info:
                await get_metrics_async()

            assert "Mactop not found in PATH" in str(exc_info.value)
