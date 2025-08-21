import asyncio
import platform
import shutil
import subprocess
from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, ValidationError


class MacMonError(Exception):
    """Exception raised for errors in the MacMon functions."""


def _get_binary_path() -> str:
    """
    Get the path to the macmon binary.

    Raises:
        MacMonError: If the binary doesn't exist or can't be made executable.
    """
    # Check for macOS with ARM chip
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system != "darwin" or not (
        "arm" in machine or "m1" in machine or "m2" in machine
    ):
        raise MacMonError("MacMon only supports macOS with Apple Silicon (ARM) chips")

    path = shutil.which("macmon")

    if path is None:
        raise MacMonError("MacMon not found in PATH")

    return path


# ---------------------------------------------------------------------------
# Pydantic metric structures
# ---------------------------------------------------------------------------


class MemoryMetrics(BaseModel):
    """Memory-related metrics returned by macmon."""

    ram_total: Optional[int] = None
    ram_usage: Optional[int] = None
    swap_total: Optional[int] = None
    swap_usage: Optional[int] = None

    model_config = ConfigDict(extra="ignore")


class TempMetrics(BaseModel):
    """Temperature-related metrics returned by macmon."""

    cpu_temp_avg: Optional[float] = None
    gpu_temp_avg: Optional[float] = None

    model_config = ConfigDict(extra="ignore")


class Metrics(BaseModel):
    """Complete set of metrics returned by *macmon* binary.

    All fields are optional to allow for partial output from the binary.
    Unknown fields are ignored for forward-compatibility.
    """

    all_power: Optional[float] = None
    ane_power: Optional[float] = None
    cpu_power: Optional[float] = None
    ecpu_usage: Optional[Tuple[int, float]] = None
    gpu_power: Optional[float] = None
    gpu_ram_power: Optional[float] = None
    gpu_usage: Optional[Tuple[int, float]] = None
    memory: Optional[MemoryMetrics] = None
    pcpu_usage: Optional[Tuple[int, float]] = None
    ram_power: Optional[float] = None
    sys_power: Optional[float] = None
    temp: Optional[TempMetrics] = None
    timestamp: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


# ---------------------------------------------------------------------------
# Synchronous helper
# ---------------------------------------------------------------------------


def get_metrics() -> Metrics:
    """
    Run the binary and return the metrics as a Python dictionary.

    Returns:
        A mapping containing system metrics.

    Raises:
        MacMonError: If there's an error running the binary.
    """
    path = _get_binary_path()

    try:
        # Run the binary with the argument -s 1 and capture its output
        result = subprocess.run(
            [path, "pipe", "-s", "1"], capture_output=True, text=True, check=True
        )

        return Metrics.model_validate_json(result.stdout)

    except subprocess.CalledProcessError as e:
        raise MacMonError(f"Error running binary: {e.stderr}") from e  # type: ignore
    except ValidationError as e:
        raise MacMonError(f"Error parsing JSON output: {e}") from e


async def get_metrics_async() -> Metrics:
    """
    Asynchronously run the binary and return the metrics as a Python dictionary.

    Args:
        binary_path: Optional path to the binary. If not provided, will use the bundled binary.

    Returns:
        A mapping containing system metrics.

    Raises:
        MacMonError: If there's an error running the binary.
    """
    path = _get_binary_path()

    try:
        proc = await asyncio.create_subprocess_exec(
            path,
            "pipe",
            "-s",
            "1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise MacMonError(f"Error running binary: {stderr.decode().strip()}")

        return Metrics.model_validate_json(stdout.decode().strip())

    except ValidationError as e:
        raise MacMonError(f"Error parsing JSON output: {e}") from e
