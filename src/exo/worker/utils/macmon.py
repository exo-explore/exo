import platform
import shutil
from subprocess import CalledProcessError

from anyio import run_process
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


class TempMetrics(BaseModel):
    """Temperature-related metrics returned by macmon."""

    cpu_temp_avg: float
    gpu_temp_avg: float

    model_config = ConfigDict(extra="ignore")


class Metrics(BaseModel):
    """Complete set of metrics returned by macmon.

    Unknown fields are ignored for forward-compatibility.
    """

    all_power: float
    ane_power: float
    cpu_power: float
    ecpu_usage: tuple[int, float]
    gpu_power: float
    gpu_ram_power: float
    gpu_usage: tuple[int, float]
    pcpu_usage: tuple[int, float]
    ram_power: float
    sys_power: float
    temp: TempMetrics
    timestamp: str

    model_config = ConfigDict(extra="ignore")


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

    result = None
    try:
        # TODO: Keep Macmon running in the background?
        result = await run_process([path, "pipe", "-s", "1"])

        return Metrics.model_validate_json(result.stdout.decode().strip())

    except ValidationError as e:
        raise MacMonError(f"Error parsing JSON output: {e}") from e
    except CalledProcessError as e:
        if result:
            raise MacMonError(
                f"MacMon failed with return code {result.returncode}"
            ) from e
        raise e
