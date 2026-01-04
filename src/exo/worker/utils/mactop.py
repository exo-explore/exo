import json
import platform
import re
import shutil
import subprocess
from subprocess import CalledProcessError
from typing import cast

from anyio import run_process
from pydantic import BaseModel, ConfigDict, ValidationError

MINIMUM_VERSION = (2, 0, 5)


class MactopError(Exception):
    """Exception raised for errors in the Mactop functions."""


def _parse_version(version_str: str) -> tuple[int, ...]:
    """Parse version string like 'v2.0.5' into tuple (2, 0, 5)."""
    match = re.search(r"v?(\d+(?:\.\d+)*)", version_str)
    if not match:
        return (0, 0, 0)
    return tuple(int(x) for x in match.group(1).split("."))


def _check_version(path: str) -> None:
    """Check mactop version meets minimum requirement."""
    try:
        result = subprocess.run([path, "-v"], capture_output=True, text=True, timeout=5)
        version = _parse_version(result.stdout.strip())
        if version < MINIMUM_VERSION:
            min_ver = ".".join(str(x) for x in MINIMUM_VERSION)
            raise MactopError(
                f"Mactop version {result.stdout.strip()} is too old. "
                f"Please upgrade to v{min_ver} or later: brew upgrade mactop"
            )
    except subprocess.TimeoutExpired:
        pass  # Skip version check if it times out
    except FileNotFoundError as e:
        raise MactopError("Mactop not found in PATH") from e


def _get_binary_path() -> str:
    """
    Get the path to the mactop binary.

    Raises:
        MactopError: If the binary doesn't exist or can't be made executable.
    """
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system != "darwin" or not (
        "arm" in machine or "m1" in machine or "m2" in machine
    ):
        raise MactopError("Mactop only supports macOS with Apple Silicon (ARM) chips")

    path = shutil.which("mactop")

    if path is None:
        raise MactopError("Mactop not found in PATH")

    _check_version(path)

    return path


class SocMetrics(BaseModel):
    """SOC metrics returned by mactop."""

    cpu_power: float
    gpu_power: float
    ane_power: float
    dram_power: float
    system_power: float
    total_power: float
    soc_temp: float
    cpu_temp: float
    gpu_temp: float

    model_config = ConfigDict(extra="ignore")


class MemoryInfo(BaseModel):
    """Memory information returned by mactop."""

    total: int
    used: int
    available: int
    swap_total: int
    swap_used: int

    model_config = ConfigDict(extra="ignore")


class SystemInfo(BaseModel):
    """System information returned by mactop."""

    name: str
    core_count: int
    e_core_count: int
    p_core_count: int
    gpu_core_count: int

    model_config = ConfigDict(extra="ignore")


class NetworkStats(BaseModel):
    """Network stats for a Thunderbolt bus."""

    interface_name: str
    bytes_in: int
    bytes_out: int
    bytes_in_per_sec: float
    bytes_out_per_sec: float

    model_config = ConfigDict(extra="ignore")


class ThunderboltDevice(BaseModel):
    """A device connected to a Thunderbolt bus."""

    name: str
    vendor: str
    mode: str

    model_config = ConfigDict(extra="ignore")


class ThunderboltBus(BaseModel):
    """Information about a Thunderbolt bus."""

    name: str
    status: str
    speed: str
    domain_uuid: str
    receptacle_id: str
    devices: list[ThunderboltDevice] = []
    network_stats: NetworkStats | None = None

    model_config = ConfigDict(extra="ignore")


class ThunderboltInfo(BaseModel):
    """Thunderbolt information returned by mactop."""

    buses: list[ThunderboltBus]

    model_config = ConfigDict(extra="ignore")


class RdmaStatus(BaseModel):
    """RDMA status returned by mactop."""

    available: bool
    status: str

    model_config = ConfigDict(extra="ignore")


class Metrics(BaseModel):
    """Complete set of metrics returned by mactop.

    Unknown fields are ignored for forward-compatibility.
    """

    timestamp: str
    soc_metrics: SocMetrics
    memory: MemoryInfo
    cpu_usage: float
    gpu_usage: float
    core_usages: list[float]
    system_info: SystemInfo
    thermal_state: str
    thunderbolt_info: ThunderboltInfo
    tb_net_total_bytes_in_per_sec: float
    tb_net_total_bytes_out_per_sec: float
    rdma_status: RdmaStatus
    ecpu_usage: tuple[int, float]
    pcpu_usage: tuple[int, float]

    model_config = ConfigDict(extra="ignore")

    @property
    def total_power(self) -> float:
        """Total power from SOC metrics."""
        return self.soc_metrics.total_power

    @property
    def sys_power(self) -> float:
        """System power from SOC metrics."""
        return self.soc_metrics.system_power

    @property
    def ane_power(self) -> float:
        """ANE power from SOC metrics."""
        return self.soc_metrics.ane_power

    @property
    def cpu_temp(self) -> float:
        """CPU temperature from SOC metrics."""
        return self.soc_metrics.cpu_temp

    @property
    def gpu_temp(self) -> float:
        """GPU temperature from SOC metrics."""
        return self.soc_metrics.gpu_temp

    @property
    def ecpu_usage_percent(self) -> float:
        """E-Core usage percentage (second element of tuple)."""
        return self.ecpu_usage[1]

    @property
    def pcpu_usage_percent(self) -> float:
        """P-Core usage percentage (second element of tuple)."""
        return self.pcpu_usage[1]


async def get_metrics_async() -> Metrics:
    """
    Asynchronously run the binary and return the metrics as a Python dictionary.

    Returns:
        A Metrics object containing system metrics.

    Raises:
        MactopError: If there's an error running the binary.
    """
    path = _get_binary_path()

    try:
        result = await run_process(
            [path, "--headless", "--count", "1", "--interval", "100"]
        )

        output = result.stdout.decode().strip()
        raw_data = cast(dict[str, object] | list[dict[str, object]], json.loads(output))
        data: dict[str, object]
        if isinstance(raw_data, list) and len(raw_data) > 0:
            data = raw_data[0]
        else:
            data = cast(dict[str, object], raw_data)

        return Metrics.model_validate(data)

    except ValidationError as e:
        raise MactopError(f"Error parsing JSON output: {e}") from e
    except json.JSONDecodeError as e:
        raise MactopError(f"Error parsing JSON output: {e}") from e
    except CalledProcessError as e:
        stderr_msg = "no stderr"
        stderr_output = cast(bytes | str | None, e.stderr)
        if stderr_output is not None:
            stderr_msg = (
                stderr_output.decode()
                if isinstance(stderr_output, bytes)
                else str(stderr_output)
            )
        raise MactopError(
            f"Mactop failed with return code {e.returncode}: {stderr_msg}"
        ) from e
