import socket
import sys
from subprocess import CalledProcessError

import psutil
from anyio import run_process

from exo.shared.types.profiling import NetworkInterfaceInfo


async def get_friendly_name() -> str:
    """
    Asynchronously gets the 'Computer Name' (friendly name) of a Mac.
    e.g., "John's MacBook Pro"
    Returns the name as a string, or None if an error occurs or not on macOS.
    """
    hostname = socket.gethostname()

    # TODO: better non mac support
    if sys.platform != "darwin":  # 'darwin' is the platform name for macOS
        return hostname

    try:
        process = await run_process(["scutil", "--get", "ComputerName"])
    except CalledProcessError:
        return hostname

    return process.stdout.decode("utf-8", errors="replace").strip() or hostname


def get_network_interfaces() -> list[NetworkInterfaceInfo]:
    """
    Retrieves detailed network interface information on macOS.
    Parses output from 'networksetup -listallhardwareports' and 'ifconfig'
    to determine interface names, IP addresses, and types (ethernet, wifi, vpn, other).
    Returns a list of NetworkInterfaceInfo objects.
    """
    interfaces_info: list[NetworkInterfaceInfo] = []

    for iface, services in psutil.net_if_addrs().items():
        for service in services:
            match service.family:
                case socket.AF_INET | socket.AF_INET6:
                    interfaces_info.append(
                        NetworkInterfaceInfo(name=iface, ip_address=service.address)
                    )
                case _:
                    pass

    return interfaces_info


async def get_model_and_chip() -> tuple[str, str]:
    """Get Mac system information using system_profiler."""
    model = "Unknown Model"
    chip = "Unknown Chip"

    # TODO: better non mac support
    if sys.platform != "darwin":
        return (model, chip)

    try:
        process = await run_process(
            [
                "system_profiler",
                "SPHardwareDataType",
            ]
        )
    except CalledProcessError:
        return (model, chip)

    # less interested in errors here because this value should be hard coded
    output = process.stdout.decode().strip()

    model_line = next(
        (line for line in output.split("\n") if "Model Name" in line), None
    )
    model = model_line.split(": ")[1] if model_line else "Unknown Model"

    chip_line = next((line for line in output.split("\n") if "Chip" in line), None)
    chip = chip_line.split(": ")[1] if chip_line else "Unknown Chip"

    return (model, chip)


def _profile_memory_bandwidth_numpy() -> int | None:
    """Profile memory bandwidth using 1GB array benchmark."""
    try:
        import numpy as np
        import time

        size = 1024 * 1024 * 1024 // 8
        num_runs = 3
        best_bandwidth = 0.0

        for _ in range(num_runs):
            src = np.random.random(size)
            start = time.perf_counter()
            dst = src.copy()
            end = time.perf_counter()
            _ = dst[0]
            bandwidth = (size * 8 * 2) / (end - start)
            best_bandwidth = max(best_bandwidth, bandwidth)
            del src, dst

        return int(best_bandwidth)
    except Exception:
        return None


def _profile_memory_bandwidth_simple() -> int | None:
    """Fallback memory bandwidth benchmark using 200MB array."""
    try:
        import numpy as np
        import time

        size = 200 * 1024 * 1024 // 8
        best_bandwidth = 0.0
        num_runs = 5

        for _ in range(num_runs):
            src = np.random.random(size)
            start = time.perf_counter()
            dst = src.copy()
            end = time.perf_counter()
            _ = dst[0]
            bandwidth = (size * 8 * 2) / (end - start)
            best_bandwidth = max(best_bandwidth, bandwidth)
            del src, dst

        return int(best_bandwidth)
    except Exception:
        return None


def profile_memory_bandwidth() -> int | None:
    """
    Profile device memory bandwidth using numpy benchmarks.

    Returns measured bandwidth which may be lower than theoretical peak.
    Relative ratios between devices remain accurate for placement decisions.
    """
    bandwidth = _profile_memory_bandwidth_numpy()
    if bandwidth and bandwidth > 0:
        return bandwidth

    bandwidth = _profile_memory_bandwidth_simple()
    if bandwidth and bandwidth > 0:
        return bandwidth

    return None


def get_memory_bandwidth(_chip_id: str) -> int | None:
    """
    Returns measured memory bandwidth in bytes/second.

    Uses runtime profiling via numpy benchmarks. Works on any platform.
    """
    return profile_memory_bandwidth()
