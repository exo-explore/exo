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


# Data from: https://en.wikipedia.org/wiki/Apple_silicon
# Bandwidth in bytes/second
APPLE_SILICON_BANDWIDTH = {
    # M1 Series
    "Apple M1": 68_250_000_000,
    "Apple M1 Pro": 200_000_000_000,
    "Apple M1 Max": 400_000_000_000,
    "Apple M1 Ultra": 800_000_000_000,
    # M2 Series
    "Apple M2": 100_000_000_000,
    "Apple M2 Pro": 200_000_000_000,
    "Apple M2 Max": 400_000_000_000,
    "Apple M2 Ultra": 800_000_000_000,
    # M3 Series
    "Apple M3": 100_000_000_000,
    "Apple M3 Pro": 150_000_000_000,
    "Apple M3 Max": 400_000_000_000,
    # M4 Series
    "Apple M4": 120_000_000_000,
    "Apple M4 Pro": 273_000_000_000,
    "Apple M4 Max": 546_000_000_000,
}


def get_memory_bandwidth(chip_id: str) -> int | None:
    """
    Returns the theoretical memory bandwidth in bytes/second for a given chip ID.
    Currently only supports Apple Silicon.
    """
    return APPLE_SILICON_BANDWIDTH.get(chip_id)
