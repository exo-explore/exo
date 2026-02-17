import math
import platform
import socket
import sys
import time
from subprocess import CalledProcessError

import psutil
from anyio import run_process

from exo.shared.types.profiling import InterfaceType, NetworkInterfaceInfo


def get_os_version() -> str:
    """Return the OS version string for this node.

    On macOS this is the macOS version (e.g. ``"15.3"``).
    On other platforms it falls back to the platform name (e.g. ``"Linux"``).
    """
    if sys.platform == "darwin":
        version = platform.mac_ver()[0]
        return version if version else "Unknown"
    return platform.system() or "Unknown"


async def get_os_build_version() -> str:
    """Return the macOS build version string (e.g. ``"24D5055b"``).

    On non-macOS platforms, returns ``"Unknown"``.
    """
    if sys.platform != "darwin":
        return "Unknown"

    try:
        process = await run_process(["sw_vers", "-buildVersion"])
    except CalledProcessError:
        return "Unknown"

    return process.stdout.decode("utf-8", errors="replace").strip() or "Unknown"


async def get_friendly_name() -> str:
    """
    Asynchronously gets the 'Computer Name' (friendly name) of a Mac.
    e.g., "John's MacBook Pro"
    Returns the name as a string, or None if an error occurs or not on macOS.
    """
    hostname = socket.gethostname()

    if sys.platform != "darwin":
        return hostname

    try:
        process = await run_process(["scutil", "--get", "ComputerName"])
    except CalledProcessError:
        return hostname

    return process.stdout.decode("utf-8", errors="replace").strip() or hostname


async def _get_interface_types_from_networksetup() -> dict[str, InterfaceType]:
    """Parse networksetup -listallhardwareports to get interface types."""
    if sys.platform != "darwin":
        return {}

    try:
        result = await run_process(["networksetup", "-listallhardwareports"])
    except CalledProcessError:
        return {}

    types: dict[str, InterfaceType] = {}
    current_type: InterfaceType = "unknown"

    for line in result.stdout.decode().splitlines():
        if line.startswith("Hardware Port:"):
            port_name = line.split(":", 1)[1].strip()
            if "Wi-Fi" in port_name:
                current_type = "wifi"
            elif "Ethernet" in port_name or "LAN" in port_name:
                current_type = "ethernet"
            elif port_name.startswith("Thunderbolt"):
                current_type = "thunderbolt"
            else:
                current_type = "unknown"
        elif line.startswith("Device:"):
            device = line.split(":", 1)[1].strip()
            # enX is ethernet adapters or thunderbolt - these must be deprioritised
            if device.startswith("en") and device not in ["en0", "en1"]:
                current_type = "maybe_ethernet"
            types[device] = current_type

    return types


async def get_network_interfaces() -> list[NetworkInterfaceInfo]:
    """
    Retrieves detailed network interface information on macOS.
    Parses output from 'networksetup -listallhardwareports' and 'ifconfig'
    to determine interface names, IP addresses, and types (ethernet, wifi, vpn, other).
    Returns a list of NetworkInterfaceInfo objects.
    """
    interfaces_info: list[NetworkInterfaceInfo] = []
    interface_types = await _get_interface_types_from_networksetup()

    for iface, services in psutil.net_if_addrs().items():
        for service in services:
            match service.family:
                case socket.AF_INET | socket.AF_INET6:
                    interfaces_info.append(
                        NetworkInterfaceInfo(
                            name=iface,
                            ip_address=service.address,
                            interface_type=interface_types.get(iface, "unknown"),
                        )
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


def profile_memory_bandwidth() -> int:
    """Profile device memory bandwidth using MLX GPU operations.

    Uses a large array copy on the GPU to measure unified memory bandwidth.
    Returns measured bandwidth in bytes/second.
    Raises exception if MLX is unavailable or profiling fails.
    """
    import mlx.core as mx

    if not mx.metal.is_available():
        raise RuntimeError("Metal is not available")

    # Use 2GB buffer to better saturate memory bandwidth
    size_bytes = 2 * 1024 * 1024 * 1024
    side = math.isqrt(size_bytes // 4)  # Square 2D array of float32
    shape = (side, side)
    actual_bytes = side * side * 4
    bytes_transferred = actual_bytes * 2  # read + write

    # Warm-up: run the full benchmark operation multiple times to stabilize GPU
    for _ in range(3):
        src = mx.random.uniform(shape=shape, dtype=mx.float32)
        mx.eval(src)
        dst = src + 0.0
        mx.eval(dst)
        mx.synchronize()
        del src, dst

    # Benchmark: measure time to copy array
    best_bandwidth = 0.0
    num_runs = 4

    for _ in range(num_runs):
        src = mx.random.uniform(shape=shape, dtype=mx.float32)
        mx.eval(src)
        mx.synchronize()

        start = time.perf_counter()
        dst = src + 0.0
        mx.eval(dst)
        mx.synchronize()
        end = time.perf_counter()

        bandwidth = bytes_transferred / (end - start)
        best_bandwidth = max(best_bandwidth, bandwidth)

        del src, dst

    return int(best_bandwidth)
