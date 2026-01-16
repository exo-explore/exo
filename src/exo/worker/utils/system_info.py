import socket
import sys
import time
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


def profile_memory_bandwidth() -> int | None:
    """
    Profile device memory bandwidth using MLX GPU operations.

    Uses a large array copy on the GPU to measure unified memory bandwidth.
    Returns measured bandwidth in bytes/second, or None if MLX is unavailable.
    """
    try:
        import mlx.core as mx

        if not mx.metal.is_available():
            return None

        # Use 2GB buffer to better saturate memory bandwidth
        # Use 2D shape to avoid potential issues with very large 1D arrays
        size_bytes = 2 * 1024 * 1024 * 1024
        side = int((size_bytes // 4) ** 0.5)  # Square 2D array of float32
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

            # Time the copy operation (src + 0.0 forces read of src, write of dst)
            start = time.perf_counter()
            dst = src + 0.0
            mx.eval(dst)
            mx.synchronize()
            end = time.perf_counter()

            bandwidth = bytes_transferred / (end - start)
            best_bandwidth = max(best_bandwidth, bandwidth)

            del src, dst

        return int(best_bandwidth)
    except Exception:
        return None


def get_memory_bandwidth(_chip_id: str) -> int | None:
    """
    Returns measured memory bandwidth in bytes/second.

    Uses MLX GPU operations for accurate unified memory bandwidth measurement.
    """
    return profile_memory_bandwidth()
