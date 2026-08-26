import platform
import socket
import sys
from pathlib import Path
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


def _classify_linux_interface(
    interface_name: str,
) -> tuple[InterfaceType, int | None]:
    """Classify a Linux interface via sysfs and report its negotiated link speed.

    ``/sys/class/net/<iface>/speed`` holds the negotiated speed in Mb/s for
    wired links (reads fail or report a non-positive value for wireless,
    virtual, or down interfaces). The distinction matters for ring host
    selection: e.g. a DGX Spark exposes both a management ethernet port and
    200 GbE ConnectX ports, and only the link speed tells them apart.
    """
    sysfs_path = Path("/sys/class/net") / interface_name
    if (sysfs_path / "wireless").exists():
        return "wifi", None

    link_speed_megabits: int | None = None
    try:
        link_speed_megabits = int((sysfs_path / "speed").read_text().strip())
    except (OSError, ValueError):
        link_speed_megabits = None
    if link_speed_megabits is not None and link_speed_megabits <= 0:
        link_speed_megabits = None

    # A "device" symlink marks a physical (non-virtual) interface.
    if (sysfs_path / "device").exists():
        return "ethernet", link_speed_megabits
    return "unknown", link_speed_megabits


async def get_network_interfaces() -> list[NetworkInterfaceInfo]:
    """
    Retrieves detailed network interface information.

    On macOS, parses 'networksetup -listallhardwareports' output to determine
    interface types (ethernet, wifi, thunderbolt). On Linux, classifies
    interfaces via sysfs and reports their negotiated link speed.
    Returns a list of NetworkInterfaceInfo objects.
    """
    interfaces_info: list[NetworkInterfaceInfo] = []
    interface_types = await _get_interface_types_from_networksetup()
    is_linux = sys.platform == "linux"

    for iface, services in psutil.net_if_addrs().items():
        interface_type = interface_types.get(iface, "unknown")
        link_speed_megabits: int | None = None
        if is_linux:
            interface_type, link_speed_megabits = _classify_linux_interface(iface)
        for service in services:
            match service.family:
                case socket.AF_INET | socket.AF_INET6:
                    interfaces_info.append(
                        NetworkInterfaceInfo(
                            name=iface,
                            ip_address=service.address,
                            interface_type=interface_type,
                            link_speed_megabits=link_speed_megabits,
                        )
                    )
                case _:
                    pass

    return interfaces_info


async def _get_cuda_gpu_name() -> str | None:
    """Name of the first CUDA GPU via nvidia-smi (e.g. "NVIDIA GeForce RTX 3090").

    Returns None when nvidia-smi is unavailable or fails.
    """
    try:
        process = await run_process(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
        )
    except (CalledProcessError, OSError):
        return None
    if process.returncode != 0:
        return None
    first_line = process.stdout.decode("utf-8", errors="replace").strip().splitlines()
    return first_line[0].strip() if first_line else None


async def get_model_and_chip() -> tuple[str, str]:
    """Get system model and chip information.

    On macOS this uses system_profiler. On other platforms the chip is the
    CUDA GPU name when one is present (it identifies the accelerator that
    actually runs inference, which placement uses to estimate memory
    bandwidth).
    """
    model = "Unknown Model"
    chip = "Unknown Chip"

    if sys.platform != "darwin":
        gpu_name = await _get_cuda_gpu_name()
        if gpu_name is not None:
            chip = gpu_name
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
