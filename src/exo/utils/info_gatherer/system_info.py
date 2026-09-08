import platform
import socket
import sys
from subprocess import CalledProcessError

import psutil
from anyio import run_process

from exo.shared.types.profiling import InterfaceType, NetworkInterfaceInfo


def get_os_version() -> str:
    """Return the OS version string for this node.

    On macOS this is the macOS version (e.g. ``"15.3"``).
    On Windows this is ``"Windows 10"`` / ``"Windows 11"``.
    On other platforms it falls back to the platform name (e.g. ``"Linux"``).
    """
    if sys.platform == "darwin":
        version = platform.mac_ver()[0]
        return version if version else "Unknown"
    if sys.platform == "win32":
        release = platform.release()
        return f"Windows {release}" if release else "Windows"
    return platform.system() or "Unknown"


async def get_os_build_version() -> str:
    """Return the OS build version string (e.g. ``"24D5055b"`` on macOS).

    On Windows this is the kernel version (e.g. ``"10.0.19045"``).
    On other non-macOS platforms, returns ``"Unknown"``.
    """
    if sys.platform == "win32":
        return platform.version() or "Unknown"

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
        iface_type = interface_types.get(iface, _guess_interface_type(iface))
        for service in services:
            match service.family:
                case socket.AF_INET | socket.AF_INET6:
                    interfaces_info.append(
                        NetworkInterfaceInfo(
                            name=iface,
                            ip_address=service.address,
                            interface_type=iface_type,
                        )
                    )
                case _:
                    pass

    return interfaces_info


def _guess_interface_type(iface: str) -> InterfaceType:
    """Best-effort type from the adapter name (Windows/Linux, PAIR-style)."""
    lowered = iface.lower()
    if any(token in lowered for token in ("wi-fi", "wifi", "wlan", "wireless")):
        return "wifi"
    if any(token in lowered for token in ("ethernet", "eth", "lan", "local area")):
        return "ethernet"
    return "unknown"


async def get_model_and_chip() -> tuple[str, str]:
    """Get machine model and accelerator/chip names."""
    model = "Unknown Model"
    chip = "Unknown Chip"

    if sys.platform == "win32":
        uname = platform.uname()
        model = (await _windows_computer_model()) or (
            f"{uname.system} {uname.release}".strip() or "Windows PC"
        )
        chip = (
            (await _windows_gpu_name())
            or uname.processor
            or uname.machine
            or "Unknown Chip"
        )
        return (model, chip)

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


async def _windows_computer_model() -> str | None:
    try:
        process = await run_process(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance -ClassName Win32_ComputerSystem).Model",
            ],
            check=False,
        )
    except OSError:
        return None
    if process.returncode != 0:
        return None
    model = process.stdout.decode("utf-8", errors="replace").strip()
    return model or None


async def _windows_gpu_name() -> str | None:
    try:
        process = await run_process(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            check=False,
        )
    except OSError:
        return None
    if process.returncode != 0:
        return None
    names = [
        line.strip()
        for line in process.stdout.decode("utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    return ", ".join(names) if names else None
