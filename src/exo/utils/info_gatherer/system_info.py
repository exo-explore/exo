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


def _classify_unknown_darwin_interface(name: str) -> InterfaceType:
    if name.lower().startswith("anpi"):
        return "thunderbolt"
    return "unknown"


async def _get_linux_network_interfaces() -> list[NetworkInterfaceInfo]:
    import json as _json

    try:
        result = await run_process(["ip", "-j", "addr", "show"])
    except (CalledProcessError, FileNotFoundError):
        return []

    data: list[dict[str, object]] = _json.loads(result.stdout)  # pyright: ignore[reportAny]
    interfaces: list[NetworkInterfaceInfo] = []
    for iface in data:
        name: str = iface.get("ifname", "")  # pyright: ignore[reportAssignmentType, reportAny]
        link_type: str = iface.get("link_type", "")  # pyright: ignore[reportAssignmentType, reportAny]

        iface_type: InterfaceType
        if link_type == "loopback":
            continue
        elif link_type == "ether":
            if name.startswith(("wl", "wlan")):
                iface_type = "wifi"
            elif name.startswith(("docker", "br-", "veth")):
                iface_type = "unknown"
            elif name.startswith(("thunderbolt", "tb", "enx")):
                iface_type = "thunderbolt"
            else:
                iface_type = "ethernet"
        elif link_type in ("none", "tun"):
            iface_type = "unknown"
        else:
            iface_type = "unknown"

        for addr_info in iface.get("addr_info", []):  # pyright: ignore[reportAny]
            family: str = addr_info.get("family", "")  # pyright: ignore[reportAny]
            ip: str = addr_info.get("local", "")  # pyright: ignore[reportAny]
            if family in ("inet", "inet6") and ip:
                interfaces.append(NetworkInterfaceInfo(name=name, ip_address=ip, interface_type=iface_type))

    return interfaces


async def get_network_interfaces() -> list[NetworkInterfaceInfo]:
    """
    Retrieves detailed network interface information on macOS or Linux.
    On MacOS: parses output from 'networksetup -listallhardwareports' and 'ifconfig'
    to determine interface names, IP addresses, and types (ethernet, wifi, vpn, other).
    Falls back to using ip -j addr show on other platforms.
    Returns a list of NetworkInterfaceInfo objects.
    """
    if sys.platform == "darwin":
        interfaces_info: list[NetworkInterfaceInfo] = []
        interface_types = await _get_interface_types_from_networksetup()
        for iface, services in psutil.net_if_addrs().items():
            for service in services:
                match service.family:
                    case socket.AF_INET | socket.AF_INET6:
                        iface_type = interface_types.get(iface, "unknown")
                        if iface_type == "unknown":
                            iface_type = _classify_unknown_darwin_interface(iface)
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

    return await _get_linux_network_interfaces()


def _read_dmi_field(name: str) -> str | None:
    """Read a single DMI sysfs field, returning ``None`` on failure."""
    try:
        path = Path(f"/sys/class/dmi/id/{name}")
        if path.exists():
            return path.read_text().strip()
    except (OSError, PermissionError):
        pass
    return None


async def _get_linux_model_and_chip() -> tuple[str, str]:
    """Get Linux system information using DMI and /proc/cpuinfo.

    Detects NVIDIA DGX Spark (DMI product_name ``"DGX_Spark"``) and other
    NVIDIA systems via the ``sys_vendor`` DMI field, falling back to generic
    Linux identification.
    """
    model = "Linux"
    chip = "Unknown Chip"

    product_name = _read_dmi_field("product_name")
    sys_vendor = _read_dmi_field("sys_vendor")

    # DGX Spark: DMI product_name may be "DGX_Spark" or "gx10" variant
    product_lower = (product_name or "").lower()
    if product_name and ("dgx" in product_lower or "gx10" in product_lower):
        model = "DGX Spark"
        try:
            process = await run_process(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
            )
            gpu_name = process.stdout.decode().strip().split("\n")[0]
            chip = gpu_name if gpu_name and gpu_name != "[N/A]" else "NVIDIA GB10"
        except (CalledProcessError, FileNotFoundError):
            chip = "NVIDIA GB10"
        return (model, chip)

    # Other NVIDIA systems (sys_vendor contains "NVIDIA")
    if sys_vendor and "NVIDIA" in sys_vendor:
        model = product_name.replace("_", " ") if product_name else "NVIDIA System"
        try:
            process = await run_process(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
            )
            gpu_name = process.stdout.decode().strip().split("\n")[0]
            if gpu_name and gpu_name != "[N/A]":
                chip = gpu_name
        except (CalledProcessError, FileNotFoundError):
            pass
        return (model, chip)

    # Generic Linux — detect laptop vs desktop via chassis_type
    # SMBIOS chassis types: 8,9,10,14,31,32 = portable/laptop
    chassis_type = _read_dmi_field("chassis_type")
    laptop_chassis_types = {"8", "9", "10", "14", "31", "32"}
    if chassis_type in laptop_chassis_types:
        model = "Linux Laptop"
    elif chassis_type is not None:
        model = "Linux Desktop"

    # Also check for battery as a fallback laptop indicator
    if model == "Linux" and Path("/sys/class/power_supply/BAT0").exists():
        model = "Linux Laptop"

    # Use /proc/cpuinfo for chip
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        try:
            for line in cpuinfo_path.read_text().splitlines():
                if line.startswith("model name"):
                    chip = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass

    return (model, chip)


async def get_model_and_chip() -> tuple[str, str]:
    """Get system model and chip information.

    On macOS, uses ``system_profiler``.  On Linux, reads DMI data from
    sysfs and CPU info from ``/proc/cpuinfo``.
    """
    model = "Unknown Model"
    chip = "Unknown Chip"

    if sys.platform == "linux":
        return await _get_linux_model_and_chip()

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
