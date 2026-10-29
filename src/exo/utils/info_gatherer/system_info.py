import platform
import socket
import sys
from subprocess import CalledProcessError

import psutil
from anyio import run_process

from exo.shared.types.memory import Memory
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


def get_gpu_working_set() -> Memory | None:
    """The most memory this Mac's GPU should use: Metal's recommended working set, which is
    also the limit a runner wires its model's memory up to. None without Metal."""
    if sys.platform != "darwin":
        return None
    try:
        import mlx.core as mx
    except ImportError:
        return None
    if not mx.metal.is_available():
        return None
    return Memory.from_bytes(int(mx.device_info()["max_recommended_working_set_size"]))


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


def get_heuristic_interface_type(ifname: str) -> InterfaceType:
    """Heuristic identification of interface type based on interface name prefix.

    Provides cross-platform support for Linux, Windows, and macOS fallback,
    referencing old exo helpers.py#L283.
    """
    ifname_lower = ifname.lower()
    if ifname_lower.startswith(("tb5", "thunderbolt5")):
        return "tb5"
    if ifname_lower.startswith(("tb", "thunderbolt", "ten", "nx")):
        return "thunderbolt"
    if ifname_lower.startswith(("ib", "mlx")):
        return "rdma"
    if (
        ifname_lower.startswith(
            (
                "docker",
                "br-",
                "veth",
                "cni",
                "flannel",
                "calico",
                "weave",
                "tun",
                "tap",
                "vtun",
                "utun",
                "gif",
                "stf",
                "awdl",
                "llw",
                "lo",
            )
        )
        or "bridge" in ifname_lower
    ):
        return "virtual"
    if ifname_lower.startswith(("wlan", "wifi", "wl")):
        return "wifi"
    if ifname_lower.startswith(("eth", "en", "eno", "ens", "enp")):
        return "ethernet"
    return "unknown"


def _parse_networksetup_ports(output: str) -> dict[str, InterfaceType]:
    """Parse text output of networksetup -listallhardwareports."""
    types: dict[str, InterfaceType] = {}
    current_type: InterfaceType = "unknown"

    for line in output.splitlines():
        if line.startswith("Hardware Port:"):
            port_name = line.split(":", 1)[1].strip()
            port_lower = port_name.lower()
            if "thunderbolt 5" in port_lower or "tb5" in port_lower:
                current_type = "tb5"
            elif "thunderbolt" in port_lower:
                current_type = "thunderbolt"
            elif "wi-fi" in port_lower or "airport" in port_lower:
                current_type = "wifi"
            elif "ethernet" in port_lower or "lan" in port_lower:
                current_type = "ethernet"
            else:
                current_type = "unknown"
        elif line.startswith("Device:"):
            device = line.split(":", 1)[1].strip()
            # If current_type is unknown and device is non-default enX, mark maybe_ethernet
            if (
                current_type == "unknown"
                and device.startswith("en")
                and device not in ["en0", "en1"]
            ):
                types[device] = "maybe_ethernet"
            else:
                types[device] = current_type

    return types


async def _get_interface_types_from_networksetup() -> dict[str, InterfaceType]:
    """Parse networksetup -listallhardwareports to get interface types."""
    if sys.platform != "darwin":
        return {}

    try:
        result = await run_process(["networksetup", "-listallhardwareports"])
    except CalledProcessError:
        return {}

    return _parse_networksetup_ports(result.stdout.decode())


async def get_network_interfaces() -> list[NetworkInterfaceInfo]:
    """Retrieves detailed network interface information.

    Parses hardware ports on macOS or uses heuristic name inspection across
    Linux/Windows to determine interface names, IP addresses, and types (TB5,
    Thunderbolt, Ethernet, WiFi, RDMA, Virtual).
    """
    interfaces_info: list[NetworkInterfaceInfo] = []
    interface_types = await _get_interface_types_from_networksetup()

    for iface, services in psutil.net_if_addrs().items():
        detected_type = interface_types.get(iface, "unknown")
        if detected_type == "unknown":
            detected_type = get_heuristic_interface_type(iface)

        for service in services:
            match service.family:
                case socket.AF_INET | socket.AF_INET6:
                    interfaces_info.append(
                        NetworkInterfaceInfo(
                            name=iface,
                            ip_address=service.address,
                            interface_type=detected_type,
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
