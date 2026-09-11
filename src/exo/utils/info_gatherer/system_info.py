import platform
import socket
import sys
from collections.abc import Mapping
from subprocess import CalledProcessError
from typing import Protocol

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


class _HasSpeed(Protocol):
    """The one field we need from psutil's per-interface stats - narrowed to
    a Protocol so the parsing logic below can be tested without going
    through psutil at all. Read-only to match psutil's NamedTuple result."""

    @property
    def speed(self) -> int: ...


def _active_speeds_from_stats(stats: Mapping[str, _HasSpeed]) -> dict[str, int]:
    """Currently negotiated link speed per interface, in Mbps.

    Pure function over psutil's stats (injected rather than fetched here)
    so it's testable without a live network stack. psutil reports 0 when
    the OS doesn't expose a speed for that interface - common for Wi-Fi,
    especially on macOS - and we treat that the same as not knowing it.
    """
    return {iface: stat.speed for iface, stat in stats.items() if stat.speed > 0}


def _get_active_speeds_mbps() -> dict[str, int]:
    try:
        stats = psutil.net_if_stats()
    except OSError:
        return {}
    return _active_speeds_from_stats(stats)


# Media type name (as ifconfig prints it, case-insensitive) -> its
# theoretical maximum speed, in Mbps. Deliberately conservative: an
# unrecognised media type is dropped rather than guessed at.
_MEDIA_SPEED_MBPS: dict[str, int] = {
    "10baset/utp": 10,
    "10baset": 10,
    "100basetx": 100,
    "100baset4": 100,
    "1000baset": 1000,
    "1000basetx": 1000,
    "1000basesx": 1000,
    "2500baset": 2500,
    "5000baset": 5000,
    "10gbase-t": 10_000,
    "25gbase-t": 25_000,
    "40gbase-t": 40_000,
}


def _parse_supported_media_mbps(ifconfig_verbose_output: str) -> dict[str, int]:
    """Parse the "supported media" block `ifconfig -v` prints per wired
    interface into the fastest media type each interface's hardware can
    negotiate, in Mbps.

    Unverified against real macOS output - built from the documented BSD
    ifconfig format, not a live capture - so treat this as a best-effort
    starting point rather than a guarantee, and adjust the parsing if it
    turns out not to match. An interface with no recognised media type
    (Wi-Fi, virtual interfaces, or a format this doesn't expect) is simply
    absent from the result rather than reported as zero.
    """
    speeds: dict[str, int] = {}
    current_iface: str | None = None
    in_supported_media = False

    for raw_line in ifconfig_verbose_output.splitlines():
        if raw_line and not raw_line[0].isspace():
            current_iface = raw_line.split(":", 1)[0].strip()
            in_supported_media = False
            continue

        line = raw_line.strip()
        if not current_iface:
            continue

        if line == "supported media:":
            in_supported_media = True
            continue
        if not in_supported_media:
            continue
        if not line.startswith("media "):
            # A differently-indented line ends the block (e.g. the next
            # interface's first attribute line, on some ifconfig versions).
            in_supported_media = False
            continue

        media_type = line.split()[1].lower()
        mbps = _MEDIA_SPEED_MBPS.get(media_type)
        if mbps is not None:
            speeds[current_iface] = max(speeds.get(current_iface, 0), mbps)

    return speeds


async def _get_supported_speeds_mbps() -> dict[str, int]:
    """Maximum speed each wired interface's hardware supports, in Mbps.

    macOS-only and best-effort - see `_parse_supported_media_mbps`.
    """
    if sys.platform != "darwin":
        return {}

    try:
        result = await run_process(["ifconfig", "-v"])
    except CalledProcessError:
        return {}

    return _parse_supported_media_mbps(result.stdout.decode(errors="replace"))


async def get_network_interfaces() -> list[NetworkInterfaceInfo]:
    """
    Retrieves detailed network interface information on macOS.
    Parses output from 'networksetup -listallhardwareports' and 'ifconfig'
    to determine interface names, IP addresses, and types (ethernet, wifi, vpn, other).
    Returns a list of NetworkInterfaceInfo objects.
    """
    interfaces_info: list[NetworkInterfaceInfo] = []
    interface_types = await _get_interface_types_from_networksetup()
    active_speeds = _get_active_speeds_mbps()
    supported_speeds = await _get_supported_speeds_mbps()

    for iface, services in psutil.net_if_addrs().items():
        for service in services:
            match service.family:
                case socket.AF_INET | socket.AF_INET6:
                    interfaces_info.append(
                        NetworkInterfaceInfo(
                            name=iface,
                            ip_address=service.address,
                            interface_type=interface_types.get(iface, "unknown"),
                            active_speed_mbps=active_speeds.get(iface),
                            supported_speed_mbps=supported_speeds.get(iface),
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
