import socket
import sys
from subprocess import CalledProcessError

import psutil
from anyio import run_process

from exo.shared.types.memory import Memory
from exo.shared.types.profiling import GpuDeviceInfo, NodeGpuInfo
from exo.shared.types.profiling import InterfaceType, NetworkInterfaceInfo


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


async def _get_gpu_devices_from_pynvml() -> list[GpuDeviceInfo] | None:
    try:
        import pynvml  # type: ignore[import-not-found]
    except Exception:
        return None

    try:
        pynvml.nvmlInit()
    except Exception:
        return None

    devices: list[GpuDeviceInfo] = []
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        for index in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            devices.append(
                GpuDeviceInfo(
                    name=str(name),
                    uuid=str(uuid),
                    memory_total=Memory.from_bytes(int(mem.total)),
                    memory_free=Memory.from_bytes(int(mem.free)),
                    memory_used=Memory.from_bytes(int(mem.used)),
                )
            )
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    return devices


async def _get_gpu_devices_from_nvidia_smi() -> list[GpuDeviceInfo]:
    result = await run_process(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,memory.free,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
    )
    if result.returncode != 0:
        return []

    devices: list[GpuDeviceInfo] = []
    for line in result.stdout.decode("utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        name, uuid, total_mb, free_mb, used_mb = parts[:5]
        try:
            devices.append(
                GpuDeviceInfo(
                    name=name,
                    uuid=uuid or None,
                    memory_total=Memory.from_mb(float(total_mb)),
                    memory_free=Memory.from_mb(float(free_mb)),
                    memory_used=Memory.from_mb(float(used_mb)),
                )
            )
        except ValueError:
            continue
    return devices


async def get_gpu_info() -> NodeGpuInfo:
    devices = await _get_gpu_devices_from_pynvml()
    if devices is None:
        devices = await _get_gpu_devices_from_nvidia_smi()
    return NodeGpuInfo(devices=devices)
