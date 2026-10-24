import re
import shutil
from collections.abc import Sequence

import anyio

from exo.shared.types.profiling import RdmaDeviceInfo, RdmaPortInfo


def parse_ibv_devinfo_output(output: str) -> list[RdmaDeviceInfo]:
    """Parse output from ibv_devinfo / ibv_devinfo -v into structured RdmaDeviceInfo objects."""
    devices: list[RdmaDeviceInfo] = []
    current_device: dict[str, str] | None = None
    current_ports: list[RdmaPortInfo] = []
    current_port: dict[str, str | int] | None = None

    def _flush_port() -> None:
        nonlocal current_port
        if current_port is not None and "port" in current_port:
            port_num = int(current_port["port"])
            state = str(current_port.get("state", "PORT_DOWN"))
            default_phys = "LinkUp" if "PORT_ACTIVE" in state else "Disabled"
            physical_state = str(current_port.get("physical_state", default_phys))
            current_ports.append(
                RdmaPortInfo(
                    port=port_num,
                    state=state,
                    physical_state=physical_state,
                    link_layer=str(current_port.get("link_layer", "")),
                    active_speed=str(current_port.get("active_speed", "")),
                    active_width=str(current_port.get("active_width", "")),
                )
            )
            current_port = None

    def _flush_device() -> None:
        nonlocal current_device, current_ports
        _flush_port()
        if current_device is not None and "hca_id" in current_device:
            devices.append(
                RdmaDeviceInfo(
                    hca_id=current_device["hca_id"],
                    transport=current_device.get("transport", ""),
                    fw_ver=current_device.get("fw_ver", ""),
                    node_guid=current_device.get("node_guid", ""),
                    ports=list(current_ports),
                )
            )
            current_device = None
            current_ports = []

    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue

        hca_match = re.match(r"^hca_id:\s*(\S+)", line)
        if hca_match:
            _flush_device()
            current_device = {"hca_id": hca_match.group(1)}
            continue

        if current_device is None:
            continue

        port_match = re.match(r"^port:\s*(\d+)", line)
        if port_match:
            _flush_port()
            current_port = {"port": int(port_match.group(1))}
            continue

        if current_port is not None:
            state_match = re.match(r"^state:\s*(.+)", line)
            if state_match:
                current_port["state"] = state_match.group(1).strip()
                continue
            phys_state_match = re.match(r"^physical_state:\s*(.+)", line)
            if phys_state_match:
                current_port["physical_state"] = phys_state_match.group(1).strip()
                continue
            link_layer_match = re.match(r"^link_layer:\s*(.+)", line)
            if link_layer_match:
                current_port["link_layer"] = link_layer_match.group(1).strip()
                continue
            speed_match = re.match(r"^active_speed:\s*(.+)", line)
            if speed_match:
                current_port["active_speed"] = speed_match.group(1).strip()
                continue
            width_match = re.match(r"^active_width:\s*(.+)", line)
            if width_match:
                current_port["active_width"] = width_match.group(1).strip()
                continue

        transport_match = re.match(r"^transport:\s*(.+)", line)
        if transport_match:
            current_device["transport"] = transport_match.group(1).strip()
            continue

        fw_match = re.match(r"^fw_ver:\s*(.+)", line)
        if fw_match:
            current_device["fw_ver"] = fw_match.group(1).strip()
            continue

        guid_match = re.match(r"^node_guid:\s*(.+)", line)
        if guid_match:
            current_device["node_guid"] = guid_match.group(1).strip()
            continue

    _flush_device()
    return devices


def is_rdma_device_active(device: RdmaDeviceInfo) -> bool:
    """Check if an RDMA device has at least one active, linked-up port."""
    for port in device.ports:
        if (
            "PORT_ACTIVE" in port.state
            and "Disabled" not in port.physical_state
            and "Down" not in port.physical_state
        ):
            return True
    return False


def validate_rdma_connections(devices: Sequence[RdmaDeviceInfo]) -> bool:
    """Validate whether any RDMA physical connection is currently active."""
    return any(is_rdma_device_active(dev) for dev in devices)


async def gather_ibv_devinfo(
    command: Sequence[str] | None = None,
) -> tuple[bool, list[RdmaDeviceInfo]]:
    """Runs ibv_devinfo to validate RDMA device and connection availability on Linux."""
    cmd = list(command) if command is not None else ["ibv_devinfo", "-v"]
    executable = cmd[0]
    if shutil.which(executable) is None:
        return False, []

    try:
        with anyio.fail_after(5):
            proc = await anyio.run_process(cmd, check=False)
    except (TimeoutError, OSError):
        return False, []

    if proc.returncode != 0:
        return False, []

    output = proc.stdout.decode("utf-8", errors="replace").strip()
    devices = parse_ibv_devinfo_output(output)
    is_active = validate_rdma_connections(devices)
    return is_active, devices
