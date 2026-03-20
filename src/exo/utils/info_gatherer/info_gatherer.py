import os
import shutil
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from subprocess import CalledProcessError
from typing import Self, cast

import anyio
from anyio import fail_after, open_process, to_thread
from anyio.streams.buffered import BufferedByteReceiveStream
from loguru import logger
from pydantic import ValidationError

from exo.shared.constants import EXO_CONFIG_FILE, EXO_DEFAULT_MODELS_DIR
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import (
    DiskUsage,
    MemoryUsage,
    NetworkInterfaceInfo,
    ThunderboltBridgeStatus,
)
from exo.shared.types.thunderbolt import (
    ThunderboltConnection,
    ThunderboltConnectivity,
    ThunderboltIdentifier,
)
from exo.utils.channels import Sender
from exo.utils.pydantic_ext import TaggedModel
from exo.utils.task_group import TaskGroup

from .macmon import MacmonMetrics
from .system_info import (
    get_friendly_name,
    get_model_and_chip,
    get_network_interfaces,
    get_os_build_version,
    get_os_version,
)

IS_DARWIN = sys.platform == "darwin"


async def _get_thunderbolt_devices() -> set[str] | None:
    """Get Thunderbolt interface device names (e.g., en2, en3) from hardware ports.

    Returns None if the networksetup command fails.
    """
    result = await anyio.run_process(
        ["networksetup", "-listallhardwareports"],
        check=False,
    )
    if result.returncode != 0:
        logger.warning(
            f"networksetup -listallhardwareports failed with code "
            f"{result.returncode}: {result.stderr.decode()}"
        )
        return None

    output = result.stdout.decode()
    thunderbolt_devices: set[str] = set()
    current_port: str | None = None

    for line in output.splitlines():
        line = line.strip()
        if line.startswith("Hardware Port:"):
            current_port = line.split(":", 1)[1].strip()
        elif line.startswith("Device:") and current_port:
            device = line.split(":", 1)[1].strip()
            if "thunderbolt" in current_port.lower():
                thunderbolt_devices.add(device)
            current_port = None

    return thunderbolt_devices


async def _get_bridge_services() -> dict[str, str] | None:
    """Get mapping of bridge device -> service name from network service order.

    Returns None if the networksetup command fails.
    """
    result = await anyio.run_process(
        ["networksetup", "-listnetworkserviceorder"],
        check=False,
    )
    if result.returncode != 0:
        logger.warning(
            f"networksetup -listnetworkserviceorder failed with code "
            f"{result.returncode}: {result.stderr.decode()}"
        )
        return None

    # Parse service order to find bridge devices and their service names
    # Format: "(1) Service Name\n(Hardware Port: ..., Device: bridge0)\n"
    service_order_output = result.stdout.decode()
    bridge_services: dict[str, str] = {}  # device -> service name
    current_service: str | None = None

    for line in service_order_output.splitlines():
        line = line.strip()
        # Match "(N) Service Name" or "(*) Service Name" (disabled)
        # but NOT "(Hardware Port: ...)" lines
        if (
            line
            and line.startswith("(")
            and ")" in line
            and not line.startswith("(Hardware Port:")
        ):
            paren_end = line.index(")")
            if paren_end + 2 <= len(line):
                current_service = line[paren_end + 2 :]
        # Match "(Hardware Port: ..., Device: bridgeX)"
        elif current_service and "Device: bridge" in line:
            # Extract device name from "..., Device: bridge0)"
            device_start = line.find("Device: ") + len("Device: ")
            device_end = line.find(")", device_start)
            if device_end > device_start:
                device = line[device_start:device_end]
                bridge_services[device] = current_service

    return bridge_services


async def _get_bridge_members(bridge_device: str) -> set[str]:
    """Get member interfaces of a bridge device via ifconfig."""
    result = await anyio.run_process(
        ["ifconfig", bridge_device],
        check=False,
    )
    if result.returncode != 0:
        logger.debug(f"ifconfig {bridge_device} failed with code {result.returncode}")
        return set()

    members: set[str] = set()
    ifconfig_output = result.stdout.decode()
    for line in ifconfig_output.splitlines():
        line = line.strip()
        if line.startswith("member:"):
            parts = line.split()
            if len(parts) > 1:
                members.add(parts[1])

    return members


async def _find_thunderbolt_bridge(
    bridge_services: dict[str, str], thunderbolt_devices: set[str]
) -> str | None:
    """Find the service name of a bridge containing Thunderbolt interfaces.

    Returns the service name if found, None otherwise.
    """
    for bridge_device, service_name in bridge_services.items():
        members = await _get_bridge_members(bridge_device)
        if members & thunderbolt_devices:  # intersection is non-empty
            return service_name
    return None


async def _is_service_enabled(service_name: str) -> bool | None:
    """Check if a network service is enabled.

    Returns True if enabled, False if disabled, None on error.
    """
    result = await anyio.run_process(
        ["networksetup", "-getnetworkserviceenabled", service_name],
        check=False,
    )
    if result.returncode != 0:
        logger.warning(
            f"networksetup -getnetworkserviceenabled '{service_name}' "
            f"failed with code {result.returncode}: {result.stderr.decode()}"
        )
        return None

    stdout = result.stdout.decode().strip().lower()
    return stdout == "enabled"


class StaticNodeInformation(TaggedModel):
    """Node information that should NEVER change, to be gathered once at startup"""

    model: str
    chip: str
    os_version: str
    os_build_version: str

    @classmethod
    async def gather(cls) -> Self:
        model, chip = await get_model_and_chip()
        return cls(
            model=model,
            chip=chip,
            os_version=get_os_version(),
            os_build_version=await get_os_build_version(),
        )


class NodeNetworkInterfaces(TaggedModel):
    ifaces: Sequence[NetworkInterfaceInfo]


class MacThunderboltIdentifiers(TaggedModel):
    idents: Sequence[ThunderboltIdentifier]


class MacThunderboltConnections(TaggedModel):
    conns: Sequence[ThunderboltConnection]


class RdmaCtlStatus(TaggedModel):
    enabled: bool

    @classmethod
    async def gather(cls) -> Self | None:
        if not IS_DARWIN or shutil.which("rdma_ctl") is None:
            return None
        try:
            with anyio.fail_after(5):
                proc = await anyio.run_process(["rdma_ctl", "status"], check=False)
        except (TimeoutError, OSError):
            return None
        if proc.returncode != 0:
            return None
        output = proc.stdout.decode("utf-8").lower().strip()
        if "enabled" in output:
            return cls(enabled=True)
        if "disabled" in output:
            return cls(enabled=False)
        return None


class ThunderboltBridgeInfo(TaggedModel):
    status: ThunderboltBridgeStatus

    @classmethod
    async def gather(cls) -> Self | None:
        """Check if a Thunderbolt Bridge network service is enabled on this node.

        Detection approach:
        1. Find all Thunderbolt interface devices (en2, en3, etc.) from hardware ports
        2. Find bridge devices from network service order (not hardware ports, as
           bridges may not appear there)
        3. Check each bridge's members via ifconfig
        4. If a bridge contains Thunderbolt interfaces, it's a Thunderbolt Bridge
        5. Check if that network service is enabled
        """
        if not IS_DARWIN:
            return None

        def _no_bridge_status() -> Self:
            return cls(
                status=ThunderboltBridgeStatus(
                    enabled=False, exists=False, service_name=None
                )
            )

        try:
            tb_devices = await _get_thunderbolt_devices()
            if tb_devices is None:
                return _no_bridge_status()

            bridge_services = await _get_bridge_services()
            if not bridge_services:
                return _no_bridge_status()

            tb_service_name = await _find_thunderbolt_bridge(
                bridge_services, tb_devices
            )
            if not tb_service_name:
                return _no_bridge_status()

            enabled = await _is_service_enabled(tb_service_name)
            if enabled is None:
                return cls(
                    status=ThunderboltBridgeStatus(
                        enabled=False, exists=True, service_name=tb_service_name
                    )
                )

            return cls(
                status=ThunderboltBridgeStatus(
                    enabled=enabled,
                    exists=True,
                    service_name=tb_service_name,
                )
            )
        except Exception as e:
            logger.opt(exception=e).warning("Failed to gather Thunderbolt Bridge info")
            return None


class NodeConfig(TaggedModel):
    """Node configuration from EXO_CONFIG_FILE, reloaded from the file only at startup. Other changes should come in through the API and propagate from there"""

    @classmethod
    async def gather(cls) -> Self | None:
        cfg_file = anyio.Path(EXO_CONFIG_FILE)
        await cfg_file.parent.mkdir(parents=True, exist_ok=True)
        await cfg_file.touch(exist_ok=True)
        async with await cfg_file.open("rb") as f:
            try:
                contents = (await f.read()).decode("utf-8")
                data = tomllib.loads(contents)
                return cls.model_validate(data)
            except (tomllib.TOMLDecodeError, UnicodeDecodeError, ValidationError):
                logger.warning("Invalid config file, skipping...")
                return None


class MiscData(TaggedModel):
    """Node information that may slowly change that doesn't fall into the other categories"""

    friendly_name: str

    @classmethod
    async def gather(cls) -> Self:
        return cls(friendly_name=await get_friendly_name())


class NodeDiskUsage(TaggedModel):
    """Disk space information for the models directory."""

    disk_usage: DiskUsage

    @classmethod
    async def gather(cls) -> Self:
        return cls(
            disk_usage=await to_thread.run_sync(
                lambda: DiskUsage.from_path(EXO_DEFAULT_MODELS_DIR)
            )
        )


async def _gather_iface_map() -> dict[str, str] | None:
    proc = await anyio.run_process(
        ["networksetup", "-listallhardwareports"], check=False
    )
    if proc.returncode != 0:
        return None

    ports: dict[str, str] = {}
    port = ""
    for line in proc.stdout.decode("utf-8").split("\n"):
        if line.startswith("Hardware Port:"):
            port = line.split(": ")[1]
        elif line.startswith("Device:"):
            ports[port] = line.split(": ")[1]
            port = ""
    if "" in ports:
        del ports[""]
    return ports


GatheredInfo = (
    MacmonMetrics
    | MemoryUsage
    | NodeNetworkInterfaces
    | MacThunderboltIdentifiers
    | MacThunderboltConnections
    | RdmaCtlStatus
    | ThunderboltBridgeInfo
    | NodeConfig
    | MiscData
    | StaticNodeInformation
    | NodeDiskUsage
)


@dataclass
class InfoGatherer:
    info_sender: Sender[GatheredInfo]
    _tg: TaskGroup = field(init=False, default_factory=TaskGroup)
    _psutil_enabled: bool = field(init=False, default=False)

    async def _can_read_macmon_metrics(self, macmon_path: str) -> bool:
        try:
            with fail_after(5):
                proc = await anyio.run_process(
                    [macmon_path, "pipe", "--samples", "1", "--interval", "100"],
                    check=False,
                )
        except Exception as e:
            logger.opt(exception=e).warning(
                f"Failed to validate macmon at {macmon_path}"
            )
            return False

        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            logger.warning(
                f"macmon preflight failed with return code {proc.returncode}: "
                f"{stderr or 'no stderr'}"
            )
            return False

        stdout = proc.stdout.decode("utf-8", errors="replace").strip()
        if not stdout:
            logger.warning("macmon preflight returned no metrics")
            return False

        try:
            MacmonMetrics.from_raw_json(stdout.splitlines()[0])
        except ValidationError as e:
            logger.opt(exception=e).warning(
                "macmon preflight returned unexpected metrics JSON"
            )
            return False

        return True

    async def run(self):
        async with self._tg as tg:
            if IS_DARWIN:
                tg.start_soon(self._monitor_macmon, 1)
                tg.start_soon(self._monitor_system_profiler_thunderbolt_data, 5)
                tg.start_soon(self._monitor_thunderbolt_bridge_status, 10)
                tg.start_soon(self._monitor_rdma_ctl_status, 10)
            if not IS_DARWIN:
                tg.start_soon(self._monitor_memory_usage, 1)
            tg.start_soon(self._watch_system_info, 10)
            tg.start_soon(self._monitor_misc, 60)
            tg.start_soon(self._monitor_static_info, 60)
            tg.start_soon(self._monitor_disk_usage, 30)

            nc = await NodeConfig.gather()
            if nc is not None:
                await self.info_sender.send(nc)

    def shutdown(self):
        self._tg.cancel_tasks()

    async def _monitor_static_info(self, static_info_poll_interval: float):
        while True:
            try:
                with fail_after(30):
                    await self.info_sender.send(await StaticNodeInformation.gather())
            except Exception as e:
                logger.opt(exception=e).warning("Error gathering static node info")
            await anyio.sleep(static_info_poll_interval)

    async def _monitor_misc(self, misc_poll_interval: float):
        while True:
            try:
                with fail_after(10):
                    await self.info_sender.send(await MiscData.gather())
            except Exception as e:
                logger.opt(exception=e).warning("Error gathering misc data")
            await anyio.sleep(misc_poll_interval)

    async def _monitor_system_profiler_thunderbolt_data(
        self, system_profiler_interval: float
    ):
        while True:
            try:
                with fail_after(30):
                    iface_map = await _gather_iface_map()
                    if iface_map is None:
                        raise ValueError("Failed to gather interface map")

                    data = await ThunderboltConnectivity.gather()
                    assert data is not None

                    idents = [
                        it for i in data if (it := i.ident(iface_map)) is not None
                    ]
                    await self.info_sender.send(
                        MacThunderboltIdentifiers(idents=idents)
                    )

                    conns = [it for i in data if (it := i.conn()) is not None]
                    await self.info_sender.send(MacThunderboltConnections(conns=conns))
            except Exception as e:
                logger.opt(exception=e).warning("Error gathering Thunderbolt data")
            await anyio.sleep(system_profiler_interval)

    async def _monitor_memory_usage(self, memory_poll_rate: float):
        if self._psutil_enabled:
            return
        self._psutil_enabled = True
        override_memory_env = os.getenv("OVERRIDE_MEMORY_MB")
        override_memory: int | None = (
            Memory.from_mb(int(override_memory_env)).in_bytes
            if override_memory_env
            else None
        )
        while True:
            try:
                await self.info_sender.send(
                    MemoryUsage.from_psutil(override_memory=override_memory)
                )
            except Exception as e:
                logger.opt(exception=e).warning("Error gathering memory usage")
            await anyio.sleep(memory_poll_rate)

    async def _watch_system_info(self, interface_watcher_interval: float):
        while True:
            try:
                with fail_after(10):
                    nics = await get_network_interfaces()
                    await self.info_sender.send(NodeNetworkInterfaces(ifaces=nics))
            except Exception as e:
                logger.opt(exception=e).warning("Error gathering network interfaces")
            await anyio.sleep(interface_watcher_interval)

    async def _monitor_thunderbolt_bridge_status(
        self, thunderbolt_bridge_poll_interval: float
    ):
        while True:
            try:
                with fail_after(30):
                    curr = await ThunderboltBridgeInfo.gather()
                    if curr is not None:
                        await self.info_sender.send(curr)
            except Exception as e:
                logger.opt(exception=e).warning(
                    "Error gathering Thunderbolt Bridge status"
                )
            await anyio.sleep(thunderbolt_bridge_poll_interval)

    async def _monitor_rdma_ctl_status(self, rdma_ctl_poll_interval: float):
        while True:
            try:
                curr = await RdmaCtlStatus.gather()
                if curr is not None:
                    await self.info_sender.send(curr)
            except Exception as e:
                logger.opt(exception=e).warning("Error gathering RDMA ctl status")
            await anyio.sleep(rdma_ctl_poll_interval)

    async def _monitor_disk_usage(self, disk_poll_interval: float):
        while True:
            try:
                with fail_after(5):
                    await self.info_sender.send(await NodeDiskUsage.gather())
            except Exception as e:
                logger.opt(exception=e).warning("Error gathering disk usage")
            await anyio.sleep(disk_poll_interval)

    async def _monitor_macmon(self, macmon_interval: float):
        if (
            macmon_path := os.getenv("EXO_MACMON_PATH") or shutil.which("macmon")
        ) is None:
            logger.warning(
                "macmon not found, falling back to psutil for memory monitoring"
            )
            self._tg.start_soon(self._monitor_memory_usage, 1)
            return
        if not await self._can_read_macmon_metrics(macmon_path):
            logger.warning(
                f"macmon at {macmon_path} is unusable, falling back to psutil memory monitoring"
            )
            self._tg.start_soon(self._monitor_memory_usage, 1)
            return
        # macmon pipe --interval [interval in ms]
        # Timeout: if macmon produces no output for this many seconds, restart it.
        # macmon writes every macmon_interval seconds, so 10x that is generous.
        read_timeout = max(macmon_interval * 10, 30)
        while True:
            try:
                async with await open_process(
                    [
                        macmon_path,
                        "pipe",
                        "--interval",
                        str(macmon_interval * 1000),
                    ]
                ) as p:
                    if not p.stdout:
                        logger.critical("MacMon closed stdout")
                        return
                    stream = BufferedByteReceiveStream(p.stdout)
                    while True:
                        with fail_after(read_timeout):
                            data = await stream.receive_until(
                                delimiter=b"\n", max_bytes=8 * 1024
                            )
                            text = data.decode("utf-8", errors="replace").strip()
                            metrics = MacmonMetrics.from_raw_json(text)
                        await self.info_sender.send(metrics)
            except TimeoutError:
                logger.warning(
                    f"MacMon produced no output for {read_timeout}s, restarting"
                )
                self._tg.start_soon(self._monitor_memory_usage, 1)
            except CalledProcessError as e:
                stderr_msg = "no stderr"
                stderr_output = cast(bytes | str | None, e.stderr)
                if stderr_output is not None:
                    stderr_msg = (
                        stderr_output.decode()
                        if isinstance(stderr_output, bytes)
                        else str(stderr_output)
                    )
                logger.warning(
                    f"MacMon failed with return code {e.returncode}: {stderr_msg}"
                )
                self._tg.start_soon(self._monitor_memory_usage, 1)
            except Exception as e:
                logger.opt(exception=e).warning("Error in macmon monitor")
                self._tg.start_soon(self._monitor_memory_usage, 1)
            await anyio.sleep(macmon_interval)
