import os
import shutil
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from subprocess import CalledProcessError
from typing import Self, cast

import anyio
from anyio import create_task_group, open_process
from anyio.abc import TaskGroup
from anyio.streams.buffered import BufferedByteReceiveStream
from anyio.streams.text import TextReceiveStream
from loguru import logger

from exo.shared.constants import EXO_CONFIG_FILE
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import (
    MemoryUsage,
    NetworkInterfaceInfo,
    NodeGpuInfo,
    ThunderboltBridgeStatus,
)
from exo.shared.types.thunderbolt import (
    ThunderboltConnection,
    ThunderboltConnectivity,
    ThunderboltIdentifier,
)
from exo.utils.channels import Sender
from exo.utils.pydantic_ext import TaggedModel

from .macmon import MacmonMetrics
from .system_info import (
    get_friendly_name,
    get_gpu_info,
    get_model_and_chip,
    get_network_interfaces,
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

    @classmethod
    async def gather(cls) -> Self:
        model, chip = await get_model_and_chip()
        return cls(model=model, chip=chip)


class NodeNetworkInterfaces(TaggedModel):
    ifaces: Sequence[NetworkInterfaceInfo]


class NodeGpuDevices(TaggedModel):
    info: NodeGpuInfo

    @classmethod
    async def gather(cls) -> Self:
        return cls(info=await get_gpu_info())


class MacThunderboltIdentifiers(TaggedModel):
    idents: Sequence[ThunderboltIdentifier]


class MacThunderboltConnections(TaggedModel):
    conns: Sequence[ThunderboltConnection]


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
            logger.warning(f"Failed to gather Thunderbolt Bridge info: {e}")
            return None


class NodeConfig(TaggedModel):
    """Node configuration from EXO_CONFIG_FILE, reloaded from the file only at startup. Other changes should come in through the API and propagate from there"""

    @classmethod
    async def gather(cls) -> Self | None:
        cfg_file = anyio.Path(EXO_CONFIG_FILE)
        await cfg_file.touch(exist_ok=True)
        async with await cfg_file.open("rb") as f:
            try:
                contents = (await f.read()).decode("utf-8")
                data = tomllib.loads(contents)
                return cls.model_validate(data)
            except (tomllib.TOMLDecodeError, UnicodeDecodeError):
                logger.warning("Invalid config file, skipping...")
                return None


class MiscData(TaggedModel):
    """Node information that may slowly change that doesn't fall into the other categories"""

    friendly_name: str

    @classmethod
    async def gather(cls) -> Self:
        return cls(friendly_name=await get_friendly_name())


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
    | NodeGpuDevices
    | MacThunderboltIdentifiers
    | MacThunderboltConnections
    | ThunderboltBridgeInfo
    | NodeConfig
    | MiscData
    | StaticNodeInformation
)


@dataclass
class InfoGatherer:
    info_sender: Sender[GatheredInfo]
    interface_watcher_interval: float | None = 10
    misc_poll_interval: float | None = 60
    system_profiler_interval: float | None = 5 if IS_DARWIN else None
    memory_poll_rate: float | None = None if IS_DARWIN else 1
    macmon_interval: float | None = 1 if IS_DARWIN else None
    thunderbolt_bridge_poll_interval: float | None = 10 if IS_DARWIN else None
    gpu_poll_interval: float | None = 10
    _tg: TaskGroup = field(init=False, default_factory=create_task_group)

    async def run(self):
        async with self._tg as tg:
            if IS_DARWIN:
                if (macmon_path := shutil.which("macmon")) is not None:
                    tg.start_soon(self._monitor_macmon, macmon_path)
                tg.start_soon(self._monitor_system_profiler_thunderbolt_data)
                tg.start_soon(self._monitor_thunderbolt_bridge_status)
            tg.start_soon(self._watch_system_info)
            tg.start_soon(self._monitor_memory_usage)
            tg.start_soon(self._monitor_gpu_info)
            tg.start_soon(self._monitor_misc)

            nc = await NodeConfig.gather()
            if nc is not None:
                await self.info_sender.send(nc)
            sni = await StaticNodeInformation.gather()
            await self.info_sender.send(sni)

    def shutdown(self):
        self._tg.cancel_scope.cancel()

    async def _monitor_misc(self):
        if self.misc_poll_interval is None:
            return
        while True:
            await self.info_sender.send(await MiscData.gather())
            await anyio.sleep(self.misc_poll_interval)

    async def _monitor_system_profiler_thunderbolt_data(self):
        if self.system_profiler_interval is None:
            return
        iface_map = await _gather_iface_map()
        if iface_map is None:
            return

        while True:
            data = await ThunderboltConnectivity.gather()
            assert data is not None

            idents = [it for i in data if (it := i.ident(iface_map)) is not None]
            await self.info_sender.send(MacThunderboltIdentifiers(idents=idents))

            conns = [it for i in data if (it := i.conn()) is not None]
            await self.info_sender.send(MacThunderboltConnections(conns=conns))

            await anyio.sleep(self.system_profiler_interval)

    async def _monitor_memory_usage(self):
        override_memory_env = os.getenv("OVERRIDE_MEMORY_MB")
        override_memory: int | None = (
            Memory.from_mb(int(override_memory_env)).in_bytes
            if override_memory_env
            else None
        )
        if self.memory_poll_rate is None:
            return
        while True:
            await self.info_sender.send(
                MemoryUsage.from_psutil(override_memory=override_memory)
            )
            await anyio.sleep(self.memory_poll_rate)

    async def _monitor_gpu_info(self):
        if self.gpu_poll_interval is None:
            return
        prev = await NodeGpuDevices.gather()
        await self.info_sender.send(prev)
        while True:
            curr = await NodeGpuDevices.gather()
            if prev != curr:
                prev = curr
                await self.info_sender.send(curr)
            await anyio.sleep(self.gpu_poll_interval)

    async def _watch_system_info(self):
        if self.interface_watcher_interval is None:
            return
        while True:
            nics = await get_network_interfaces()
            await self.info_sender.send(NodeNetworkInterfaces(ifaces=nics))
            await anyio.sleep(self.interface_watcher_interval)

    async def _monitor_thunderbolt_bridge_status(self):
        if self.thunderbolt_bridge_poll_interval is None:
            return
        while True:
            curr = await ThunderboltBridgeInfo.gather()
            if curr is not None:
                await self.info_sender.send(curr)
            await anyio.sleep(self.thunderbolt_bridge_poll_interval)

    async def _monitor_macmon(self, macmon_path: str):
        if self.macmon_interval is None:
            return
        # macmon pipe --interval [interval in ms]
        try:
            async with await open_process(
                [macmon_path, "pipe", "--interval", str(self.macmon_interval * 1000)]
            ) as p:
                if not p.stdout:
                    logger.critical("MacMon closed stdout")
                    return
                async for text in TextReceiveStream(
                    BufferedByteReceiveStream(p.stdout)
                ):
                    await self.info_sender.send(MacmonMetrics.from_raw_json(text))
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
