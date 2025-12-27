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
)
from exo.shared.types.thunderbolt import TBConnection, TBConnectivity, TBIdentifier
from exo.utils.channels import Sender
from exo.utils.pydantic_ext import TaggedModel

from .macmon import MacmonMetrics
from .system_info import get_friendly_name, get_model_and_chip, get_network_interfaces

IS_DARWIN = sys.platform == "darwin"


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


class MacTBIdentifiers(TaggedModel):
    idents: Sequence[TBIdentifier]


class MacTBConnections(TaggedModel):
    conns: Sequence[TBConnection]


class NodeConfig(TaggedModel):
    """Node configuration from EXO_CONFIG_FILE, reloaded from the file only at startup. Other changes should come in through the API and propagate from there"""

    # TODO
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
    | MacTBIdentifiers
    | MacTBConnections
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
    _tg: TaskGroup = field(init=False, default_factory=create_task_group)

    async def run(self):
        async with self._tg as tg:
            if (macmon_path := shutil.which("macmon")) is not None:
                tg.start_soon(self._monitor_macmon, macmon_path)
            if IS_DARWIN:
                tg.start_soon(self._monitor_system_profiler)
            tg.start_soon(self._watch_system_info)
            tg.start_soon(self._monitor_memory_usage)
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
        prev = await MiscData.gather()
        while True:
            curr = await MiscData.gather()
            if prev != curr:
                prev = curr
                await self.info_sender.send(curr)
            await anyio.sleep(self.misc_poll_interval)

    async def _monitor_system_profiler(self):
        if self.system_profiler_interval is None:
            return
        iface_map = await _gather_iface_map()
        if iface_map is None:
            return

        old_idents = []
        while True:
            data = await TBConnectivity.gather()
            assert data is not None

            idents = [it for i in data if (it := i.ident(iface_map)) is not None]
            if idents != old_idents:
                await self.info_sender.send(MacTBIdentifiers(idents=idents))
            old_idents = idents

            conns = [it for i in data if (it := i.conn()) is not None]
            await self.info_sender.send(MacTBConnections(conns=conns))

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

    async def _watch_system_info(self):
        if self.interface_watcher_interval is None:
            return
        old_nics = []
        while True:
            nics = get_network_interfaces()
            if nics != old_nics:
                old_nics = nics
                await self.info_sender.send(NodeNetworkInterfaces(ifaces=nics))
            await anyio.sleep(self.interface_watcher_interval)

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
