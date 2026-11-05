"""What the harness tells every node about the simulated network: where each node's API listens,
and which pairs of nodes are cut off from each other.

The harness rewrites the file (EXO_TESTKIT_NETWORK) on every change; nodes read it on each use.
"""

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from exo.shared.types.common import NodeId

NETWORK_TABLE_ENV = "EXO_TESTKIT_NETWORK"
# Set to 1 for nodes to probe every peer at their own API port, as exo itself does
PROBE_OWN_PORT_ENV = "EXO_TESTKIT_PROBE_OWN_PORT"

# blackhole: packets vanish, as when a cable is pulled or Wi-Fi drops (connections hang until
# a timeout notices). reset: connections are refused and torn down at once.
CutMode = Literal["blackhole", "reset"]


class Cut(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    between: tuple[NodeId, NodeId]
    mode: CutMode


class NetworkTable(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    api_ports: dict[NodeId, int] = {}
    # The address each node's peers know it by: on a real network every node has its own,
    # and exo's worker tells topology edges apart only by address (all of these are this
    # machine's loopback)
    addresses: dict[NodeId, str] = {}
    cuts: list[Cut] = []

    def cut_between(self, a: NodeId, b: NodeId) -> CutMode | None:
        for cut in self.cuts:
            if set(cut.between) == {a, b}:
                return cut.mode
        return None


def read_table(path: Path | None = None) -> NetworkTable:
    if path is None:
        env = os.environ.get(NETWORK_TABLE_ENV)
        if not env:
            return NetworkTable()
        path = Path(env)
    try:
        return NetworkTable.model_validate_json(path.read_bytes())
    except (FileNotFoundError, ValidationError):
        # Mid-replace or not written yet: nothing known
        return NetworkTable()


def write_table(path: Path, table: NetworkTable) -> None:
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(table.model_dump_json())
    os.replace(temporary, path)
