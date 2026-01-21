"""FLASH plugin types - commands and instances."""
# ruff: noqa: I001 - Import order intentional for Pydantic model_rebuild

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from exo.shared.types.common import CommandId, Host, NodeId
from exo.shared.types.worker.runners import ShardAssignments
from exo.utils.pydantic_ext import TaggedModel

if TYPE_CHECKING:
    from exo.shared.types.worker.instances import InstanceId
    from exo.shared.types.worker.runners import RunnerId
    from exo.shared.types.worker.shards import (
        PipelineShardMetadata,
        TensorShardMetadata,
    )


# ============================================================================
# Commands
# ============================================================================


class LaunchFLASH(TaggedModel):
    """Command to launch a FLASH MPI simulation."""

    command_id: CommandId = Field(default_factory=CommandId)
    simulation_name: str
    flash_executable_path: str
    parameter_file_path: str
    working_directory: str
    ranks_per_node: int = 1
    min_nodes: int = 1
    # Optional: explicit hostnames for MPI (e.g., "s14,james21-1")
    # Used when topology edges don't contain IP addresses
    hosts: str = ""


class StopFLASH(TaggedModel):
    """Command to stop a running FLASH simulation."""

    command_id: CommandId = Field(default_factory=CommandId)
    instance_id: "InstanceId"


# ============================================================================
# Instances
# ============================================================================


class FLASHInstance(TaggedModel):
    """Instance for FLASH MPI simulation.

    Unlike MLX instances which do tensor parallelism, FLASH instances
    coordinate MPI processes across nodes. Each node runs one or more
    MPI ranks of the FLASH simulation.
    """

    instance_id: "InstanceId"
    shard_assignments: ShardAssignments
    hosts_by_node: dict[NodeId, list[Host]]
    flash_executable_path: str
    parameter_file_path: str
    working_directory: str
    ranks_per_node: int = 1
    total_ranks: int
    simulation_name: str
    coordinator_ip: str
    network_interface: str = "en0"  # Network interface for MPI (e.g., en0, eth0)

    def shard(
        self, runner_id: "RunnerId"
    ) -> "PipelineShardMetadata | TensorShardMetadata | None":
        return self.shard_assignments.runner_to_shard.get(runner_id, None)


# Import types into module namespace for Pydantic model_rebuild() to resolve forward refs
from exo.shared.types.worker.instances import InstanceId as InstanceId  # noqa: E402, I001
from exo.shared.types.worker.runners import RunnerId as RunnerId  # noqa: E402, I001
from exo.shared.types.worker.shards import (  # noqa: E402, I001
    PipelineShardMetadata as PipelineShardMetadata,
    TensorShardMetadata as TensorShardMetadata,
)

# Rebuild models to resolve forward references
StopFLASH.model_rebuild()
FLASHInstance.model_rebuild()
