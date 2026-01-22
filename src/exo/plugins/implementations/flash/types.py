"""FLASH plugin types - commands and instances."""

from exo.plugins.type_registry import command_registry, instance_registry
from exo.shared.types.commands import BaseCommand
from exo.shared.types.common import Host, NodeId
from exo.shared.types.worker.instances import BaseInstance, InstanceId
from exo.shared.types.worker.runners import RunnerId
from exo.shared.types.worker.shards import ShardMetadata

# ============================================================================
# Commands
# ============================================================================


@command_registry.register
class LaunchFLASH(BaseCommand):
    """Command to launch a FLASH MPI simulation."""

    simulation_name: str
    flash_executable_path: str
    parameter_file_path: str
    working_directory: str
    ranks_per_node: int = 1
    min_nodes: int = 1
    # Optional: explicit hostnames for MPI (e.g., "s14,james21-1")
    # Used when topology edges don't contain IP addresses
    hosts: str = ""


@command_registry.register
class StopFLASH(BaseCommand):
    """Command to stop a running FLASH simulation."""

    instance_id: InstanceId


# ============================================================================
# Instances
# ============================================================================


@instance_registry.register
class FLASHInstance(BaseInstance):
    """Instance for FLASH MPI simulation.

    Unlike MLX instances which do tensor parallelism, FLASH instances
    coordinate MPI processes across nodes. Each node runs one or more
    MPI ranks of the FLASH simulation.
    """

    hosts_by_node: dict[NodeId, list[Host]]
    flash_executable_path: str
    parameter_file_path: str
    working_directory: str
    ranks_per_node: int = 1
    total_ranks: int
    simulation_name: str
    coordinator_ip: str
    network_interface: str = "en0"  # Network interface for MPI (e.g., en0, eth0)

    def shard(self, runner_id: RunnerId) -> ShardMetadata | None:
        return self.shard_assignments.runner_to_shard.get(runner_id, None)
