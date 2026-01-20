"""FLASH plugin API handlers."""

from typing import Any

from fastapi import HTTPException

from exo.plugins.context import PluginContext

# Use core types for serialization compatibility
from exo.shared.types.commands import LaunchFLASH, StopFLASH
from exo.shared.types.worker.instances import FLASHInstance


async def handle_launch_flash(
    ctx: PluginContext,
    simulation_name: str,
    flash_executable_path: str,
    working_directory: str,
    parameter_file_path: str = "",
    ranks_per_node: int = 1,
    min_nodes: int = 1,
    hosts: str = "",
) -> dict[str, str]:
    """Launch a FLASH MPI simulation across the cluster.

    Args:
        ctx: Plugin context with state and send_command
        simulation_name: Name of the simulation
        flash_executable_path: Path to the FLASH executable
        working_directory: Working directory for the simulation
        parameter_file_path: Path to parameter file (optional)
        ranks_per_node: Number of MPI ranks per node
        min_nodes: Minimum number of nodes required
        hosts: Optional comma-separated hostnames (e.g., "s14,james21-1").
               If not provided, IPs are discovered from topology edges.
    """
    command = LaunchFLASH(
        simulation_name=simulation_name,
        flash_executable_path=flash_executable_path,
        parameter_file_path=parameter_file_path,
        working_directory=working_directory,
        ranks_per_node=ranks_per_node,
        min_nodes=min_nodes,
        hosts=hosts,
    )
    await ctx.send_command(command)

    return {
        "message": "FLASH launch command received",
        "command_id": str(command.command_id),
        "simulation_name": simulation_name,
    }


async def handle_stop_flash(
    ctx: PluginContext,
    instance_id: str,
) -> dict[str, str]:
    """Stop a running FLASH simulation."""
    from exo.shared.types.worker.instances import InstanceId

    inst_id = InstanceId(instance_id)

    if inst_id not in ctx.state.instances:
        raise HTTPException(status_code=404, detail="Instance not found")

    instance = ctx.state.instances[inst_id]
    if not isinstance(instance, FLASHInstance):
        raise HTTPException(
            status_code=400, detail="Instance is not a FLASH simulation"
        )

    command = StopFLASH(instance_id=inst_id)
    await ctx.send_command(command)

    return {
        "message": "Stop command received",
        "command_id": str(command.command_id),
        "instance_id": str(instance_id),
    }


async def handle_list_flash_instances(ctx: PluginContext) -> list[dict[str, Any]]:
    """List all FLASH simulation instances."""
    flash_instances: list[dict[str, Any]] = []
    for instance_id, instance in ctx.state.instances.items():
        if isinstance(instance, FLASHInstance):
            # Get runner statuses for this instance
            runner_statuses: dict[str, str | None] = {}
            for (
                node_id,
                runner_id,
            ) in instance.shard_assignments.node_to_runner.items():
                runner_status = ctx.state.runners.get(runner_id)
                runner_statuses[str(node_id)] = (
                    str(runner_status) if runner_status else None
                )

            flash_instances.append(
                {
                    "instance_id": str(instance_id),
                    "simulation_name": instance.simulation_name,
                    "total_ranks": instance.total_ranks,
                    "working_directory": instance.working_directory,
                    "runner_statuses": runner_statuses,
                }
            )
    return flash_instances
