"""FLASH plugin planning logic."""

from collections.abc import Mapping

# Use core types for serialization compatibility
from exo.shared.types.worker.instances import FLASHInstance
from exo.shared.types.tasks import LoadModel, Task
from exo.shared.types.worker.instances import Instance, InstanceId
from exo.shared.types.worker.runners import RunnerId, RunnerIdle
from exo.worker.runner.runner_supervisor import RunnerSupervisor


def plan_flash(
    runners: Mapping[RunnerId, RunnerSupervisor],
    instances: Mapping[InstanceId, Instance],
) -> Task | None:
    """Plan tasks specifically for FLASH instances.

    FLASH instances have a simpler lifecycle:
    - CreateRunner (handled by core _create_runner)
    - LoadModel (starts the simulation immediately)
    - Shutdown (handled by core _kill_runner)

    This function handles the LoadModel step for FLASH instances,
    skipping the MLX-specific download/init/warmup steps.
    """
    for runner in runners.values():
        instance = runner.bound_instance.instance

        # Only handle FLASH instances
        if not isinstance(instance, FLASHInstance):
            continue

        # If runner is idle, emit LoadModel to start the simulation
        if isinstance(runner.status, RunnerIdle):
            return LoadModel(instance_id=instance.instance_id)

    return None
