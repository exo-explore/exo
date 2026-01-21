"""FLASH Plugin - Main plugin class."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from exo.plugins.base import EXOPlugin
from exo.plugins.implementations.flash.api_handlers import (
    handle_launch_flash,
    handle_list_flash_instances,
    handle_stop_flash,
)
from exo.plugins.implementations.flash.placement import place_flash_instance
from exo.plugins.implementations.flash.planning import plan_flash
from exo.plugins.implementations.flash.runner import main as flash_runner_main
from exo.plugins.implementations.flash.types import (
    FLASHInstance,
    LaunchFLASH,
    StopFLASH,
)
from exo.shared.topology import Topology
from exo.shared.types.commands import DeleteInstance
from exo.shared.types.events import Event
from exo.shared.types.tasks import Task
from exo.shared.types.worker.instances import BoundInstance, Instance, InstanceId
from exo.shared.types.worker.runners import RunnerId
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.runner.runner_supervisor import RunnerSupervisor


class FLASHPlugin(EXOPlugin):
    """Plugin for FLASH MPI simulations."""

    @property
    def name(self) -> str:
        return "flash"

    @property
    def version(self) -> str:
        return "1.0.0"

    def get_command_types(self) -> Sequence[type]:
        return [LaunchFLASH, StopFLASH]

    def get_instance_type(self) -> type:
        return FLASHInstance

    def get_api_routes(
        self,
    ) -> Sequence[tuple[str, str, Callable[..., Any]]]:
        return [
            ("post", "/flash/launch", handle_launch_flash),
            ("delete", "/flash/{instance_id}", handle_stop_flash),
            ("get", "/flash/instances", handle_list_flash_instances),
        ]

    def handles_command(self, command: Any) -> bool:  # pyright: ignore[reportAny]
        return isinstance(command, (LaunchFLASH, StopFLASH))

    def process_command(
        self,
        command: Any,  # pyright: ignore[reportAny]
        topology: Topology,
        current_instances: Mapping[InstanceId, Instance],
    ) -> Sequence[Event]:
        from exo.master.placement import delete_instance, get_transition_events

        if isinstance(command, LaunchFLASH):
            placement = place_flash_instance(command, topology, current_instances)
            return list(get_transition_events(current_instances, placement))
        elif isinstance(command, StopFLASH):
            placement = delete_instance(
                DeleteInstance(instance_id=command.instance_id),
                current_instances,
            )
            return list(get_transition_events(current_instances, placement))
        return []

    def handles_instance(self, instance: object) -> bool:
        return isinstance(instance, FLASHInstance)

    def plan_task(
        self,
        runners: Mapping[RunnerId, RunnerSupervisor],
        instances: Mapping[InstanceId, Instance],
    ) -> Task | None:
        return plan_flash(runners, instances)

    def should_skip_download(self, instance: object) -> bool:
        # FLASH instances don't need model downloads
        return True

    def create_runner(
        self,
        bound_instance: BoundInstance,
        event_sender: MpSender[Event],
        task_receiver: MpReceiver[Task],
    ) -> None:
        flash_runner_main(bound_instance, event_sender, task_receiver)
