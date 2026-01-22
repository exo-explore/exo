"""Base classes and protocols for Exo plugins."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import Field

from exo.shared.types.common import CommandId
from exo.shared.types.events import Event
from exo.shared.types.tasks import Task
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runners import RunnerId
from exo.utils.pydantic_ext import TaggedModel

if TYPE_CHECKING:
    from exo.shared.topology import Topology
    from exo.shared.types.worker.instances import BaseInstance, BoundInstance
    from exo.utils.channels import MpReceiver, MpSender
    from exo.worker.runner.runner_supervisor import RunnerSupervisor


class PluginCommand(TaggedModel):
    """Base class for plugin-defined commands.

    All plugin commands must inherit from this class. Commands are serialized
    with their class name as a tag for routing.
    """

    command_id: CommandId = Field(default_factory=CommandId)


class PluginInstance(TaggedModel):
    """Base class for plugin-defined instances.

    All plugin instances must inherit from this class. Plugins are expected
    to define their own instance type with workload-specific fields.
    """

    instance_id: InstanceId


class EXOPlugin(ABC):
    """Protocol that all exo plugins must implement.

    A plugin provides:
    - Custom command types for API -> Master communication
    - Custom instance types representing running workloads
    - Placement logic for distributing work across nodes
    - Planning logic for local task scheduling
    - Runner implementation for executing work
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this plugin (e.g., 'flash', 'pytorch', 'mpi')."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Semantic version string (e.g., '1.0.0')."""
        ...

    # ========== Type Registration ==========

    @abstractmethod
    def get_command_types(self) -> Sequence[type]:
        """Return command types this plugin handles.

        These commands are routed to this plugin's process_command method.
        Can return core BaseCommand types or PluginCommand types.
        """
        ...

    @abstractmethod
    def get_instance_type(self) -> type:
        """Return the instance type this plugin creates.

        This instance type is used for routing in planning and runner bootstrap.
        Can return core Instance types or PluginInstance types.
        """
        ...

    # ========== API Routes ==========

    @abstractmethod
    def get_api_routes(
        self,
    ) -> Sequence[tuple[str, str, Callable[..., Any]]]:
        """Return FastAPI routes to register.

        Each tuple: (method, path, handler)
        Example: [('post', '/flash/launch', self.launch_handler)]

        Handlers receive a PluginContext with access to:
        - state: Current State object
        - send_command: Async function to send commands
        - node_id: Current node's ID
        """
        ...

    # ========== Master Command Handling ==========

    @abstractmethod
    def handles_command(self, command: Any) -> bool:  # pyright: ignore[reportAny]
        """Return True if this plugin handles the given command type."""
        ...

    @abstractmethod
    def process_command(
        self,
        command: Any,  # pyright: ignore[reportAny]
        topology: "Topology",
        current_instances: Mapping[InstanceId, "BaseInstance"],
    ) -> Sequence[Event]:
        """Process a command and return events to emit.

        Typically creates placement and returns InstanceCreated/InstanceDeleted events.

        Args:
            command: The command to process
            topology: Current cluster topology
            current_instances: Currently running instances

        Returns:
            Sequence of events to emit (e.g., InstanceCreated, InstanceDeleted)
        """
        ...

    # ========== Worker Planning ==========

    @abstractmethod
    def handles_instance(self, instance: object) -> bool:
        """Return True if this plugin manages the given instance type."""
        ...

    @abstractmethod
    def plan_task(
        self,
        runners: Mapping[RunnerId, "RunnerSupervisor"],
        instances: Mapping[InstanceId, "BaseInstance"],
    ) -> Task | None:
        """Plan the next task for plugin instances.

        Called during each planning cycle.
        Return None if no task is needed.
        """
        ...

    @abstractmethod
    def should_skip_download(self, instance: object) -> bool:
        """Return True if this instance type doesn't need model downloads."""
        ...

    # ========== Runner Bootstrap ==========

    @abstractmethod
    def create_runner(
        self,
        bound_instance: "BoundInstance",
        event_sender: "MpSender[Event]",
        task_receiver: "MpReceiver[Task]",
    ) -> None:
        """Entry point for the runner process.

        Called in a subprocess to execute the actual workload.
        This function should block until the workload completes.
        """
        ...
