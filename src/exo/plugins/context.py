"""Context objects passed to plugin handlers."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from exo.shared.types.commands import BaseCommand
from exo.shared.types.common import NodeId
from exo.shared.types.state import State


@dataclass
class PluginContext:
    """Context provided to plugin API handlers.

    This gives plugins access to the current state and the ability to send
    commands without direct access to internal API components.
    """

    state: State
    send_command: Callable[[BaseCommand], Awaitable[None]]
    node_id: NodeId
