"""The runner process's entrypoint under the test kit: exo's own bootstrap, with the stub engine.

`exo.testkit.node` points exo's runner supervisor at this function. It runs in the runner
process (spawned, so it imports this module by name), puts the stub in place of the two MLX
modules exo's bootstrap imports, then calls exo's real bootstrap entrypoint. Everything else
the runner does, its signal and error handling included, is exo's own code.
"""

import sys
import time
import types

import loguru

from exo.shared.types.events import Event
from exo.shared.types.tasks import Task, TaskId
from exo.shared.types.worker.instances import BoundInstance
from exo.testkit.stub.config import read_config
from exo.testkit.stub.engine import StubBuilder
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.runner import bootstrap
from exo.worker.runner.bootstrap import RunnerTerminationError


def _no_patches() -> None:
    pass


def install_stub_engine() -> None:
    """Make `from exo.worker.engines.mlx.builder import MlxBuilder` give the stub builder, and
    `apply_mlx_patches` do nothing, in this process (neither may import MLX here)."""
    patches = types.ModuleType("exo.worker.engines.mlx.patches")
    patches.__dict__["apply_mlx_patches"] = _no_patches
    builder = types.ModuleType("exo.worker.engines.mlx.builder")
    builder.__dict__["MlxBuilder"] = StubBuilder
    sys.modules[patches.__name__] = patches
    sys.modules[builder.__name__] = builder


def entrypoint(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event | RunnerTerminationError],
    task_receiver: MpReceiver[Task],
    cancel_receiver: MpReceiver[TaskId],
    _logger: "loguru.Logger",
) -> None:
    install_stub_engine()
    time.sleep(read_config().start_seconds)
    bootstrap.entrypoint(
        bound_instance, event_sender, task_receiver, cancel_receiver, _logger
    )
