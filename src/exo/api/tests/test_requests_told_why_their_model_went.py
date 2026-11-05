# pyright: reportPrivateUsage=false, reportAny=false
"""A request whose model's instance is removed is told why, from the removal itself.

The API used to guess: a node missing from the topology meant a device left, and anything else
meant the model was stopped. A model whose runner kept failing to start has all its devices still
in the cluster, so its requests were told "The model was stopped before the reply finished": a
user stop that nobody made.
"""

import anyio
import pytest

from exo.api.tests.test_request_to_deleted_instance import (
    MODEL,
    NODE,
    error_message,
    instance_on,
    make_api,
    read_body,
    start_chat,
)
from exo.shared.request_errors import generation_error_code
from exo.shared.topology import Topology
from exo.shared.types.events import (
    IndexedEvent,
    InstanceCreated,
    InstanceDeleted,
    RunnerStatusUpdated,
    TaskCreated,
)
from exo.shared.types.state import State
from exo.shared.types.tasks import TaskStatus, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceRemovalReason
from exo.shared.types.worker.runners import RunnerReady


@pytest.mark.parametrize(
    ("reason", "message", "code"),
    [
        (
            "NodeLost",
            "A device serving this model left the cluster before the reply finished.",
            "device_left",
        ),
        (
            "NodeLeft",
            "A device serving this model left the cluster before the reply finished.",
            "device_left",
        ),
        (
            "UserStopped",
            "The model was stopped before the reply finished.",
            "model_stopped",
        ),
        (
            "StartFailed",
            "The model stopped unexpectedly before the reply finished.",
            "model_failed",
        ),
    ],
)
async def test_a_request_is_told_why_its_model_went(
    reason: InstanceRemovalReason, message: str, code: str
) -> None:
    instance = instance_on(NODE)
    # The device is still in the cluster, whatever the reason
    topology = Topology()
    topology.add_node(NODE)
    api, events, _commands = make_api(State(topology=topology))
    body: list[str] = []

    async def read() -> None:
        body.extend(await read_body(await start_chat(api)))

    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:
            tg.start_soon(api._apply_state)
            await events.send(
                IndexedEvent(idx=0, event=InstanceCreated(instance=instance))
            )
            (runner,) = instance.shard_assignments.runner_to_shard
            await events.send(
                IndexedEvent(
                    idx=1,
                    event=RunnerStatusUpdated(
                        runner_id=runner, runner_status=RunnerReady()
                    ),
                )
            )
            while runner not in api.state.runners:
                await anyio.sleep(0.001)
            tg.start_soon(read)
            # The request's task is running on the instance
            while not api._text_generation_queues:
                await anyio.sleep(0.001)
            (command_id,) = api._text_generation_queues
            task = TextGeneration(
                instance_id=instance.instance_id,
                command_id=command_id,
                task_status=TaskStatus.Running,
                task_params=TextGenerationTaskParams(
                    model=MODEL,
                    input=[
                        InputMessage(role="user", content=InputMessageContent("hi"))
                    ],
                ),
            )
            await events.send(
                IndexedEvent(idx=2, event=TaskCreated(task_id=task.task_id, task=task))
            )
            await events.send(
                IndexedEvent(
                    idx=3,
                    event=InstanceDeleted(
                        instance_id=instance.instance_id,
                        reason=reason,
                        at="2026-11-03T12:00:00+00:00",
                    ),
                )
            )
            while not body:
                await anyio.sleep(0.001)
            tg.cancel_scope.cancel()

    assert error_message(body[-2]) == message
    assert generation_error_code(error_message(body[-2])) == code
