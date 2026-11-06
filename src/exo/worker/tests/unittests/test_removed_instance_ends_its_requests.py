# pyright: reportPrivateUsage=false
"""As a worker applies an instance's removal, the requests streaming from that instance on its
node end, saying why it was removed (the removal's own reason, not a guess)."""

import anyio
import pytest

from exo.routing.tests.fake_reachability import FakeReachability
from exo.shared.models.model_cards import ModelId
from exo.shared.request_errors import generation_error_code
from exo.shared.types.chunks import Chunk, ErrorChunk
from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.events import (
    Event,
    IndexedEvent,
    InstanceCreated,
    InstanceDeleted,
)
from exo.shared.types.tasks import TaskStatus, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId, InstanceRemovalReason
from exo.shared.types.worker.runners import RunnerId
from exo.utils.channels import channel
from exo.worker.main import Worker
from exo.worker.serving import _Delivery, _Open
from exo.worker.tests.unittests.conftest import (
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)

MODEL = ModelId("test/model")


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
async def test_a_removed_instances_request_is_told_the_removals_reason(
    reason: InstanceRemovalReason, message: str, code: str
) -> None:
    node_id = NodeId()
    worker = Worker(
        node_id,
        event_receiver=channel[IndexedEvent]()[1],
        event_sender=channel[Event]()[0],
        command_sender=channel[ForwarderCommand]()[0],
        download_command_sender=channel[ForwarderDownloadCommand]()[0],
        api_port=52415,
        reachability=FakeReachability(node_id),
    )
    instance_id, runner_id = InstanceId(), RunnerId()
    instance = get_mlx_ring_instance(
        instance_id,
        MODEL,
        {node_id: runner_id},
        {runner_id: get_pipeline_shard_metadata(MODEL, 0)},
    )
    # A request streaming from the instance on this node
    task = TextGeneration(
        instance_id=instance_id,
        command_id=CommandId(),
        task_status=TaskStatus.Running,
        task_params=TextGenerationTaskParams(
            model=MODEL,
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        ),
    )
    output, stream = channel[Chunk]()
    worker.serving._open[task.task_id] = _Open(
        task=task,
        output=output,
        delivery=_Delivery(nodes=frozenset({node_id})),
    )

    events, worker.event_receiver = channel[IndexedEvent]()
    await events.send(IndexedEvent(idx=0, event=InstanceCreated(instance=instance)))
    await events.send(
        IndexedEvent(
            idx=1,
            event=InstanceDeleted(
                instance_id=instance_id, reason=reason, at="2026-11-05T12:00:00+00:00"
            ),
        )
    )
    events.close()
    with anyio.fail_after(5):
        await worker._event_applier()
        # The stream ends with the reason, then closes
        chunks = [chunk async for chunk in stream]
    (ending,) = chunks
    assert isinstance(ending, ErrorChunk)
    assert ending.error_message == message
    assert generation_error_code(ending.error_message) == code
