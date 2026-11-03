# pyright: reportPrivateUsage=false
"""review2: with one ready and one downloading instance of a model, the API lets a request
through (the model is "ready") but the master sends it to the downloading instance, because it
has fewer requests in flight. The request then waits for the whole download: the hang the 503
fix was meant to end."""

from datetime import datetime, timezone

import anyio

from exo.master.main import Master
from exo.routing.router import get_node_zid
from exo.shared.model_readiness import model_readiness
from exo.shared.models.model_cards import ModelId
from exo.shared.topology import Topology
from exo.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
    TextGeneration,
)
from exo.shared.types.common import CommandId, NodeId, SessionId, SystemId
from exo.shared.types.events import (
    GlobalForwarderEvent,
    LocalForwarderEvent,
    StateSnapshot,
    TaskCreated,
)
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.tasks import TaskStatus
from exo.shared.types.tasks import TextGeneration as TextGenerationTask
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.downloads import DownloadOngoing, DownloadProgressData
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runners import RunnerId, RunnerIdle, RunnerRunning
from exo.utils.channels import channel
from exo.worker.tests.unittests.conftest import (
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)

MODEL = ModelId("test-model")
READY_NODE, NEW_NODE = NodeId("ready-node"), NodeId("new-node")


def _instance(node: NodeId, runner: RunnerId):
    return get_mlx_ring_instance(
        instance_id=InstanceId(),
        model_id=MODEL,
        node_to_runner={node: runner},
        runner_to_shard={runner: get_pipeline_shard_metadata(MODEL, device_rank=0)},
    )


def _params() -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=MODEL,
        input=[InputMessage(role="user", content=InputMessageContent("hi"))],
    )


async def test_request_goes_to_the_downloading_instance_although_one_is_ready() -> None:
    ready_runner, new_runner = RunnerId(), RunnerId()
    ready, downloading = (
        _instance(READY_NODE, ready_runner),
        _instance(NEW_NODE, new_runner),
    )
    busy = TextGenerationTask(
        instance_id=ready.instance_id,
        command_id=CommandId(),
        task_status=TaskStatus.Running,
        task_params=_params(),
    )
    shard = downloading.shard_assignments.runner_to_shard[new_runner]
    topology = Topology()
    topology.add_node(READY_NODE)
    topology.add_node(NEW_NODE)
    now = datetime.now(tz=timezone.utc)
    state = State(
        topology=topology,
        last_seen={READY_NODE: now, NEW_NODE: now},
        instances={ready.instance_id: ready, downloading.instance_id: downloading},
        runners={ready_runner: RunnerRunning(), new_runner: RunnerIdle()},
        tasks={busy.task_id: busy},
        downloads={
            NEW_NODE: [
                DownloadOngoing(
                    node_id=NEW_NODE,
                    shard_metadata=shard,
                    download_progress=DownloadProgressData(
                        total=Memory.from_bytes(10 * 1024**3),
                        downloaded=Memory.from_bytes(1024**3),
                        downloaded_this_session=Memory.from_bytes(1024**3),
                        completed_files=0,
                        total_files=4,
                        speed=1.0,
                        eta_ms=600_000,
                        files={},
                    ),
                )
            ]
        },
    )

    # The API's check: the model has a ready instance, so no 503
    assert model_readiness(state, MODEL) is None

    node_id = get_node_zid()
    global_event_sender, broadcast = channel[GlobalForwarderEvent]()
    commands_sender, commands = channel[ForwarderCommand]()
    master = Master(
        node_id,
        SessionId(master_node_id=node_id, election_clock=0),
        command_receiver=commands,
        local_event_receiver=channel[LocalForwarderEvent]()[1],
        global_event_sender=global_event_sender,
        snapshot_sender=channel[StateSnapshot]()[0],
        download_command_sender=channel[ForwarderDownloadCommand]()[0],
        initial_state=state,
    )
    # _without_requests drops generation tasks from an initial state; put the busy one back
    master.state = state
    created: TaskCreated | None = None
    async with anyio.create_task_group() as tg:
        tg.start_soon(master._command_processor)
        await commands_sender.send(
            ForwarderCommand(
                origin=SystemId(), command=TextGeneration(task_params=_params())
            )
        )
        with anyio.fail_after(5):
            while created is None:
                event = (await broadcast.receive()).event
                if isinstance(event, TaskCreated):
                    created = event
        tg.cancel_scope.cancel()

    # Desired: the ready instance. Actual: the downloading one.
    assert created is not None
    assert created.task.instance_id == ready.instance_id, (
        "request routed to the instance that is still downloading"
    )
