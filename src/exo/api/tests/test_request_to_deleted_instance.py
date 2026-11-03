# pyright: reportPrivateUsage=false, reportAny=false
"""A chat request sent just as its model's instance is deleted ends promptly with an error.

The master used to give such a request a task on the deleted instance when it handled the
request a few milliseconds after the deletion, before its own state had caught up. The API
ends an instance's requests when it applies the deletion, which came before that task, so
the request's stream got nothing but keepalives until the client gave up. The master now
applies its own events at once; the API still ends a request whose task wasn't added, which
a master of an older version can still send.
"""

import json
from datetime import datetime, timezone

import anyio
from fastapi.responses import StreamingResponse

from exo.api.main import API
from exo.api.recent_events import RecentEvents
from exo.api.types import ChatCompletionMessage, ChatCompletionRequest
from exo.master.main import Master
from exo.routing.router import get_node_zid
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.commands import (
    DeleteInstance,
    ForwarderCommand,
    ForwarderDownloadCommand,
)
from exo.shared.types.common import NodeId, SessionId, SystemId
from exo.shared.types.events import (
    GlobalForwarderEvent,
    IndexedEvent,
    InstanceCreated,
    InstanceDeleted,
    LocalForwarderEvent,
    RunnerStatusUpdated,
    StateSnapshot,
    TaskCreated,
)
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.tasks import TaskStatus, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import RunnerId, RunnerReady, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.task_group import TaskGroup

MODEL = ModelId("test-model")
NODE = NodeId("node")
_DONE_EVENT = "data: [DONE]\n\n"


def instance_on(node: NodeId) -> MlxRingInstance:
    card = ModelCard(
        model_id=MODEL,
        storage_size=Memory.from_mb(100),
        n_layers=1,
        hidden_size=64,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )
    runner = RunnerId()
    return MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=MODEL,
            runner_to_shard={
                runner: PipelineShardMetadata(
                    model_card=card,
                    device_rank=0,
                    world_size=1,
                    start_layer=0,
                    end_layer=1,
                    n_layers=1,
                )
            },
            node_to_runner={node: runner},
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )


def make_api(
    state: State,
) -> tuple[API, Sender[IndexedEvent], Receiver[ForwarderCommand]]:
    api = object.__new__(API)
    api.state = state
    api.paused = False
    api._system_id = SystemId()
    api._recent_events = RecentEvents()
    api._tg = TaskGroup()
    api._text_generation_queues = {}
    api._image_generation_queues = {}
    api._cancelled_command_ids = set()
    events_sender, api.event_receiver = channel[IndexedEvent]()
    api.command_sender, commands = channel[ForwarderCommand]()
    return api, events_sender, commands


async def start_chat(api: API) -> StreamingResponse:
    response = await api.chat_completions(
        ChatCompletionRequest(
            model=MODEL,
            messages=[ChatCompletionMessage(role="user", content="Count to ten.")],
            stream=True,
        )
    )
    assert isinstance(response, StreamingResponse)
    return response


async def read_body(response: StreamingResponse) -> list[str]:
    body: list[str] = []
    async for event in response.body_iterator:
        assert isinstance(event, str)
        body.append(event)
    return body


def error_message(sse_event: str) -> str:
    payload: dict[str, dict[str, str]] = json.loads(sse_event.removeprefix("data: "))
    return payload["error"]["message"]


async def test_a_request_whose_task_was_not_added_ends_with_an_error() -> None:
    """Whatever sends a request to a deleted instance, the task isn't added, and the API
    ends the request instead of waiting for output that will never come."""
    instance = instance_on(NODE)
    api, events, _commands = make_api(State())
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
            while not api._text_generation_queues:
                await anyio.sleep(0.001)
            (command_id,) = api._text_generation_queues
            await events.send(
                IndexedEvent(
                    idx=2, event=InstanceDeleted(instance_id=instance.instance_id)
                )
            )
            task = TextGeneration(
                instance_id=instance.instance_id,
                command_id=command_id,
                task_status=TaskStatus.Pending,
                task_params=TextGenerationTaskParams(
                    model=MODEL,
                    input=[
                        InputMessage(role="user", content=InputMessageContent("hi"))
                    ],
                ),
            )
            await events.send(
                IndexedEvent(idx=3, event=TaskCreated(task_id=task.task_id, task=task))
            )
            while not body:
                await anyio.sleep(0.001)
            assert task.task_id not in api.state.tasks
            tg.cancel_scope.cancel()

    assert body[-1] == _DONE_EVENT
    assert error_message(body[-2]) == (
        "The model instance was shut down before this request could start."
    )


async def test_a_chat_sent_just_after_deleting_its_instance_ends_promptly() -> None:
    """The whole path: the API sends DeleteInstance and then a chat request, the master
    handles both back to back, and the API applies what the master indexed. The request must
    end with an error, not hang.

    DELETE /instance answers only once the deletion has applied, so a chat from the same
    client comes too late for this race. A chat from another client, sent while the deletion
    is on its way, still meets it: the DeleteInstance is sent here without waiting for it."""
    instance = instance_on(NODE)
    topology = Topology()
    topology.add_node(NODE)
    # Ready to serve, as the API checks before sending a request
    initial = State(
        topology=topology,
        last_seen={NODE: datetime.now(tz=timezone.utc)},
        instances={instance.instance_id: instance},
        runners={
            runner: RunnerReady()
            for runner in instance.shard_assignments.runner_to_shard
        },
    )
    api, api_events, commands = make_api(initial)

    node_id = get_node_zid()
    session = SessionId(master_node_id=node_id, election_clock=0)
    _local_events_sender, local_events = channel[LocalForwarderEvent]()
    global_events_sender, global_events = channel[GlobalForwarderEvent]()
    master = Master(
        node_id,
        session,
        command_receiver=commands,
        local_event_receiver=local_events,
        global_event_sender=global_events_sender,
        snapshot_sender=channel[StateSnapshot]()[0],
        download_command_sender=channel[ForwarderDownloadCommand]()[0],
        initial_state=initial,
    )

    async def to_api() -> None:
        with global_events as indexed:
            async for message in indexed:
                await api_events.send(
                    IndexedEvent(idx=message.origin_idx, event=message.event)
                )

    body: list[str] = []
    with anyio.fail_after(10):
        async with anyio.create_task_group() as tg:
            tg.start_soon(master._command_processor)
            tg.start_soon(master._event_processor)
            tg.start_soon(to_api)
            tg.start_soon(api._apply_state)

            await api._send(DeleteInstance(instance_id=instance.instance_id))
            body = await read_body(await start_chat(api))
            tg.cancel_scope.cancel()

    assert body[-1] == _DONE_EVENT
    assert error_message(body[-2]) == (
        "The model instance was shut down before this request could start."
    )
    assert instance.instance_id not in api.state.instances
    assert not api.state.tasks
