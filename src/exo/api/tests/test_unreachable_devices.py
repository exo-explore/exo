# pyright: reportPrivateUsage=false, reportAny=false
"""Every API checks, on its own node's membership view, that it can reach the devices a request
needs. A request whose model's instances all have a device it can't reach now ends at once
("A device serving this model can't be reached right now. Try again in a moment."), and so does a
request running on an instance whose device leaves Alive: their output can't arrive, and waiting
would last until the device is back or removed after the grace period (or, on a node cut off from
most of the cluster, until the network is back)."""

import json

import anyio
import pytest

import exo.api.main as api_main
from exo.api.tests.test_model_not_ready import (
    NODES,
    chat,
    instance,
    make_api,
    state_of,
)
from exo.api.tests.test_request_to_deleted_instance import (
    _DONE_EVENT,
    MODEL,
    NODE,
    error_message,
    instance_on,
    read_body,
    start_chat,
)
from exo.api.tests.test_request_to_deleted_instance import (
    make_api as make_streaming_api,
)
from exo.routing.tests.fake_membership import FakeMembership
from exo.shared.request_errors import DEVICE_UNREACHABLE
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.events import ChunkGenerated, IndexedEvent, TaskCreated
from exo.shared.types.state import State
from exo.shared.types.tasks import TaskStatus, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import Instance
from exo.shared.types.worker.runners import RunnerLoading, RunnerReady

ME = NodeId("api-node")


# ---- Before a request is sent ----


@pytest.mark.parametrize("stream", [True, False])
async def test_a_request_whose_model_has_a_device_it_cant_reach_ends_at_once(
    stream: bool,
) -> None:
    inst = instance()
    api, send = make_api(state_of(inst, RunnerReady(), RunnerReady()))
    view = FakeMembership(ME, *NODES)
    await view.lose(NODES[1])
    api.membership = view

    status, _, body = chat(api, stream)

    assert status == 503
    assert body["error"]["message"] == DEVICE_UNREACHABLE
    assert body["error"]["code"] == "device_unreachable"
    send.assert_not_called()


async def test_a_request_goes_ahead_while_an_instance_it_reaches_is_ready() -> None:
    unreachable, reachable = instance(), instance_on(NodeId("node-c"))
    state = state_of(unreachable, RunnerReady(), RunnerReady())
    (runner,) = reachable.shard_assignments.runner_to_shard
    api, _ = make_api(
        state.model_copy(
            update={
                "instances": {**state.instances, reachable.instance_id: reachable},
                "runners": {**state.runners, runner: RunnerReady()},
            }
        )
    )
    view = FakeMembership(ME, *NODES, NodeId("node-c"))
    await view.lose(NODES[0])
    api.membership = view

    with anyio.fail_after(5):
        assert await api._validate_model_has_instance(MODEL) == MODEL


async def test_a_ready_instance_it_cant_reach_doesnt_make_the_model_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The instance it reaches is still loading: the request waits for that one, as for any
    model that isn't ready, instead of going to the one that can't serve it"""
    monkeypatch.setattr(api_main, "MODEL_READY_WAIT_SECONDS", 0.3)
    loading = instance()
    unreachable = instance_on(NodeId("node-c"))
    state = state_of(loading, RunnerLoading(), RunnerLoading())
    (runner,) = unreachable.shard_assignments.runner_to_shard
    api, send = make_api(
        state.model_copy(
            update={
                "instances": {**state.instances, unreachable.instance_id: unreachable},
                "runners": {**state.runners, runner: RunnerReady()},
            }
        )
    )
    view = FakeMembership(ME, *NODES)
    await view.lose(NodeId("node-c"))
    api.membership = view

    status, _, body = chat(api, stream=False)

    assert status == 503
    assert body["error"]["code"] == "model_not_ready"
    send.assert_not_called()


# ---- While it runs ----


def ready_state(inst: Instance) -> State:
    (runner,) = inst.shard_assignments.runner_to_shard
    return State(instances={inst.instance_id: inst}, runners={runner: RunnerReady()})


def task_for(command_id: CommandId, inst: Instance) -> TextGeneration:
    return TextGeneration(
        instance_id=inst.instance_id,
        command_id=command_id,
        task_status=TaskStatus.Running,
        task_params=TextGenerationTaskParams(
            model=MODEL,
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        ),
    )


async def test_a_running_stream_ends_at_once_when_its_device_leaves_alive() -> None:
    inst = instance_on(NODE)
    api, events, _commands = make_streaming_api(ready_state(inst))
    view = FakeMembership(ME, NODE)
    api.membership = view
    body: list[str] = []

    async def read() -> None:
        body.extend(await read_body(await start_chat(api)))

    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:
            tg.start_soon(api._apply_state)
            tg.start_soon(api._end_requests_on_unreachable_nodes)
            tg.start_soon(read)
            while not api._text_generation_queues:
                await anyio.sleep(0.001)
            (command_id,) = api._text_generation_queues
            task = task_for(command_id, inst)
            await events.send(
                IndexedEvent(idx=0, event=TaskCreated(task_id=task.task_id, task=task))
            )
            await events.send(
                IndexedEvent(
                    idx=1,
                    event=ChunkGenerated(
                        command_id=command_id,
                        chunk=TokenChunk(
                            model=MODEL, text="Hel", token_id=1, usage=None
                        ),
                    ),
                )
            )
            while task.task_id not in api.state.tasks:
                await anyio.sleep(0.001)

            await view.lose(NODE)
            with anyio.fail_after(1):
                while not body:
                    await anyio.sleep(0.001)
            tg.cancel_scope.cancel()

    assert body[-1] == _DONE_EVENT
    # What it got before is kept
    first: dict[str, list[dict[str, dict[str, str]]]] = json.loads(
        next(event for event in body if event.startswith("data: {")).removeprefix(
            "data: "
        )
    )
    assert first["choices"][0]["delta"]["content"] == "Hel"
    assert error_message(body[-2]) == DEVICE_UNREACHABLE


async def test_a_request_placed_on_an_instance_it_cant_reach_ends_at_once() -> None:
    """The master chose the instance on its own view; this node can't reach its device (it
    became unreachable after the request was sent, before the request had a task)"""
    inst = instance_on(NODE)
    api, events, _commands = make_streaming_api(ready_state(inst))
    view = FakeMembership(ME, NODE)
    api.membership = view
    body: list[str] = []

    async def read() -> None:
        body.extend(await read_body(await start_chat(api)))

    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:
            tg.start_soon(api._apply_state)
            tg.start_soon(read)
            while not api._text_generation_queues:
                await anyio.sleep(0.001)
            (command_id,) = api._text_generation_queues
            view.set(NODE, "Suspect", "Disconnected")
            task = task_for(command_id, inst)
            await events.send(
                IndexedEvent(idx=0, event=TaskCreated(task_id=task.task_id, task=task))
            )
            with anyio.fail_after(1):
                while not body:
                    await anyio.sleep(0.001)
            tg.cancel_scope.cancel()

    assert body[-1] == _DONE_EVENT
    assert error_message(body[-2]) == DEVICE_UNREACHABLE
