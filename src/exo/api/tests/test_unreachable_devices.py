# pyright: reportPrivateUsage=false, reportAny=false
"""Every API checks, on its own node's membership view, that it can reach the devices a request
needs. A request whose model's instances all have a device it can't reach now ends at once
("A device serving this model can't be reached right now. Try again in a moment."): its output
can't arrive, and waiting would last until the device is back or removed after the grace period
(or, on a node cut off from most of the cluster, until the network is back). A request running
on an instance whose device leaves Alive ends at once too, with the same message: see
exo.api.tests.test_serving_client and exo.worker.tests.unittests.test_serving."""

import contextlib
from collections.abc import AsyncIterator

import anyio
import pytest

import exo.api.main as api_main
from exo.api.main import API
from exo.api.serving_client import ServeStream, ServingClient
from exo.api.tests.test_model_not_ready import (
    MODEL,
    NODES,
    chat,
    instance,
    make_api,
    state_of,
)
from exo.routing.tests.fake_membership import FakeMembership
from exo.shared.request_errors import DEVICE_UNREACHABLE
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.profiling import NetworkInterfaceInfo, NodeNetworkInfo
from exo.shared.types.serving import Accepted, ServeFrame, ServeRequest
from exo.shared.types.state import State
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import Instance, InstanceId
from exo.shared.types.worker.runners import RunnerId, RunnerLoading, RunnerReady
from exo.worker.tests.unittests.conftest import (
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)

ME = NodeId("api-node")


def instance_on(node: NodeId) -> Instance:
    """A one-node instance of the model"""
    runner = RunnerId()
    return get_mlx_ring_instance(
        instance_id=InstanceId(),
        model_id=MODEL,
        node_to_runner={node: runner},
        runner_to_shard={runner: get_pipeline_shard_metadata(MODEL, device_rank=0)},
    )


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


# ---- Which instance a request goes to ----


async def test_a_request_goes_to_the_instance_it_reaches_without_trying_the_other() -> (
    None
):
    """Two ready instances of the model, one on a device this node can't reach now. The check
    of the model as a whole passes (one instance is reachable), and the request goes to that
    one, every time, without trying the other: a frozen or asleep device accepts the connection
    and never answers, so a request sent there waited for the 30 s bound before moving on."""
    reachable_node, unreachable_node = NODES
    on_reachable, on_unreachable = (
        instance_on(reachable_node),
        instance_on(unreachable_node),
    )
    state = State(
        instances={
            on_reachable.instance_id: on_reachable,
            on_unreachable.instance_id: on_unreachable,
        },
        runners={
            runner: RunnerReady()
            for instance in (on_reachable, on_unreachable)
            for runner in instance.shard_assignments.runner_to_shard
        },
        node_network={
            node: NodeNetworkInfo(
                interfaces=[NetworkInterfaceInfo(name="en0", ip_address=address)]
            )
            for node, address in (
                (reachable_node, "192.0.2.1"),
                (unreachable_node, "192.0.2.2"),
            )
        },
    )
    tried: list[NodeId] = []

    @contextlib.asynccontextmanager
    async def frames_from(
        request: ServeRequest,
    ) -> AsyncIterator[AsyncIterator[ServeFrame | None]]:
        tried.append(request.target)

        async def frames() -> AsyncIterator[ServeFrame | None]:
            yield Accepted()

        yield frames()

    api = object.__new__(API)
    api.state = state
    view = FakeMembership(ME, *NODES)
    await view.lose(unreachable_node)
    api.membership = view
    api._serving = ServingClient(
        ME,
        api_port=52415,
        host=None,
        state=lambda: state,
        reachability=view,
        frames_from=frames_from,
    )
    params = TextGenerationTaskParams(
        model=MODEL,
        input=[InputMessage(role="user", content=InputMessageContent("hi"))],
    )
    async with anyio.create_task_group() as tg:
        tg.start_soon(api._serving.run)
        await anyio.sleep(0.05)  # the client's task group is up
        with anyio.fail_after(5):
            for _ in range(20):
                stream = await api._try_instances(params, CommandId())
                assert isinstance(stream, ServeStream)
                stream.close()
        tg.cancel_scope.cancel()
    assert tried == [reachable_node] * 20
