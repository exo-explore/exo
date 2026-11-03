# pyright: reportPrivateUsage=false, reportAny=false
"""A request for a model that can't serve yet gets a 503 after a bounded wait, not an endless one.

Requests were sent whatever the state of the model's instance, and waited (with keepalives, or
with nothing for a whole response) until it was ready: for a whole download, or forever when
the download never progressed.
"""

import time
from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import Any
from unittest.mock import AsyncMock

import anyio
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import exo.api.main as api_main
from exo.api.adapters.chat_completions import collect_chat_response
from exo.api.collected_response import CollectedResponse
from exo.api.main import API
from exo.routing.tests.fake_membership import FakeMembership
from exo.shared.model_readiness import NotReady, model_readiness
from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.common import CommandId, ModelId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
    DownloadProgress,
    DownloadProgressData,
)
from exo.shared.types.worker.instances import Instance, InstanceId
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerIdle,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerStatus,
    RunnerWarmingUp,
)
from exo.worker.tests.unittests.conftest import (
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)

MODEL = ModelId("test-model")
NODES = (NodeId("node-a"), NodeId("node-b"))
GB = 1024**3


def instance() -> Instance:
    runners = {node: RunnerId() for node in NODES}
    return get_mlx_ring_instance(
        instance_id=InstanceId(),
        model_id=MODEL,
        node_to_runner=runners,
        runner_to_shard={
            runner: get_pipeline_shard_metadata(MODEL, device_rank=rank, world_size=2)
            for rank, runner in enumerate(runners.values())
        },
    )


def state_of(
    inst: Instance,
    *runners: RunnerStatus,
    downloads: Mapping[NodeId, Sequence[DownloadProgress]] | None = None,
) -> State:
    runner_ids = list(inst.shard_assignments.node_to_runner.values())
    return State(
        instances={inst.instance_id: inst},
        runners=dict(zip(runner_ids, runners, strict=False)),
        downloads=downloads or {},
    )


def ongoing(inst: Instance, node: NodeId, downloaded_gb: int, eta_ms: int):
    runner = inst.shard_assignments.node_to_runner[node]
    return DownloadOngoing(
        node_id=node,
        shard_metadata=inst.shard_assignments.runner_to_shard[runner],
        download_progress=DownloadProgressData(
            total=Memory.from_bytes(10 * GB),
            downloaded=Memory.from_bytes(downloaded_gb * GB),
            downloaded_this_session=Memory.from_bytes(downloaded_gb * GB),
            completed_files=0,
            total_files=4,
            speed=1.0,
            eta_ms=eta_ms,
            files={},
        ),
    )


def completed(inst: Instance, node: NodeId) -> DownloadCompleted:
    runner = inst.shard_assignments.node_to_runner[node]
    return DownloadCompleted(
        node_id=node,
        shard_metadata=inst.shard_assignments.runner_to_shard[runner],
        total=Memory.from_bytes(10 * GB),
    )


def test_a_ready_model_is_ready() -> None:
    inst = instance()
    assert (
        model_readiness(state_of(inst, RunnerReady(), RunnerRunning()), MODEL) is None
    )


def test_a_model_with_one_ready_instance_is_ready() -> None:
    loading, ready = instance(), instance()
    state = state_of(loading, RunnerIdle(), RunnerIdle())
    state = state.model_copy(
        update={
            "instances": {**state.instances, ready.instance_id: ready},
            "runners": {
                **state.runners,
                **{
                    runner: RunnerReady()
                    for runner in ready.shard_assignments.node_to_runner.values()
                },
            },
        }
    )
    assert model_readiness(state, MODEL) is None


def test_a_downloading_model_says_how_far_it_got() -> None:
    inst = instance()
    state = state_of(
        inst,
        RunnerIdle(),
        RunnerIdle(),
        downloads={
            NODES[0]: [ongoing(inst, NODES[0], downloaded_gb=3, eta_ms=180_000)],
            NODES[1]: [completed(inst, NODES[1])],
        },
    )
    assert model_readiness(state, MODEL) == NotReady(
        status="downloading (65%, about 3 min left)", retry_after_seconds=60
    )


def test_a_download_that_has_not_started_says_so() -> None:
    inst = instance()
    runner = inst.shard_assignments.node_to_runner[NODES[0]]
    state = state_of(
        inst,
        RunnerIdle(),
        RunnerIdle(),
        downloads={
            NODES[0]: [
                DownloadPending(
                    node_id=NODES[0],
                    shard_metadata=inst.shard_assignments.runner_to_shard[runner],
                )
            ]
        },
    )
    assert model_readiness(state, MODEL) == NotReady(
        status="waiting for its download to start", retry_after_seconds=5
    )


def test_a_download_about_to_finish_is_retried_soon() -> None:
    inst = instance()
    state = state_of(
        inst,
        RunnerIdle(),
        RunnerIdle(),
        downloads={NODES[0]: [ongoing(inst, NODES[0], downloaded_gb=9, eta_ms=2_000)]},
    )
    readiness = model_readiness(state, MODEL)
    assert readiness is not None
    assert readiness.retry_after_seconds == 5


def test_a_failed_download_says_why() -> None:
    inst = instance()
    runner = inst.shard_assignments.node_to_runner[NODES[0]]
    state = state_of(
        inst,
        RunnerIdle(),
        RunnerIdle(),
        downloads={
            NODES[0]: [
                DownloadFailed(
                    node_id=NODES[0],
                    shard_metadata=inst.shard_assignments.runner_to_shard[runner],
                    error_message="Not enough disk space",
                )
            ]
        },
    )
    readiness = model_readiness(state, MODEL)
    assert readiness is not None
    assert readiness.status == "its download failed: Not enough disk space"
    assert readiness.failed


@pytest.mark.parametrize(
    ("runners", "status"),
    [
        ((), "starting"),
        ((RunnerIdle(), RunnerIdle()), "starting"),
        (
            (
                RunnerLoading(layers_loaded=10, total_layers=16),
                RunnerLoading(layers_loaded=2, total_layers=16),
            ),
            "loading (12 of 32 layers)",
        ),
        ((RunnerReady(), RunnerWarmingUp()), "warming up"),
    ],
)
def test_a_loading_model_is_retried_soon(
    runners: tuple[RunnerStatus, ...], status: str
) -> None:
    inst = instance()
    downloads = {node: [completed(inst, node)] for node in NODES}
    assert model_readiness(
        state_of(inst, *runners, downloads=downloads), MODEL
    ) == NotReady(status=status, retry_after_seconds=5)


def test_a_model_whose_runner_failed_is_not_waited_for() -> None:
    inst = instance()
    readiness = model_readiness(
        state_of(inst, RunnerFailed(diagnostics=[]), RunnerReady()), MODEL
    )
    assert readiness is not None
    assert readiness.failed


def make_api(state: State) -> tuple[API, AsyncMock]:
    api = object.__new__(API)
    api.app = FastAPI()
    # Reaching every node
    api.membership = FakeMembership(NodeId("api-node"), *NODES)
    api.state = state
    api.paused = False
    api._text_generation_queues = {}
    api._image_generation_queues = {}
    api._cancelled_command_ids = set()
    api._send = send = AsyncMock()
    api._setup_exception_handlers()
    api.app.post("/v1/chat/completions", response_model=None)(api.chat_completions)
    return api, send


def chat(api: API, stream: bool) -> tuple[int, dict[str, str], Any]:
    response = TestClient(api.app).post(
        "/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": stream,
        },
    )
    return response.status_code, dict(response.headers), response.json()


@pytest.mark.parametrize("stream", [True, False])
def test_a_request_during_a_download_gets_a_503_after_the_wait(
    stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_main, "MODEL_READY_WAIT_SECONDS", 0.3)
    inst = instance()
    api, send = make_api(
        state_of(
            inst,
            RunnerIdle(),
            RunnerIdle(),
            downloads={
                NODES[0]: [ongoing(inst, NODES[0], downloaded_gb=4, eta_ms=30_000)]
            },
        )
    )

    status, headers, body = chat(api, stream)

    assert status == 503
    assert headers["retry-after"] == "30"
    assert body["error"]["message"] == (
        "The model is not ready yet: downloading (40%, about 30 s left). "
        "Try again in 30 s."
    )
    assert body["error"]["code"] == "model_not_ready"
    send.assert_not_called()


def test_a_model_that_failed_to_start_gets_a_503_at_once() -> None:
    inst = instance()
    api, send = make_api(state_of(inst, RunnerFailed(diagnostics=[]), RunnerIdle()))

    started = time.monotonic()
    status, headers, body = chat(api, stream=False)

    assert time.monotonic() - started < 5
    assert status == 503
    assert headers["retry-after"] == "10"
    assert body["error"]["message"] == (
        "The model is not ready yet: it failed to start, and is being started again. "
        "Try again in 10 s."
    )
    send.assert_not_called()


def test_a_request_while_loading_gets_a_503_after_a_short_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_main, "MODEL_READY_WAIT_SECONDS", 0.3)
    inst = instance()
    api, send = make_api(
        state_of(
            inst,
            RunnerLoading(layers_loaded=1, total_layers=16),
            RunnerIdle(),
            downloads={node: [completed(inst, node)] for node in NODES},
        )
    )

    status, headers, body = chat(api, stream=True)

    assert status == 503
    assert headers["retry-after"] == "5"
    assert body["error"]["message"] == (
        "The model is not ready yet: loading (1 of 16 layers). Try again in 5 s."
    )
    send.assert_not_called()


async def test_a_request_while_loading_goes_ahead_once_the_model_is_ready() -> None:
    inst = instance()
    api, _ = make_api(state_of(inst, RunnerWarmingUp(), RunnerWarmingUp()))

    async def finish_warming_up() -> None:
        await anyio.sleep(0.5)
        api.state = state_of(inst, RunnerReady(), RunnerReady())

    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:
            tg.start_soon(finish_warming_up)
            assert await api._validate_model_has_instance(MODEL) == MODEL


async def test_a_model_deleted_while_waiting_is_not_found() -> None:
    inst = instance()
    api, _ = make_api(state_of(inst, RunnerWarmingUp(), RunnerWarmingUp()))

    async def delete() -> None:
        await anyio.sleep(0.3)
        api.state = State()

    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:
            tg.start_soon(delete)
            with pytest.raises(api_main.HTTPException) as raised:
                await api._validate_model_has_instance(MODEL)
            assert raised.value.status_code == 404


def test_a_whole_response_refused_by_the_master_as_not_ready_is_a_503() -> None:
    app = FastAPI()
    app.get("/")(
        lambda: CollectedResponse(
            collect_chat_response(
                CommandId("cmd"),
                _chunks(
                    ErrorChunk(
                        model=MODEL,
                        error_message="The model is not ready yet: loading (1 of 2 "
                        "layers). Try again in 5 s.",
                    )
                ),
            ),
            media_type="application/json",
        )
    )

    reply = TestClient(app).get("/")

    assert reply.status_code == 503
    assert reply.json()["error"]["code"] == "model_not_ready"


async def _chunks(*items: ErrorChunk) -> AsyncGenerator[ErrorChunk, None]:
    for item in items:
        yield item
