"""The macOS app's tests decode a /state answer from a committed file. This test builds
that answer from exo's own types, through the API's /state handler, and fails when the
file no longer matches, so the app's decoding can't drift from what exo sends.

After changing a type that /state includes, write the file again, then run the app's
tests (EXOTests in app/EXO) to see whether the app still reads it:

    uv run python src/exo/api/tests/test_app_state_fixture.py
"""

import json
from pathlib import Path
from typing import cast

from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api.main import API
from exo.shared.apply import apply
from exo.shared.types.common import CommandId, ModelId, NodeId
from exo.shared.types.events import (
    Event,
    IndexedEvent,
    InstanceCreated,
    NodeDownloadProgress,
    NodeGatheredInfo,
    RunnerStatusUpdated,
    TaskCreated,
    TopologyEdgeCreated,
)
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage, ThunderboltBridgeStatus
from exo.shared.types.state import State
from exo.shared.types.tasks import TaskId, TaskStatus, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.topology import Connection, RDMAConnection
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
    DownloadProgressData,
)
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runners import (
    RunnerId,
    RunnerIdle,
    RunnerReady,
    RunnerRunning,
)
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.info_gatherer.info_gatherer import (
    MiscData,
    RdmaCtlStatus,
    StaticNodeInformation,
    ThunderboltBridgeInfo,
)
from exo.worker.tests.unittests.conftest import (
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)

FIXTURE = (
    Path(__file__).resolve().parents[4]
    / "app"
    / "EXO"
    / "EXOTests"
    / "Fixtures"
    / "state.json"
)

_WHEN = "2026-01-01T00:00:00+00:00"


def _progress(downloaded: int, total: int) -> DownloadProgressData:
    """A download of one file, as far as `downloaded` of `total` bytes."""
    file = DownloadProgressData(
        total=Memory.from_bytes(total),
        downloaded=Memory.from_bytes(downloaded),
        downloaded_this_session=Memory.from_bytes(downloaded),
        completed_files=0,
        total_files=1,
        speed=1000.0,
        eta_ms=575,
        files={},
    )
    return file.model_copy(update={"files": {"model.safetensors": file}})


def _state() -> State:
    """Two Macs: "this" runs Llama with "other", which is downloading Qwen. Between
    them, the downloads include one of every status."""
    this, other = NodeId("this"), NodeId("other")
    llama = ModelId("mlx-community/Llama-3.2-3B-Instruct-4bit")
    qwen = ModelId("mlx-community/Qwen3-30B-A3B-4bit")
    some_other = ModelId("mlx-community/Some-Other-Model")
    not_started = ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit")
    r1, r2, r3 = RunnerId("r1"), RunnerId("r2"), RunnerId("r3")

    def shard(model: ModelId, rank: int = 0, world_size: int = 1) -> ShardMetadata:
        return get_pipeline_shard_metadata(model, rank, world_size)

    events: list[Event] = [
        NodeGatheredInfo(
            node_id=this,
            when=_WHEN,
            info=StaticNodeInformation(
                model="Mac Studio",
                chip="Apple M3 Ultra",
                os_version="26.0",
                os_build_version="25A354",
            ),
        ),
        NodeGatheredInfo(
            node_id=this, when=_WHEN, info=MiscData(friendly_name="Studio")
        ),
        NodeGatheredInfo(
            node_id=this,
            when=_WHEN,
            info=MemoryUsage(
                ram_total=Memory.from_gb(192),
                ram_available=Memory.from_gb(151),
                swap_total=Memory(),
                swap_available=Memory(),
            ),
        ),
        NodeGatheredInfo(
            node_id=this,
            when=_WHEN,
            info=ThunderboltBridgeInfo(
                status=ThunderboltBridgeStatus(
                    enabled=False, exists=True, service_name="Thunderbolt Bridge"
                )
            ),
        ),
        NodeGatheredInfo(node_id=this, when=_WHEN, info=RdmaCtlStatus(enabled=True)),
        NodeGatheredInfo(
            node_id=other,
            when=_WHEN,
            info=StaticNodeInformation(
                model="Mac mini",
                chip="Apple M4 Pro",
                os_version="26.0",
                os_build_version="25A354",
            ),
        ),
        NodeGatheredInfo(
            node_id=other,
            when=_WHEN,
            info=MemoryUsage(
                ram_total=Memory.from_gb(64),
                ram_available=Memory.from_gb(52),
                swap_total=Memory(),
                swap_available=Memory(),
            ),
        ),
        TopologyEdgeCreated(
            conn=Connection(
                source=this,
                sink=other,
                edge=RDMAConnection(
                    source_rdma_iface="rdma_en2", sink_rdma_iface="rdma_en2"
                ),
            )
        ),
        InstanceCreated(
            instance=get_mlx_ring_instance(
                InstanceId("i1"),
                llama,
                {this: r1, other: r2},
                {r1: shard(llama, 0, 2), r2: shard(llama, 1, 2)},
            )
        ),
        InstanceCreated(
            instance=get_mlx_ring_instance(
                InstanceId("i2"), qwen, {other: r3}, {r3: shard(qwen)}
            )
        ),
        RunnerStatusUpdated(runner_id=r1, runner_status=RunnerReady()),
        RunnerStatusUpdated(runner_id=r2, runner_status=RunnerRunning()),
        RunnerStatusUpdated(runner_id=r3, runner_status=RunnerIdle()),
        NodeDownloadProgress(
            download_progress=DownloadCompleted(
                node_id=this,
                shard_metadata=shard(llama, 0, 2),
                total=Memory.from_bytes(1000),
                model_directory="/models/Llama-3.2-3B-Instruct-4bit",
            )
        ),
        NodeDownloadProgress(
            download_progress=DownloadFailed(
                node_id=this,
                shard_metadata=shard(some_other),
                error_message="Connection lost",
            )
        ),
        NodeDownloadProgress(
            download_progress=DownloadOngoing(
                node_id=other,
                shard_metadata=shard(qwen),
                download_progress=_progress(downloaded=425, total=1000),
            )
        ),
        NodeDownloadProgress(
            download_progress=DownloadOngoing(
                node_id=other,
                shard_metadata=shard(some_other),
                download_progress=_progress(downloaded=1, total=1000),
            )
        ),
        NodeDownloadProgress(
            download_progress=DownloadPending(
                node_id=other, shard_metadata=shard(not_started)
            )
        ),
        TaskCreated(
            task_id=TaskId("t1"),
            task=TextGeneration(
                task_id=TaskId("t1"),
                command_id=CommandId("c1"),
                instance_id=InstanceId("i1"),
                task_status=TaskStatus.Running,
                task_params=TextGenerationTaskParams(
                    model=llama,
                    input=[
                        InputMessage(role="user", content=InputMessageContent("hi"))
                    ],
                ),
            ),
        ),
    ]
    state = State()
    for idx, event in enumerate(events):
        state = apply(state, IndexedEvent(idx=idx, event=event))
    return state


def served_state() -> object:
    """The state as GET /state sends it."""
    api = object.__new__(API)
    api.state = _state()
    app = FastAPI()
    app.get("/state")(api.get_state)
    return cast(object, TestClient(app).get("/state").json())


def test_app_state_fixture_is_what_exo_sends() -> None:
    committed = cast(object, json.loads(FIXTURE.read_text()))
    assert committed == served_state(), (
        f"{FIXTURE} no longer matches what /state sends. Write it again (see "
        "this file's docstring) and check that the app's tests still pass."
    )


if __name__ == "__main__":
    _ = FIXTURE.write_text(json.dumps(served_state(), indent=2) + "\n")
