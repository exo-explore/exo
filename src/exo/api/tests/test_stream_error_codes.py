# pyright: reportPrivateUsage=false, reportAny=false
"""A failed generation reaches the client as one short sentence with a stable code, in a
stream and in a whole response, whatever the endpoint, and an instance deleted under a request
says why."""

import json
from collections.abc import AsyncGenerator
from typing import cast

from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api.adapters.chat_completions import (
    collect_chat_response,
    generate_chat_stream,
)
from exo.api.adapters.claude import generate_claude_stream
from exo.api.adapters.responses import generate_responses_stream
from exo.api.collected_response import CollectedResponse
from exo.shared.request_errors import generation_error_code
from exo.shared.serving import why_instance_ended
from exo.shared.topology import Topology
from exo.shared.types.chunks import (
    ErrorChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.common import CommandId, ModelId, NodeId
from exo.shared.types.state import State
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runners import RunnerId
from exo.worker.tests.unittests.conftest import (
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)

MODEL = ModelId("test-model")
DEVICE_LEFT = "A device serving this model disconnected before the reply finished."

type Chunk = ErrorChunk | TokenChunk | ToolCallChunk | PrefillProgressChunk


async def chunks(*items: Chunk) -> AsyncGenerator[Chunk, None]:
    for item in items:
        yield item


def failing() -> AsyncGenerator[Chunk, None]:
    return chunks(
        TokenChunk(model=MODEL, text="Hel", token_id=1, usage=None),
        ErrorChunk(model=MODEL, error_message=DEVICE_LEFT),
    )


async def test_a_stream_error_carries_its_code() -> None:
    events = [
        event async for event in generate_chat_stream(CommandId("cmd"), failing())
    ]

    error = json.loads(events[-2].removeprefix("data: "))["error"]
    assert error == {
        "message": DEVICE_LEFT,
        "type": "InternalServerError",
        "code": "device_left",
    }
    assert events[-1] == "data: [DONE]\n\n"


def test_a_failed_whole_response_carries_its_code() -> None:
    app = FastAPI()
    app.get("/")(
        lambda: CollectedResponse(
            collect_chat_response(CommandId("cmd"), failing()),
            media_type="application/json",
        )
    )

    reply = TestClient(app).get("/")

    assert reply.status_code == 500
    error = cast(dict[str, dict[str, object]], reply.json())["error"]
    assert error["message"] == DEVICE_LEFT
    assert error["code"] == "device_left"


def test_an_error_without_a_code_keeps_the_status_as_code() -> None:
    app = FastAPI()
    app.get("/")(
        lambda: CollectedResponse(
            collect_chat_response(
                CommandId("cmd"),
                chunks(ErrorChunk(model=MODEL, error_message="Something failed")),
            ),
            media_type="application/json",
        )
    )

    error = cast(dict[str, dict[str, object]], TestClient(app).get("/").json())["error"]
    assert error["code"] == 500


def _ending_of_request_when_instance_deleted(node_still_in_cluster: bool) -> str:
    """What the requests of an instance are told when it is deleted (the output node ends them
    with this, exo.worker.serving.ServingHost.end_instance)"""
    node, runner = NodeId("node"), RunnerId()
    instance = get_mlx_ring_instance(
        instance_id=InstanceId(),
        model_id=MODEL,
        node_to_runner={node: runner},
        runner_to_shard={runner: get_pipeline_shard_metadata(MODEL, device_rank=0)},
    )
    topology = Topology()
    if node_still_in_cluster:
        topology.add_node(node)
    return why_instance_ended(State(topology=topology), instance)


def test_a_request_whose_device_left_says_so() -> None:
    ending = _ending_of_request_when_instance_deleted(node_still_in_cluster=False)
    assert generation_error_code(ending) == "device_left"
    assert ending == (
        "A device serving this model left the cluster before the reply finished."
    )


def test_a_request_whose_model_was_stopped_says_so() -> None:
    ending = _ending_of_request_when_instance_deleted(node_still_in_cluster=True)
    assert generation_error_code(ending) == "model_stopped"
    assert ending == "The model was stopped before the reply finished."


def test_every_message_exo_ends_a_request_with_has_a_code() -> None:
    for message in (
        "A device serving this model disconnected before the reply finished.",
        "A device serving this model left the cluster before the reply finished.",
        "The model was stopped before the reply finished.",
        "The model instance was shut down before this request could start.",
        "No instance found for model test-model",
        "A device serving this model ran out of GPU memory before the reply finished.",
        "The GPU of a device serving this model timed out before the reply finished.",
        "The model stopped responding before the reply finished and is being restarted.",
        "The model stopped unexpectedly before the reply finished.",
        "The model stopped before the reply finished: Unsupported image format.",
        "The model instance serving this request stopped before it finished",
    ):
        assert generation_error_code(message) is not None, message
    assert generation_error_code("Something else") is None


async def test_a_claude_stream_ends_with_an_error_event_carrying_the_code() -> None:
    events = [
        event
        async for event in generate_claude_stream(CommandId("cmd"), MODEL, failing())
    ]

    assert events[-1].startswith("event: error\n")
    error = json.loads(events[-1].split("data: ", 1)[1])
    assert error == {
        "type": "error",
        "error": {"type": "api_error", "message": DEVICE_LEFT, "code": "device_left"},
    }
    assert not any("message_stop" in event for event in events)


async def test_a_responses_stream_fails_with_the_code() -> None:
    events = [
        event
        async for event in generate_responses_stream(CommandId("cmd"), MODEL, failing())
    ]

    assert "response.failed" in events[-1]
    failed = json.loads(events[-1].split("data: ", 1)[1])
    assert failed["response"]["status"] == "failed"
    assert failed["response"]["error"] == {
        "code": "device_left",
        "message": DEVICE_LEFT,
    }
    assert not any("response.completed" in event for event in events)
