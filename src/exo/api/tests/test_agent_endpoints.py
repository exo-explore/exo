# pyright: reportPrivateUsage=false

from collections.abc import AsyncGenerator
from types import MethodType
from typing import Any, cast

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from exo.api.main import API
from exo.api.types import CreateInstanceParams, ModelList
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.commands import Command, CreateInstance, TextGeneration
from exo.shared.types.common import CommandId, Host, ModelId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage
from exo.shared.types.state import State
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata


class _RequestStub:
    base_url = "http://testserver/"


def _model_card(model_id: ModelId) -> ModelCard:
    return ModelCard(
        model_id=model_id,
        storage_size=Memory.from_mb(1),
        n_layers=1,
        hidden_size=1,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )


def _instance(model_id: ModelId, instance_id: InstanceId) -> MlxRingInstance:
    node_id = NodeId("node-one")
    runner_id = RunnerId(f"runner-{instance_id}")
    shard = PipelineShardMetadata(
        model_card=_model_card(model_id),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    )
    return MlxRingInstance(
        instance_id=instance_id,
        shard_assignments=ShardAssignments(
            model_id=model_id,
            runner_to_shard={runner_id: shard},
            node_to_runner={node_id: runner_id},
        ),
        hosts_by_node={node_id: [Host(ip="127.0.0.1", port=1)]},
        ephemeral_port=1,
    )


def _memory_usage(*, available_mb: int, total_mb: int) -> MemoryUsage:
    return MemoryUsage.from_bytes(
        ram_total=Memory.from_mb(total_mb).in_bytes,
        ram_available=Memory.from_mb(available_mb).in_bytes,
        swap_total=0,
        swap_available=0,
    )


def _api_with_instances(
    instances: dict[InstanceId, MlxRingInstance],
) -> API:
    api = API.__new__(API)
    api.state = State(instances=instances)

    async def _noop_notify(_: API, __: ModelId) -> None:
        return None

    api._trigger_notify_user_to_download_model = MethodType(_noop_notify, api)
    return api


def _route_api_with_instances(
    instances: dict[InstanceId, MlxRingInstance],
) -> API:
    api = _api_with_instances(instances)
    api.app = FastAPI()

    async def _capture_send(
        self: API,
        task_params: TextGenerationTaskParams,
        target_instance_id: InstanceId | None = None,
    ) -> TextGeneration:
        return TextGeneration(
            task_params=task_params, target_instance_id=target_instance_id
        )

    async def _finite_token_stream(
        _self: API, _command_id: CommandId
    ) -> AsyncGenerator[TokenChunk, None]:
        yield TokenChunk(
            model=ModelId("mlx-community/Test-Model-4bit"),
            token_id=0,
            text="hello",
            usage=None,
            finish_reason="stop",
        )

    api._send_text_generation_with_images = MethodType(_capture_send, api)
    api._token_chunk_stream = MethodType(_finite_token_stream, api)
    api._setup_exception_handlers()
    api._setup_routes()
    return api


def _text_params(model_id: ModelId) -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=model_id,
        input=[
            InputMessage(role="user", content=InputMessageContent("hello")),
        ],
    )


def test_provider_list_includes_default_model_and_instance_endpoints() -> None:
    model_id = ModelId("mlx-community/Test-Model-4bit")
    instance_id = InstanceId("instance-one")
    api = _api_with_instances({instance_id: _instance(model_id, instance_id)})

    providers = api.get_agent_endpoints(_RequestStub()).data  # type: ignore[arg-type]

    assert providers[0].name == "default"
    assert providers[0].openai_base_url == "http://testserver/v1"
    assert providers[0].claude_base_url == "http://testserver"
    assert any(
        provider.kind == "model"
        and provider.model_id == model_id
        and provider.target_instance_id is None
        and provider.claude_base_url is None
        for provider in providers
    )
    assert any(
        provider.kind == "instance"
        and provider.model_id == model_id
        and provider.target_instance_id == instance_id
        and provider.openai_base_url == "http://testserver/agents/inst-instance-one/v1"
        and provider.claude_base_url is None
        for provider in providers
    )


@pytest.mark.asyncio
async def test_create_instance_allows_single_node_total_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_id = ModelId("mlx-community/Test-Model-4bit")
    instance = _instance(model_id, InstanceId("instance-one"))
    api = _api_with_instances({})
    api.state = State(
        node_memory={
            NodeId("node-one"): _memory_usage(available_mb=1_000, total_mb=2_000)
        }
    )
    sent_commands: list[Command] = []

    async def _load_model_card(_: ModelId) -> ModelCard:
        return _model_card(model_id).model_copy(
            update={"storage_size": Memory.from_mb(1_500)}
        )

    async def _capture_send(self: API, command: Command) -> None:
        sent_commands.append(command)

    monkeypatch.setattr(ModelCard, "load", _load_model_card)
    api._send = MethodType(_capture_send, api)

    response = await api.create_instance(CreateInstanceParams(instance=instance))

    assert response.model_card.storage_size == Memory.from_mb(1_500)
    assert len(sent_commands) == 1
    assert isinstance(sent_commands[0], CreateInstance)
    assert sent_commands[0].instance == instance


@pytest.mark.asyncio
async def test_create_instance_rejects_single_node_over_total_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_id = ModelId("mlx-community/Test-Model-4bit")
    instance = _instance(model_id, InstanceId("instance-one"))
    api = _api_with_instances({})
    api.state = State(
        node_memory={
            NodeId("node-one"): _memory_usage(available_mb=1_000, total_mb=1_200)
        }
    )

    async def _load_model_card(_: ModelId) -> ModelCard:
        return _model_card(model_id).model_copy(
            update={"storage_size": Memory.from_mb(1_500)}
        )

    monkeypatch.setattr(ModelCard, "load", _load_model_card)

    with pytest.raises(HTTPException, match="Insufficient memory"):
        await api.create_instance(CreateInstanceParams(instance=instance))


@pytest.mark.asyncio
async def test_agent_chat_dispatches_with_target_instance_id() -> None:
    model_id = ModelId("mlx-community/Test-Model-4bit")
    instance_id = InstanceId("instance-one")
    api = _api_with_instances({instance_id: _instance(model_id, instance_id)})
    captured: dict[str, Any] = {}

    async def _capture_send(
        self: API,
        task_params: TextGenerationTaskParams,
        target_instance_id: InstanceId | None = None,
    ) -> TextGeneration:
        captured["model"] = task_params.model
        captured["target_instance_id"] = target_instance_id
        return TextGeneration(
            task_params=task_params, target_instance_id=target_instance_id
        )

    api._send_text_generation_with_images = MethodType(_capture_send, api)
    route = api._resolve_text_generation_route(
        ModelId("ignored-request-model"),
        f"inst-{instance_id}",
        _RequestStub(),  # type: ignore[arg-type]
    )

    command = await api._send_routed_text_generation(
        _text_params(ModelId("ignored-request-model")), route
    )

    assert captured == {"model": model_id, "target_instance_id": instance_id}
    assert command.task_params.model == model_id
    assert command.target_instance_id == instance_id


@pytest.mark.asyncio
async def test_model_endpoint_dispatches_without_target_instance_id() -> None:
    model_id = ModelId("mlx-community/Test-Model-4bit")
    instance_id = InstanceId("instance-one")
    api = _api_with_instances({instance_id: _instance(model_id, instance_id)})
    captured: dict[str, Any] = {}
    endpoint = api._model_endpoint_name(model_id)

    async def _capture_send(
        self: API,
        task_params: TextGenerationTaskParams,
        target_instance_id: InstanceId | None = None,
    ) -> TextGeneration:
        captured["model"] = task_params.model
        captured["target_instance_id"] = target_instance_id
        return TextGeneration(
            task_params=task_params, target_instance_id=target_instance_id
        )

    api._send_text_generation_with_images = MethodType(_capture_send, api)
    route = api._resolve_text_generation_route(
        ModelId("ignored-request-model"),
        endpoint,
        _RequestStub(),  # type: ignore[arg-type]
    )

    command = await api._send_routed_text_generation(
        _text_params(ModelId("ignored-request-model")), route
    )

    assert captured == {"model": model_id, "target_instance_id": None}
    assert command.task_params.model == model_id
    assert command.target_instance_id is None


def test_unknown_agent_endpoint_returns_404_before_dispatch() -> None:
    api = _api_with_instances({})

    with pytest.raises(HTTPException) as exception_info:
        api._resolve_text_generation_route(
            ModelId("model"),
            "inst-deleted",
            _RequestStub(),  # type: ignore[arg-type]
        )

    assert exception_info.value.status_code == 404


def test_http_provider_routes_list_agent_endpoints() -> None:
    model_id = ModelId("mlx-community/Test-Model-4bit")
    instance_id = InstanceId("instance-one")
    api = _route_api_with_instances({instance_id: _instance(model_id, instance_id)})
    client = TestClient(api.app)

    providers_response = client.get("/v1/providers")
    agents_response = client.get("/agents")

    assert providers_response.status_code == 200
    assert agents_response.status_code == 200
    assert providers_response.json() == agents_response.json()
    providers_payload = cast(dict[str, object], providers_response.json())
    providers = cast(list[dict[str, object]], providers_payload["data"])
    provider_names = [provider["name"] for provider in providers]
    assert "default" in provider_names
    assert f"inst-{instance_id}" in provider_names


def test_http_agent_models_returns_backing_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_id = ModelId("mlx-community/Test-Model-4bit")
    instance_id = InstanceId("instance-one")
    api = _route_api_with_instances({instance_id: _instance(model_id, instance_id)})
    client = TestClient(api.app)

    async def _fail_load(_: ModelId) -> ModelCard:
        raise AssertionError("agent model listing should use active shard metadata")

    monkeypatch.setattr(ModelCard, "load", _fail_load)

    response = client.get(f"/agents/inst-{instance_id}/v1/models")

    assert response.status_code == 200
    payload = cast(dict[str, object], response.json())
    models = cast(list[dict[str, object]], payload["data"])
    assert models == [
        {
            "id": str(model_id),
            "object": "model",
            "created": models[0]["created"],
            "owned_by": "exo",
            "hugging_face_id": str(model_id),
            "name": model_id.short(),
            "description": "",
            "context_length": 0,
            "tags": [],
            "storage_size_megabytes": 1,
            "supports_tensor": True,
            "tasks": ["TextGeneration"],
            "is_custom": False,
            "reasoning_dialect": "none",
            "family": "",
            "quantization": "",
            "base_model": "",
            "capabilities": [],
            "drafter_model_ids": [],
        }
    ]


def test_http_default_agent_models_forwards_status_filter() -> None:
    api = _route_api_with_instances({})
    client = TestClient(api.app)
    captured: dict[str, str | None] = {}

    async def _capture_get_models(
        _self: API,
        status: str | None = None,
    ) -> ModelList:
        captured["status"] = status
        return ModelList(data=[])

    api.get_models = MethodType(_capture_get_models, api)

    response = client.get("/agents/default/v1/models?status=downloaded")

    assert response.status_code == 200
    assert response.json() == {"object": "list", "data": []}
    assert captured == {"status": "downloaded"}


def test_http_unknown_agent_chat_returns_404_before_dispatch() -> None:
    api = _route_api_with_instances({})
    client = TestClient(api.app)

    response = client.post(
        "/agents/missing/v1/chat/completions",
        json={
            "model": "missing-model",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 404
    assert response.json()["error"]["message"] == "Agent endpoint not found: missing"


def test_http_agent_responses_reports_resolved_model() -> None:
    model_id = ModelId("mlx-community/Test-Model-4bit")
    instance_id = InstanceId("instance-one")
    api = _route_api_with_instances({instance_id: _instance(model_id, instance_id)})
    client = TestClient(api.app)

    response = client.post(
        f"/agents/inst-{instance_id}/v1/responses",
        json={"model": "ignored-request-model", "input": "hello"},
    )

    assert response.status_code == 200
    assert response.json()["model"] == str(model_id)


def test_http_default_endpoint_preserves_body_model_routing() -> None:
    model_id = ModelId("mlx-community/Test-Model-4bit")
    instance_id = InstanceId("instance-one")
    api = _route_api_with_instances({instance_id: _instance(model_id, instance_id)})
    client = TestClient(api.app)

    response = client.post(
        "/agents/default/v1/chat/completions",
        json={"model": str(model_id), "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    assert response.json()["model"] == str(model_id)
