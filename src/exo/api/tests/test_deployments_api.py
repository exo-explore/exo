# pyright: reportUnusedFunction=false, reportAny=false
from typing import Any
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api.main import API
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.commands import CreateDeployment, DeleteDeployment
from exo.shared.types.deployments import Deployment, DeploymentId
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.worker.instances import InstanceMeta
from exo.shared.types.worker.shards import Sharding

MODEL = ModelCard(
    model_id=ModelId("test/model-a"),
    storage_size=Memory.from_bytes(1000),
    n_layers=10,
    hidden_size=30,
    supports_tensor=True,
    tasks=[ModelTask.TextGeneration],
    backends=[Backend.MlxMetal],
)


def _make_api(state: State) -> Any:
    app = FastAPI()
    api = object.__new__(API)
    api.app = app
    api.state = state
    api._send = AsyncMock()  # pyright: ignore[reportPrivateUsage]
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]
    app.get("/deployments")(api.list_deployments)
    app.post("/deployments")(api.create_deployment)
    app.delete("/deployments/{deployment_id}")(api.delete_deployment)
    return api


def _deployment() -> Deployment:
    return Deployment(
        deployment_id=DeploymentId(),
        model_card=MODEL,
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=1,
    )


def test_create_deployment_sends_the_request_to_the_master() -> None:
    api = _make_api(State())
    client = TestClient(api.app)

    with patch.object(ModelCard, "load", AsyncMock(return_value=MODEL)):
        response = client.post(
            "/deployments",
            json={
                "model_id": "test/model-a",
                "sharding": "Tensor",
                "instance_meta": "MlxJaccl",
                "min_nodes": 2,
            },
        )

    assert response.status_code == 200
    api._send.assert_called_once()
    command = api._send.call_args[0][0]
    assert isinstance(command, CreateDeployment)
    assert command.deployment.model_card == MODEL
    assert command.deployment.sharding == Sharding.Tensor
    assert command.deployment.instance_meta == InstanceMeta.MlxJaccl
    assert command.deployment.min_nodes == 2
    assert response.json()["deployment_id"] == command.deployment.deployment_id


def test_create_deployment_refuses_a_model_already_kept_running() -> None:
    existing = _deployment()
    api = _make_api(State(deployments={existing.deployment_id: existing}))
    client = TestClient(api.app)

    with patch.object(ModelCard, "load", AsyncMock(return_value=MODEL)):
        response = client.post("/deployments", json={"model_id": "test/model-a"})

    assert response.status_code == 409
    assert existing.deployment_id in response.json()["error"]["message"]
    api._send.assert_not_called()


def test_delete_deployment_sends_the_request_to_the_master() -> None:
    existing = _deployment()
    api = _make_api(State(deployments={existing.deployment_id: existing}))
    client = TestClient(api.app)

    response = client.delete(f"/deployments/{existing.deployment_id}")

    assert response.status_code == 200
    command = api._send.call_args[0][0]
    assert isinstance(command, DeleteDeployment)
    assert command.deployment_id == existing.deployment_id


def test_delete_unknown_deployment_returns_404() -> None:
    api = _make_api(State())
    client = TestClient(api.app)

    response = client.delete(f"/deployments/{DeploymentId()}")

    assert response.status_code == 404
    api._send.assert_not_called()


def test_list_deployments_shows_each_with_its_status() -> None:
    waiting = _deployment()
    failing = _deployment().model_copy(
        update={
            "model_card": MODEL.model_copy(update={"model_id": ModelId("test/b")}),
            "placement_error": "No cycles found with sufficient memory",
        }
    )
    api = _make_api(
        State(
            deployments={
                waiting.deployment_id: waiting,
                failing.deployment_id: failing,
            }
        )
    )
    client = TestClient(api.app)

    response = client.get("/deployments")

    assert response.status_code == 200
    listed = {
        item["deployment"]["deploymentId"]: item
        for item in response.json()["deployments"]
    }
    assert listed[waiting.deployment_id]["status"] == "placing"
    assert listed[failing.deployment_id]["status"] == "cant_place"
    assert (
        listed[failing.deployment_id]["deployment"]["placementError"]
        == "No cycles found with sufficient memory"
    )
