# pyright: reportUnusedFunction=false, reportAny=false
from typing import Any
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api.main import API
from exo.shared.models.model_cards import Backend, ModelCard, ModelTask
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.worker.downloads import DownloadCompleted
from exo.shared.types.worker.shards import PipelineShardMetadata

NODE_A = NodeId("node-a")


def _card(model_id: str) -> ModelCard:
    return ModelCard(
        model_id=model_id,  # pyright: ignore[reportArgumentType]
        storage_size=Memory(),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )


def _downloaded(card: ModelCard) -> DownloadCompleted:
    return DownloadCompleted(
        node_id=NODE_A,
        shard_metadata=PipelineShardMetadata(
            model_card=card,
            device_rank=0,
            world_size=1,
            start_layer=0,
            end_layer=1,
            n_layers=1,
        ),
        total=Memory(),
    )


def _make_api(state: State) -> Any:
    app = FastAPI()
    api = object.__new__(API)
    api.app = app
    api.state = state
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]
    app.get("/models")(api.get_models)
    app.get("/v1/models")(api.get_v1_models)
    return api


def test_v1_models_defaults_to_downloaded_only() -> None:
    downloaded = _card("downloaded/model")
    not_downloaded = _card("not-downloaded/model")
    api = _make_api(
        State(downloads={NODE_A: [_downloaded(downloaded)]}),
    )
    client = TestClient(api.app)

    with patch(
        "exo.api.main.model_cards.card_cache.list_all",
        new=AsyncMock(return_value=[downloaded, not_downloaded]),
    ):
        response = client.get("/v1/models")

    assert response.status_code == 200
    ids = [m["id"] for m in response.json()["data"]]
    assert ids == ["downloaded/model"]


def test_v1_models_status_all_returns_everything() -> None:
    downloaded = _card("downloaded/model")
    not_downloaded = _card("not-downloaded/model")
    api = _make_api(
        State(downloads={NODE_A: [_downloaded(downloaded)]}),
    )
    client = TestClient(api.app)

    with patch(
        "exo.api.main.model_cards.card_cache.list_all",
        new=AsyncMock(return_value=[downloaded, not_downloaded]),
    ):
        response = client.get("/v1/models?status=all")

    assert response.status_code == 200
    ids = {m["id"] for m in response.json()["data"]}
    assert ids == {"downloaded/model", "not-downloaded/model"}


def test_plain_models_endpoint_still_defaults_to_all() -> None:
    """The dashboard's own model browser (/models, no status) must keep
    seeing every known card, not just downloaded ones."""
    downloaded = _card("downloaded/model")
    not_downloaded = _card("not-downloaded/model")
    api = _make_api(
        State(downloads={NODE_A: [_downloaded(downloaded)]}),
    )
    client = TestClient(api.app)

    with patch(
        "exo.api.main.model_cards.card_cache.list_all",
        new=AsyncMock(return_value=[downloaded, not_downloaded]),
    ):
        response = client.get("/models")

    assert response.status_code == 200
    ids = {m["id"] for m in response.json()["data"]}
    assert ids == {"downloaded/model", "not-downloaded/model"}
