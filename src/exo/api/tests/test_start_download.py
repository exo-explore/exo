from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from exo.api.main import API
from exo.api.types import StartDownloadParams
from exo.shared.models.model_cards import ModelCard
from exo.shared.tests.conftest import get_pipeline_shard_metadata
from exo.shared.types.commands import StartDownload
from exo.shared.types.common import ModelId, NodeId
from exo.worker.tests.constants import MODEL_A_ID


def _api() -> tuple[API, AsyncMock]:
    api = object.__new__(API)
    api.app = FastAPI()
    sent = AsyncMock()
    api._send_download = sent  # pyright: ignore[reportPrivateUsage]
    api.app.post("/download/start")(api.start_download)
    return api, sent


def test_a_catalogue_model_is_downloaded_whole_by_id() -> None:
    # The downloads page lists the catalogue, and the state no longer has an entry (with
    # its shard) for a model that isn't on any node
    card = get_pipeline_shard_metadata(
        MODEL_A_ID, device_rank=0, world_size=1
    ).model_card
    api, sent = _api()
    with patch.object(ModelCard, "load", AsyncMock(return_value=card)):
        response = TestClient(api.app).post(
            "/download/start", json={"targetNodeId": "node-1", "modelId": MODEL_A_ID}
        )
    assert response.status_code == 200, response.text
    command = cast(StartDownload, sent.await_args.args[0])  # pyright: ignore[reportOptionalMemberAccess]
    assert command.target_node_id == NodeId("node-1")
    shard = command.shard_metadata
    assert shard.model_card == card
    assert (shard.start_layer, shard.end_layer, shard.world_size) == (
        0,
        card.n_layers,
        1,
    )


def test_a_shard_is_still_downloaded_as_given() -> None:
    shard = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    api, sent = _api()
    response = TestClient(api.app).post(
        "/download/start",
        json={"targetNodeId": "node-1", "shardMetadata": shard.model_dump(mode="json")},
    )
    assert response.status_code == 200, response.text
    assert cast(StartDownload, sent.await_args.args[0]).shard_metadata == shard  # pyright: ignore[reportOptionalMemberAccess]


def test_exactly_one_of_shard_or_model_is_given() -> None:
    shard = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=1)
    with pytest.raises(ValidationError):
        StartDownloadParams(target_node_id=NodeId("n"))
    with pytest.raises(ValidationError):
        StartDownloadParams(
            target_node_id=NodeId("n"), shard_metadata=shard, model_id=ModelId("m")
        )
