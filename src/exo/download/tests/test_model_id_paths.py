"""A model id can never make a model path point outside its models directory."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api.main import API
from exo.download.download_utils import delete_model
from exo.shared.types.common import ModelId


@pytest.mark.parametrize("model_id", ["", ".", ".."])
def test_ids_naming_the_current_or_parent_directory_are_rejected(model_id: str) -> None:
    with pytest.raises(ValueError):
        # Id("") would become a random id, so build the str directly
        str.__new__(ModelId, model_id).normalize()


@pytest.mark.parametrize(
    ("model_id", "folder"),
    [
        (
            "mlx-community/Llama-3.2-1B-Instruct-4bit",
            "mlx-community--Llama-3.2-1B-Instruct-4bit",
        ),
        ("../..", "..--.."),
        ("..model", "..model"),
    ],
)
def test_other_ids_become_a_single_folder_name(model_id: str, folder: str) -> None:
    assert ModelId(model_id).normalize() == folder


async def test_deleting_the_parent_directory_is_refused(tmp_path: Path) -> None:
    models = tmp_path / "models"
    (models / "some--model").mkdir(parents=True)
    precious = tmp_path / "precious.txt"
    precious.write_text("not a model")

    with (
        patch("exo.download.download_utils.EXO_MODELS_DIRS", (models,)),
        patch("exo.download.download_utils.EXO_DEFAULT_MODELS_DIR", models),
        pytest.raises(ValueError),
    ):
        await delete_model(ModelId(".."))

    assert precious.exists()
    assert (models / "some--model").exists()


@pytest.mark.parametrize("raw_id", ["%2e%2e", "%2E"])
def test_api_rejects_a_delete_for_the_parent_directory(raw_id: str) -> None:
    api = object.__new__(API)
    api._send_download = AsyncMock()  # pyright: ignore[reportPrivateUsage]
    app = FastAPI()
    app.delete("/download/{node_id}/{model_id:path}")(api.delete_download)

    response = TestClient(app).delete(f"/download/some-node/{raw_id}")

    assert response.status_code == 400
    api._send_download.assert_not_called()  # pyright: ignore[reportPrivateUsage]
