from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import pytest
from pydantic import TypeAdapter

from exo.download.model_store_server import ModelStoreServer
from exo.shared.types.common import ModelId
from exo.shared.types.worker.downloads import FileListEntry


async def test_model_store_tree_and_resolve_supports_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    import exo.download.model_store_server as model_store_server
    import exo.shared.constants as constants

    monkeypatch.setattr(constants, "EXO_MODELS_DIR", models_dir)
    monkeypatch.setattr(model_store_server, "EXO_MODELS_DIR", models_dir)

    model_id = "foo/bar"
    normalized = ModelId(model_id).normalize()
    model_dir = models_dir / normalized
    model_dir.mkdir(parents=True, exist_ok=True)

    content = b"hello world"
    (model_dir / "config.json").write_bytes(content)

    server = ModelStoreServer(port=0)
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        tree_res = await client.get(f"/api/models/{model_id}/tree/main")
        assert tree_res.status_code == 200
        files = TypeAdapter(list[FileListEntry]).validate_json(tree_res.text)
        assert any(f.path == "config.json" for f in files)

        head_res = await client.head(f"/{model_id}/resolve/main/config.json")
        assert head_res.status_code == 200
        assert head_res.headers["content-length"] == str(len(content))
        assert head_res.headers["etag"] == hashlib.sha256(content).hexdigest()

        range_res = await client.get(
            f"/{model_id}/resolve/main/config.json",
            headers={"Range": "bytes=1-3"},
        )
        assert range_res.status_code == 206
        assert range_res.content == content[1:4]
        assert range_res.headers["content-range"] == f"bytes 1-3/{len(content)}"


async def test_model_store_blocks_metadata_and_partial_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    import exo.download.model_store_server as model_store_server
    import exo.shared.constants as constants

    monkeypatch.setattr(constants, "EXO_MODELS_DIR", models_dir)
    monkeypatch.setattr(model_store_server, "EXO_MODELS_DIR", models_dir)

    model_id = "foo/bar"
    normalized = ModelId(model_id).normalize()
    model_dir = models_dir / normalized
    (model_dir / ".exo" / "download_metadata").mkdir(parents=True, exist_ok=True)

    (model_dir / ".exo" / "download_metadata" / "secret.json").write_text("nope")
    (model_dir / "weights.safetensors.partial").write_bytes(b"partial")

    server = ModelStoreServer(port=0)
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res1 = await client.get(
            f"/{model_id}/resolve/main/.exo/download_metadata/secret.json"
        )
        assert res1.status_code == 404

        res2 = await client.get(f"/{model_id}/resolve/main/weights.safetensors.partial")
        assert res2.status_code == 404

