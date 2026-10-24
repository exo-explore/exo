"""Unit tests for cluster model streaming fast load (TODO #17)."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiofiles
import aiofiles.os as aios
import pytest
from fastapi import HTTPException, Request

from exo.download.download_utils import (
    _download_file,  # pyright: ignore[reportPrivateUsage]
    fetch_file_list_from_peer,
    get_cluster_model_peer_urls,
)
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import (
    NetworkInterfaceInfo,
    NodeNetworkInfo,
)
from exo.shared.types.state import State
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadPending,
)
from exo.shared.types.worker.shards import PipelineShardMetadata


def _make_dummy_shard(model_id: ModelId) -> PipelineShardMetadata:
    model_card = ModelCard(
        model_id=model_id,
        storage_size=Memory.from_mb(100),
        n_layers=8,
        hidden_size=64,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )
    return PipelineShardMetadata(
        model_card=model_card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=8,
        n_layers=8,
    )


class TestClusterPeerDiscovery:
    """Tests for get_cluster_model_peer_urls."""

    def test_get_cluster_model_peer_urls_discovers_peers_and_prioritizes_thunderbolt(
        self,
    ) -> None:
        model_id = ModelId("org/test-model")
        node_a = NodeId()
        node_b = NodeId()

        shard = _make_dummy_shard(model_id)
        completed_download = DownloadCompleted(
            shard_metadata=shard,
            node_id=node_b,
            total=Memory.from_mb(100),
            model_directory="/models/test",
            read_only=False,
        )

        topology = Topology()
        topology.add_node(node_a)
        topology.add_node(node_b)

        # Node B has Thunderbolt and WiFi interfaces
        node_network = {
            node_b: NodeNetworkInfo(
                interfaces=[
                    NetworkInterfaceInfo(
                        name="en0",
                        ip_address="192.168.1.50",
                        interface_type="wifi",
                    ),
                    NetworkInterfaceInfo(
                        name="bridge0",
                        ip_address="10.0.0.2",
                        interface_type="thunderbolt",
                    ),
                ]
            )
        }

        state = State(
            downloads={node_b: [completed_download]},
            topology=topology,
            node_network=node_network,
        )

        urls = get_cluster_model_peer_urls(model_id, state, current_node_id=node_a)
        # Should discover node_b's endpoints and rank Thunderbolt (10.0.0.2) before WiFi (192.168.1.50)
        assert len(urls) >= 2
        assert urls[0] == "http://10.0.0.2:52415"
        assert urls[1] == "http://192.168.1.50:52415"

    def test_get_cluster_model_peer_urls_ignores_self_and_incomplete(self) -> None:
        model_id = ModelId("org/test-model")
        node_a = NodeId()
        node_b = NodeId()

        shard = _make_dummy_shard(model_id)

        # Node A (self) has completed, Node B is ongoing
        completed_self = DownloadCompleted(
            shard_metadata=shard,
            node_id=node_a,
            total=Memory.from_mb(100),
            model_directory="/models/test",
        )
        ongoing_peer = DownloadPending(
            shard_metadata=shard,
            node_id=node_b,
            model_directory="/models/test",
        )

        state = State(
            downloads={
                node_a: [completed_self],
                node_b: [ongoing_peer],
            }
        )

        urls = get_cluster_model_peer_urls(model_id, state, current_node_id=node_a)
        assert urls == []


class TestPeerFileList:
    """Tests for fetch_file_list_from_peer."""

    async def test_fetch_file_list_from_peer_success(self) -> None:
        model_id = ModelId("org/test-model")
        peer_sources = ["http://10.0.0.2:52415"]

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(
            return_value={
                "model_id": str(model_id),
                "files": [
                    {"path": "config.json", "size": 512},
                    {"path": "model.safetensors", "size": 1048576},
                ],
            }
        )

        mock_session = AsyncMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_resp))
        )
        mock_session_ctx = MagicMock(
            __aenter__=AsyncMock(return_value=mock_session), __aexit__=AsyncMock()
        )

        with patch(
            "exo.download.download_utils.create_http_session",
            return_value=mock_session_ctx,
        ):
            files = await fetch_file_list_from_peer(model_id, peer_sources)
            assert files is not None
            assert len(files) == 2
            assert files[0].path == "config.json"
            assert files[0].size == 512
            assert files[1].path == "model.safetensors"
            assert files[1].size == 1048576

    async def test_fetch_file_list_from_peer_failure_returns_none(self) -> None:
        model_id = ModelId("org/test-model")
        peer_sources = ["http://10.0.0.2:52415"]

        mock_resp = AsyncMock()
        mock_resp.status = 500

        mock_session = AsyncMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_resp))
        )
        mock_session_ctx = MagicMock(
            __aenter__=AsyncMock(return_value=mock_session), __aexit__=AsyncMock()
        )

        with patch(
            "exo.download.download_utils.create_http_session",
            return_value=mock_session_ctx,
        ):
            files = await fetch_file_list_from_peer(model_id, peer_sources)
            assert files is None


class TestDownloadFileWithPeerStreaming:
    """Tests for P2P streaming in _download_file."""

    async def test_download_file_streams_successfully_from_peer(
        self, tmp_path: Path
    ) -> None:
        model_id = ModelId("org/test-model")
        target_dir = tmp_path / "models" / "org_test-model"
        await aios.makedirs(target_dir, exist_ok=True)

        content = b"streaming bytes payload from peer over high speed thunderbolt link"
        total_len = len(content)

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.headers = {"Content-Length": str(total_len)}

        # Stream chunks
        chunks = [content[:20], content[20:]]

        async def read_mock(size: int = 8192) -> bytes:
            if chunks:
                return chunks.pop(0)
            return b""

        mock_resp.content = MagicMock()
        mock_resp.content.read = AsyncMock(side_effect=read_mock)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_resp))
        )
        mock_session_ctx = MagicMock(
            __aenter__=AsyncMock(return_value=mock_session), __aexit__=AsyncMock()
        )

        progress_calls: list[tuple[int, int, bool]] = []

        def on_progress(n: int, total: int, done: bool) -> None:
            progress_calls.append((n, total, done))

        with patch(
            "exo.download.download_utils.create_http_session",
            return_value=mock_session_ctx,
        ):
            result = await _download_file(
                model_id=model_id,
                revision="main",
                path="weights.safetensors",
                target_dir=target_dir,
                on_progress=on_progress,
                skip_internet=True,
                peer_sources=["http://10.0.0.2:52415"],
            )

            assert await aios.path.exists(result)
            async with aiofiles.open(result, "rb") as f:
                saved_content = await f.read()
            assert saved_content == content
            assert len(progress_calls) >= 2
            assert progress_calls[-1] == (total_len, total_len, True)

    async def test_download_file_offline_mode_fails_if_no_peer_available(
        self, tmp_path: Path
    ) -> None:
        model_id = ModelId("org/test-model")
        target_dir = tmp_path / "models" / "org_test-model"

        with pytest.raises(FileNotFoundError, match="cannot download in offline mode"):
            await _download_file(
                model_id=model_id,
                revision="main",
                path="nonexistent.safetensors",
                target_dir=target_dir,
                skip_internet=True,
                peer_sources=[],
            )


class TestApiStreamingHandlers:
    """Tests for API streaming and file list endpoints."""

    async def test_stream_model_file_and_get_model_files(self, tmp_path: Path) -> None:
        from exo.api.main import API

        # Mock API instance
        api = MagicMock(spec=API)
        model_dir = tmp_path / "models" / "test-model"
        model_dir.mkdir(parents=True, exist_ok=True)

        sample_file = model_dir / "config.json"
        sample_file.write_text('{"model_type": "llama", "hidden_size": 4096}')

        with patch("exo.api.main.resolve_existing_model", return_value=model_dir):
            # Test get_model_files
            resp = await API.get_model_files(api, "test-model")
            body = resp.body.decode()
            assert "config.json" in body

            # Test stream_model_file full
            mock_req = MagicMock(spec=Request)
            mock_req.headers = {}
            stream_resp = await API.stream_model_file(
                api, "test-model", "config.json", mock_req
            )
            assert stream_resp.status_code == 200

            # Test stream_model_file Range request
            mock_req.headers = {"Range": "bytes=0-10"}
            range_resp = await API.stream_model_file(
                api, "test-model", "config.json", mock_req
            )
            assert range_resp.status_code == 206
            assert "Content-Range" in range_resp.headers
            assert range_resp.headers["Content-Range"].startswith("bytes 0-10/")

    async def test_stream_model_file_blocks_path_traversal(
        self, tmp_path: Path
    ) -> None:
        from exo.api.main import API

        api = MagicMock(spec=API)
        model_dir = tmp_path / "models" / "test-model"
        model_dir.mkdir(parents=True, exist_ok=True)

        mock_req = MagicMock(spec=Request)
        mock_req.headers = {}

        with patch("exo.api.main.resolve_existing_model", return_value=model_dir):
            with pytest.raises(HTTPException) as exc_info:
                await API.stream_model_file(
                    api, "test-model", "../../etc/passwd", mock_req
                )
            assert exc_info.value.status_code == 403
            assert "path traversal" in exc_info.value.detail
