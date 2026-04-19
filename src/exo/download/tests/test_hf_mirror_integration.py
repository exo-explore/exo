"""Opt-in integration tests against the real hf-mirror.com.

These transfer < 5 KB total (one 783-byte config.json download + a few HEAD calls)
and take under 3 seconds. They are deselected by default (`slow` marker) AND gated
behind `EXO_TEST_NETWORK=1` so CI never hits an external service.

Run manually:
    EXO_TEST_NETWORK=1 uv run pytest -m slow src/exo/download/tests/test_hf_mirror_integration.py -v
"""

import os
from pathlib import Path
from unittest.mock import patch

import aiofiles
import aiofiles.os as aios
import pytest

from exo.download.download_utils import (
    _download_file,  # pyright: ignore[reportPrivateUsage]
    _fetch_file_list,  # pyright: ignore[reportPrivateUsage]
    file_meta,
)
from exo.shared.types.common import ModelId

# A stable small repo we know is served by hf-mirror.
_SMALL_REPO = ModelId("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
_CONFIG_SIZE = 783  # verified 2026-04-19
_SAFETENSORS_SIZE = 278064920  # verified 2026-04-19


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("EXO_TEST_NETWORK") != "1",
        reason="Set EXO_TEST_NETWORK=1 to run network-dependent tests",
    ),
]


@pytest.fixture
def hf_mirror_endpoint() -> str:
    return "https://hf-mirror.com"


class TestHfMirrorRealTraffic:
    async def test_fetch_file_list_shape_matches_huggingface(
        self, hf_mirror_endpoint: str
    ) -> None:
        files = await _fetch_file_list(
            _SMALL_REPO, "main", "", False, endpoint=hf_mirror_endpoint
        )
        paths = {f.path for f in files}
        # These files exist in this repo regardless of re-uploads; asserting
        # exact sizes would be brittle.
        assert "config.json" in paths
        assert "tokenizer.json" in paths
        assert "model.safetensors" in paths

    async def test_file_meta_small_file_reads_correct_size(
        self, hf_mirror_endpoint: str
    ) -> None:
        length, etag = await file_meta(
            _SMALL_REPO, "main", "config.json", endpoint=hf_mirror_endpoint
        )
        assert length == _CONFIG_SIZE
        assert etag  # git sha1

    async def test_file_meta_lfs_file_reads_x_linked_size(
        self, hf_mirror_endpoint: str
    ) -> None:
        """HEAD only — does not download the 278 MB body."""
        length, etag = await file_meta(
            _SMALL_REPO, "main", "model.safetensors", endpoint=hf_mirror_endpoint
        )
        assert length == _SAFETENSORS_SIZE
        assert len(etag) == 64, f"Expected sha256 (xet/LFS), got {etag!r}"

    async def test_download_small_file_end_to_end(
        self, hf_mirror_endpoint: str, tmp_path: Path
    ) -> None:
        """Full /resolve/ → 307 → /api/resolve-cache/ → 200 path; hash-verified."""
        target_dir = tmp_path / "d"
        await aios.makedirs(target_dir, exist_ok=True)
        with patch.dict(os.environ, {"HF_ENDPOINT": hf_mirror_endpoint}):
            path = await _download_file(
                _SMALL_REPO,
                "main",
                "config.json",
                target_dir,
                endpoint=hf_mirror_endpoint,
            )
        assert await aios.path.exists(path)
        async with aiofiles.open(path, "rb") as f:
            content = await f.read()
        assert len(content) == _CONFIG_SIZE
        # It's JSON — quick sanity check
        assert content.lstrip().startswith(b"{")
