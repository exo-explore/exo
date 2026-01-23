from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

import exo.download.download_utils as download_utils
from exo.shared.types.common import ModelId


async def test_download_file_with_retry_retries_file_not_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: int = 0

    async def fake_download_file(
        model_id: ModelId,
        revision: str,
        path: str,
        target_dir: Path,
        on_progress: Callable[[int, int, bool], None],
        *,
        endpoint: str | None = None,
    ) -> Path:
        _ = endpoint
        _ = on_progress
        nonlocal calls
        calls += 1
        if calls < 3:
            raise FileNotFoundError("Not ready yet")
        return target_dir / path

    async def fast_sleep(_: float) -> None: ...

    monkeypatch.setattr(download_utils, "_download_file", fake_download_file)
    monkeypatch.setattr(download_utils.asyncio, "sleep", fast_sleep)

    out = await download_utils.download_file_with_retry(
        model_id=ModelId("foo/bar"),
        revision="main",
        path="config.json",
        target_dir=tmp_path,
        retry_on_not_found=True,
    )
    assert out == tmp_path / "config.json"
    assert calls == 3


async def test_download_file_with_retry_does_not_retry_file_not_found_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: int = 0

    async def fake_download_file(
        model_id: ModelId,
        revision: str,
        path: str,
        target_dir: Path,
        on_progress: Callable[[int, int, bool], None],
        *,
        endpoint: str | None = None,
    ) -> Path:
        _ = endpoint
        _ = on_progress
        _ = target_dir
        _ = model_id
        _ = revision
        _ = path
        nonlocal calls
        calls += 1
        raise FileNotFoundError("Missing")

    monkeypatch.setattr(download_utils, "_download_file", fake_download_file)

    with pytest.raises(FileNotFoundError):
        await download_utils.download_file_with_retry(
            model_id=ModelId("foo/bar"),
            revision="main",
            path="config.json",
            target_dir=tmp_path,
        )
    assert calls == 1

