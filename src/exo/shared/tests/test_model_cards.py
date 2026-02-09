import pytest

from exo.shared.models.model_cards import ModelId, fetch_safetensors_size
from exo.shared.types.memory import Memory


@pytest.mark.asyncio
async def test_fetch_safetensors_size_falls_back_when_index_missing(monkeypatch):
    async def _raise_not_found(*_args, **_kwargs):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(
        "exo.download.download_utils.download_file_with_retry", _raise_not_found
    )

    class _FakeSafetensors:
        total = 1234

    class _FakeInfo:
        safetensors = _FakeSafetensors()

    monkeypatch.setattr("exo.shared.models.model_cards.model_info", lambda _m: _FakeInfo())

    size = await fetch_safetensors_size(ModelId("fake/model"))
    assert size == Memory.from_bytes(1234)
