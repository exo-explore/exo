import pytest
from fastapi import HTTPException

from exo.shared.types.common import CommandId


def _make_api_with_inflight(n: int, *, limit: int = 2):
    from exo.master.api import API

    api = object.__new__(API)
    api.max_in_flight_text_generations = limit
    api._text_generation_queues = {
        CommandId(f"cmd-{i}"): object() for i in range(n)
    }  # pyright: ignore[reportPrivateUsage]
    return api


def test_backpressure_allows_when_below_limit() -> None:
    api = _make_api_with_inflight(1, limit=2)
    api._enforce_text_generation_backpressure()  # pyright: ignore[reportPrivateUsage]


def test_backpressure_rejects_when_at_limit() -> None:
    api = _make_api_with_inflight(2, limit=2)
    with pytest.raises(HTTPException) as exc:
        api._enforce_text_generation_backpressure()  # pyright: ignore[reportPrivateUsage]

    assert exc.value.status_code == 429
    assert "Server is busy processing other generations" in str(exc.value.detail)
