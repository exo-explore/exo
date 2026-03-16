from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from exo.master.api import API
from exo.shared.types.common import NodeId
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
)

TARGET_MODEL = "mlx-community/Qwen3.5-397B-A17B-4bit"


def _api_with_state(
    *, runners: dict[RunnerId, object], runner_ids: list[RunnerId] | None = None
) -> API:
    api = API.__new__(API)
    assigned_runner_ids = runner_ids or list(runners.keys())
    api.state = SimpleNamespace(
        instances={
            "inst-1": SimpleNamespace(
                shard_assignments=SimpleNamespace(
                    model_id=TARGET_MODEL,
                    runner_to_shard={
                        runner_id: object() for runner_id in assigned_runner_ids
                    },
                )
            )
        },
        runners=runners,
    )
    api.node_id = NodeId("node-1")
    return api


def test_model_readiness_poll_waiting_when_no_runners_assigned() -> None:
    api = _api_with_state(runners={}, runner_ids=[])

    response = API.model_readiness_poll(api, TARGET_MODEL)
    payload = json.loads(response.body)

    assert payload["ready"] is False
    assert payload["terminal"] is False
    assert payload["status"] == "waiting"
    assert payload["message"] == "No runners assigned yet"


def test_model_readiness_poll_reports_loading_details() -> None:
    runner_id = RunnerId("runner-1")
    api = _api_with_state(
        runners={runner_id: RunnerLoading(layers_loaded=12, total_layers=48)}
    )

    response = API.model_readiness_poll(api, TARGET_MODEL)
    payload = json.loads(response.body)

    assert payload["ready"] is False
    assert payload["terminal"] is False
    assert payload["status"] == "loading"
    assert payload["runner_details"]["runner-1"]["layers_loaded"] == 12
    assert payload["runner_details"]["runner-1"]["total_layers"] == 48
    assert payload["status_counts"]["RunnerLoading"] == 1


def test_model_readiness_poll_reports_terminal_failure() -> None:
    runner_id = RunnerId("runner-1")
    api = _api_with_state(runners={runner_id: RunnerFailed(error_message="boom")})

    response = API.model_readiness_poll(api, TARGET_MODEL)
    payload = json.loads(response.body)

    assert payload["ready"] is False
    assert payload["terminal"] is True
    assert payload["status"] == "failed"
    assert payload["runner_details"]["runner-1"]["error_message"] == "boom"
    assert payload["status_counts"]["RunnerFailed"] == 1


def test_model_readiness_poll_reports_warming_phase() -> None:
    runner_id = RunnerId("runner-1")
    api = _api_with_state(runners={runner_id: RunnerLoaded()})

    response = API.model_readiness_poll(api, TARGET_MODEL)
    payload = json.loads(response.body)

    assert payload["ready"] is False
    assert payload["terminal"] is False
    assert payload["status"] == "warming"
    assert payload["runners"]["runner-1"] == "RunnerLoaded"


def test_model_readiness_poll_reports_ready() -> None:
    runner_id = RunnerId("runner-1")
    api = _api_with_state(runners={runner_id: RunnerReady()})

    response = API.model_readiness_poll(api, TARGET_MODEL)
    payload = json.loads(response.body)

    assert payload["ready"] is True
    assert payload["terminal"] is False
    assert payload["status"] == "ready"
    assert payload["message"] == f"{TARGET_MODEL} ready for inference"


@pytest.mark.asyncio
async def test_model_readiness_stream_stops_on_terminal_failure() -> None:
    runner_id = RunnerId("runner-1")
    api = _api_with_state(runners={runner_id: RunnerFailed(error_message="boom")})

    response = API.model_readiness(api, TARGET_MODEL)
    chunks = [chunk async for chunk in response.body_iterator]

    assert len(chunks) == 1
    payload = json.loads(chunks[0].removeprefix("data: ").strip())
    assert payload["status"] == "failed"
    assert payload["terminal"] is True
    assert payload["runner_details"]["runner-1"]["error_message"] == "boom"
