# pyright: reportUnusedFunction=false, reportAny=false
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api.main import API
from exo.shared.types.common import CommandId
from exo.shared.types.worker.instances import InstanceId


def _make_api() -> Any:
    """Create a minimal API instance with cancel route and error handler."""

    app = FastAPI()
    api = object.__new__(API)
    api.app = app
    api._bridge_command_instances = {}  # pyright: ignore[reportPrivateUsage]
    api.task_requester = MagicMock()
    api.task_requester.interrupt = AsyncMock()
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]
    app.post("/v1/cancel/{command_id}")(api.cancel_command)
    return api


def test_cancel_nonexistent_command_returns_404() -> None:
    """Cancel for an unknown command_id returns 404 in OpenAI error format."""
    api = _make_api()
    client = TestClient(api.app)

    response = client.post("/v1/cancel/nonexistent-id")
    assert response.status_code == 404
    data: dict[str, Any] = response.json()
    assert "error" in data
    assert data["error"]["message"] == "Command not found or already completed"
    assert data["error"]["type"] == "Not Found"
    assert data["error"]["code"] == 404


def test_cancel_active_text_generation() -> None:
    """Cancel an active text generation command: returns 200, interrupt sent."""
    api = _make_api()
    client = TestClient(api.app)

    cid = CommandId("text-cmd-123")
    instance_id = InstanceId("instance-a")
    api._bridge_command_instances[cid] = instance_id

    response = client.post(f"/v1/cancel/{cid}")
    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    assert data["message"] == "Command cancelled."
    assert data["command_id"] == str(cid)
    api.task_requester.interrupt.assert_called_once()
    args = api.task_requester.interrupt.call_args.args
    assert args[0] == instance_id
    assert args[1] == cid
    assert '"cancelled_command_id":"text-cmd-123"' in args[2]


def test_cancel_active_image_generation() -> None:
    """Cancel an active image generation command: returns 200, interrupt sent."""
    api = _make_api()
    client = TestClient(api.app)

    cid = CommandId("img-cmd-456")
    instance_id = InstanceId("instance-b")
    api._bridge_command_instances[cid] = instance_id

    response = client.post(f"/v1/cancel/{cid}")
    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    assert data["message"] == "Command cancelled."
    assert data["command_id"] == str(cid)
    api.task_requester.interrupt.assert_called_once()
    args = api.task_requester.interrupt.call_args.args
    assert args[0] == instance_id
    assert args[1] == cid
    assert '"cancelled_command_id":"img-cmd-456"' in args[2]
