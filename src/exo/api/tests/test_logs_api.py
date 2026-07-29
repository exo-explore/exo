# pyright: reportUnusedFunction=false, reportAny=false
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api import main as api_main
from exo.api.main import (
    API,
    _tail_file,  # pyright: ignore[reportPrivateUsage]
)


def _make_api() -> Any:
    """Create a minimal API instance with the log routes registered."""
    app = FastAPI()
    api = object.__new__(API)
    api.app = app
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]
    app.get("/v1/logs")(api.list_logs)
    app.get("/v1/logs/{name}")(api.get_log_tail)
    app.get("/v1/logs/{name}/raw")(api.get_log_raw)
    return api


@pytest.fixture(autouse=True)
def _isolated_log_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the whitelist at a temp dir so tests never touch a real ~/.exo log."""
    log_path = tmp_path / "exo.log"
    monkeypatch.setattr(api_main, "_LOG_FILES", {"main": log_path})
    return log_path


def test_tail_file_returns_last_n_lines(tmp_path: Path) -> None:
    path = tmp_path / "a.log"
    path.write_text("\n".join(f"line{i}" for i in range(10)))

    content, truncated = _tail_file(path, max_lines=3)

    assert content == "line7\nline8\nline9"
    assert truncated is True


def test_tail_file_returns_everything_when_under_limit(tmp_path: Path) -> None:
    path = tmp_path / "a.log"
    path.write_text("line0\nline1")

    content, truncated = _tail_file(path, max_lines=1000)

    assert content == "line0\nline1"
    assert truncated is False


def test_tail_file_zero_lines_returns_empty(tmp_path: Path) -> None:
    """Regression test: max_lines<=0 must not fall back to returning everything
    (Python's `lines[-0:]` is the full list, not an empty slice)."""
    path = tmp_path / "a.log"
    path.write_text("line0\nline1\nline2")

    content, truncated = _tail_file(path, max_lines=0)

    assert content == ""
    assert truncated is True


def test_tail_file_negative_lines_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "a.log"
    path.write_text("line0\nline1")

    content, truncated = _tail_file(path, max_lines=-5)

    assert content == ""
    assert truncated is True


def test_list_logs_only_includes_existing_files(_isolated_log_files: Path) -> None:
    api = _make_api()
    client = TestClient(api.app)

    response = client.get("/v1/logs")
    assert response.status_code == 200
    assert response.json() == {"logs": []}

    _isolated_log_files.write_text("hello\n")
    response = client.get("/v1/logs")
    data = response.json()
    assert len(data["logs"]) == 1
    assert data["logs"][0]["name"] == "main"
    assert data["logs"][0]["fileSize"] == len("hello\n")


def test_get_log_tail_unknown_name_returns_404() -> None:
    api = _make_api()
    client = TestClient(api.app)

    response = client.get("/v1/logs/does_not_exist")
    assert response.status_code == 404


def test_get_log_tail_missing_file_returns_404(_isolated_log_files: Path) -> None:
    api = _make_api()
    client = TestClient(api.app)

    response = client.get("/v1/logs/main")
    assert response.status_code == 404


def test_get_log_tail_returns_content(_isolated_log_files: Path) -> None:
    _isolated_log_files.write_text("line0\nline1\nline2")
    api = _make_api()
    client = TestClient(api.app)

    response = client.get("/v1/logs/main?lines=2")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "main"
    assert data["content"] == "line1\nline2"
    assert data["truncated"] is True


def test_get_log_raw_unknown_name_returns_404() -> None:
    api = _make_api()
    client = TestClient(api.app)

    response = client.get("/v1/logs/does_not_exist/raw")
    assert response.status_code == 404


def test_get_log_raw_returns_full_file(_isolated_log_files: Path) -> None:
    _isolated_log_files.write_text("full contents\n")
    api = _make_api()
    client = TestClient(api.app)

    response = client.get("/v1/logs/main/raw")
    assert response.status_code == 200
    assert response.text == "full contents\n"
