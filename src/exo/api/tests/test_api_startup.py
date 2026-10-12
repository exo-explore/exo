"""The startup banner is printed only once the API answers requests, and a port that's in use stops
exo with one clear line rather than a traceback."""

import os
import socket
import subprocess
import sys
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import cast

import anyio
import httpx
import pytest
from fastapi import FastAPI

import exo.api.main as api_main
from exo.api.listening_socket import ApiPortInUseError, listen_on_api_port
from exo.api.main import API


def bound_port(bound_socket: socket.socket) -> int:
    return cast(tuple[str, int], bound_socket.getsockname())[1]


@contextmanager
def port_in_use() -> Iterator[int]:
    """A free port, taken by another listener for the duration."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as other_listener:
        other_listener.bind(("0.0.0.0", 0))
        other_listener.listen()
        yield bound_port(other_listener)


def test_a_port_in_use_is_reported_clearly():
    with port_in_use() as port, pytest.raises(ApiPortInUseError) as error_info:
        listen_on_api_port(port)

    assert str(error_info.value) == (
        f"Port {port} is already in use - is exo already running? "
        "Use --api-port to choose another."
    )


def test_exo_exits_with_one_line_when_its_port_is_in_use(tmp_path: Path):
    # In case exo ever got as far as the banner, don't open a browser
    (tmp_path / ".dashboard_opened").touch()
    environment = {**os.environ, "EXO_HOME": str(tmp_path)}

    with port_in_use() as port:
        exo = subprocess.run(
            [
                sys.executable,
                "-m",
                "exo",
                "--no-worker",
                "--api-port",
                str(port),
                "--namespace",
                f"port-in-use-test-{port}",
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )

    output = exo.stdout + exo.stderr
    assert exo.returncode == 1, output
    assert f"Port {port} is already in use - is exo already running?" in output
    assert "Traceback" not in output
    assert "Dashboard & API Ready" not in output


async def test_the_banner_is_printed_once_the_api_answers(
    monkeypatch: pytest.MonkeyPatch,
):
    events: list[str] = []

    @asynccontextmanager
    async def slow_startup(_app: FastAPI) -> AsyncIterator[None]:
        await anyio.sleep(0.3)
        events.append("serving")
        yield

    app = FastAPI(lifespan=slow_startup)
    app.get("/node_id")(lambda: "test-node")

    banner_printed = anyio.Event()

    def record_banner(_port: int) -> None:
        events.append("banner")
        banner_printed.set()

    monkeypatch.setattr(api_main, "print_startup_banner", record_banner)

    listening_socket = listen_on_api_port(0)
    port = bound_port(listening_socket)
    api = object.__new__(API)
    api.app = app
    api.port = port
    api._listening_socket = listening_socket  # pyright: ignore[reportPrivateUsage]

    shutdown = anyio.Event()
    async with anyio.create_task_group() as tg:
        # As API.run starts them
        tg.start_soon(api.run_api, shutdown)
        tg.start_soon(api._print_banner_once_serving)  # pyright: ignore[reportPrivateUsage]
        try:
            with anyio.fail_after(10):
                await banner_printed.wait()
            assert events == ["serving", "banner"]
            async with httpx.AsyncClient(trust_env=False) as client:
                response = await client.get(f"http://127.0.0.1:{port}/node_id")
            assert response.status_code == 200
        finally:
            # The server is shielded from cancellation, it stops only when told to
            shutdown.set()
