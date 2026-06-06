import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

from exo.shared.telemetry import RunnerStderrSubmission, TelemetryService


@dataclass(frozen=True)
class RecordedRequest:
    method: str
    url: str
    content: bytes


def _queue_submission(
    service: TelemetryService,
    submission: RunnerStderrSubmission,
) -> None:
    service._send.send_nowait(submission)  # pyright: ignore[reportPrivateUsage]
    service._send.close()  # pyright: ignore[reportPrivateUsage]


@pytest.mark.anyio
async def test_runner_stderr_upload_hashes_and_uploads_file_bytes(tmp_path: Path):
    log_bytes = b"runner stderr\nsecond line\n"
    log_path = tmp_path / "runner.stderr.log"
    log_path.write_bytes(log_bytes)
    requests: list[RecordedRequest] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(
            RecordedRequest(
                method=request.method,
                url=str(request.url),
                content=await request.aread(),
            )
        )
        if request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "key": "runner_log/test.stderr.log",
                    "uploadUrl": "https://uploads.example/runner.stderr.log",
                    "expiresIn": 300,
                    "maxSize": 52428800,
                },
            )
        if request.method == "PUT":
            return httpx.Response(200)
        return httpx.Response(404)

    service = TelemetryService.create(
        telemetry_disabled=False,
        api_url="https://telemetry.example/",
        http_transport=httpx.MockTransport(handler),
    )

    await service._process_submission(  # pyright: ignore[reportPrivateUsage]
        RunnerStderrSubmission(path=log_path)
    )

    assert [r.method for r in requests] == ["POST", "PUT"]
    assert requests[0].url == "https://telemetry.example/telemetry/runner-log/presign"
    assert json.loads(requests[0].content) == {
        "sha256": hashlib.sha256(log_bytes).hexdigest(),
        "size": len(log_bytes),
    }
    assert requests[1].url == "https://uploads.example/runner.stderr.log"
    assert requests[1].content == log_bytes


@pytest.mark.anyio
async def test_runner_stderr_upload_failure_is_swallowed(tmp_path: Path):
    log_path = tmp_path / "runner.stderr.log"
    log_path.write_text("runner stderr\n")
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    service = TelemetryService.create(
        telemetry_disabled=False,
        api_url="https://telemetry.example",
        http_transport=httpx.MockTransport(handler),
    )
    _queue_submission(service, RunnerStderrSubmission(path=log_path))

    await service._process()  # pyright: ignore[reportPrivateUsage]

    assert len(requests) == 1
