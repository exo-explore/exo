import contextlib
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self
from urllib.parse import urlparse

import httpx
from anyio import BrokenResourceError, ClosedResourceError, WouldBlock, to_thread
from loguru import logger

from exo.shared.constants import EXO_TELEMETRY_API_URL
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.pydantic_ext import FrozenModel, TaggedModel
from exo.utils.task_group import TaskGroup

CHANNEL_BOUND_SIZE = 64
TELEMETRY_HTTP_TIMEOUT_SECONDS = 10.0


class BaseTelemetrySubmission(TaggedModel):
    pass


class TestSubmission(BaseTelemetrySubmission):
    pass


class RunnerStderrSubmission(BaseTelemetrySubmission):
    path: Path


TelemetrySubmission = TestSubmission | RunnerStderrSubmission


class TelemetryPresignResponse(FrozenModel):
    key: str
    upload_url: str
    expires_in: int
    max_size: int


@dataclass(eq=False)
class TelemetrySink:
    """
    A non-blocking non-throwing bounded wrapper around sender/receiver channels
    to ensure telemetry never blocks or has adverse side-effects, since telemetry
    is an optional diagnostic feature and hence should never break the main app.
    """

    _send: Sender[TelemetrySubmission]

    @classmethod
    def pair(cls) -> tuple[Self, Receiver[TelemetrySubmission]]:
        send, recv = channel[TelemetrySubmission](CHANNEL_BOUND_SIZE)
        return cls(_send=send), recv

    def submit(self, submission: TelemetrySubmission):
        try:
            self._send.send_nowait(submission)
        except WouldBlock:
            logger.debug("Telemetry submission would block. why so many submissions??")
        except (BrokenResourceError, ClosedResourceError):
            logger.debug("Telemetry submission receivers are broken or closed. why??")

    def clone(self) -> "TelemetrySink":
        return TelemetrySink(_send=self._send.clone())

    def close(self):
        with contextlib.suppress(BrokenResourceError, ClosedResourceError):
            self._send.close()


@dataclass(eq=False)
class TelemetryService:
    dry_run: bool
    api_url: str
    _send: Sender[TelemetrySubmission]
    _recv: Receiver[TelemetrySubmission]
    _http_transport: httpx.AsyncBaseTransport | None
    _tg: TaskGroup = field(default_factory=TaskGroup, init=False)

    @classmethod
    def create(
        cls,
        dry_run: bool,
        api_url: str = EXO_TELEMETRY_API_URL,
        http_transport: httpx.AsyncBaseTransport | None = None,
    ) -> Self:
        api_url = urlparse(api_url).geturl().rstrip("/")

        send, recv = channel[TelemetrySubmission](CHANNEL_BOUND_SIZE)

        return cls(
            dry_run=dry_run,
            api_url=api_url,
            _send=send,
            _recv=recv,
            _http_transport=http_transport,
        )

    @classmethod
    def dummy(cls) -> Self:
        return cls.create(True)

    async def run(self):
        try:
            async with self._tg as tg:
                tg.start_soon(self._process)
        finally:
            self._send.close()
            self._recv.close()

    async def _process(self):
        with self._recv as submissions:
            async for submission in submissions:
                if not self.dry_run:
                    try:
                        await self._process_submission(submission)
                    except Exception as e:
                        logger.opt(exception=e).warning(
                            "Exception when processing telemetry submission"
                        )

    async def _process_submission(self, submission: TelemetrySubmission):
        match submission:
            case TestSubmission():
                pass
            case RunnerStderrSubmission(path=path):
                await self._submit_runner_stderr(path)

    async def _submit_runner_stderr(self, path: Path):
        data = await to_thread.run_sync(path.read_bytes)
        if not data:
            logger.debug(f"Skipping empty runner stderr telemetry file: {path}")
            return

        sha256 = hashlib.sha256(data).hexdigest()

        async with httpx.AsyncClient(
            timeout=TELEMETRY_HTTP_TIMEOUT_SECONDS,
            transport=self._http_transport,
        ) as client:
            presign_response = await client.post(
                f"{self.api_url}/telemetry/runner-log/presign",
                json={
                    "sha256": sha256,
                    "size": len(data),
                },
            )
            presign_response.raise_for_status()
            presign = TelemetryPresignResponse.model_validate_json(
                presign_response.text,
            )

            upload_response = await client.put(
                presign.upload_url,
                content=data,
            )
            upload_response.raise_for_status()

    def sink(self) -> TelemetrySink:
        sink, recv = TelemetrySink.pair()
        if self._tg.is_running():
            self._tg.start_soon(self._ingest, recv)
        else:
            self._tg.queue(self._ingest, recv)
        return sink

    async def _ingest(self, recv: Receiver[TelemetrySubmission]):
        try:
            with recv as submissions:
                async for submission in submissions:
                    await self._send.send(submission)
        except ClosedResourceError:
            pass
