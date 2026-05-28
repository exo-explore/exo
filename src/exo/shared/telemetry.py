from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

from anyio import BrokenResourceError, ClosedResourceError, WouldBlock
from loguru import logger

from exo.utils.channels import Receiver, Sender, channel
from exo.utils.pydantic_ext import TaggedModel
from exo.utils.task_group import TaskGroup

CHANNEL_BOUND_SIZE = 64


class BaseTelemetrySubmission(TaggedModel):
    pass


class TestSubmission(BaseTelemetrySubmission):
    pass


class RunnerStderrSubmission(BaseTelemetrySubmission):
    path: Path


TelemetrySubmission = TestSubmission | RunnerStderrSubmission


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


@dataclass(eq=False)
class TelemetryService:
    _send: Sender[TelemetrySubmission]
    _recv: Receiver[TelemetrySubmission]
    _tg: TaskGroup = field(default_factory=TaskGroup, init=False)

    @classmethod
    def create(cls) -> Self:
        send, recv = channel[TelemetrySubmission](CHANNEL_BOUND_SIZE)

        return cls(
            _send=send,
            _recv=recv,
        )

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
            case RunnerStderrSubmission():
                pass

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
