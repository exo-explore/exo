import asyncio
import contextlib
import multiprocessing as mp
import os
import signal
import tempfile
import traceback
from multiprocessing import Process
from multiprocessing.connection import Connection
from typing import Any, AsyncGenerator, Callable, Coroutine, Optional

import psutil
from loguru import logger

from exo.shared.global_conn import (
    AsyncConnection,
)
from exo.shared.types.chunks import GenerationChunk, TokenChunk
from exo.shared.types.common import CommandId, Host
from exo.shared.types.tasks import ChatCompletionTaskParams, Task
from exo.shared.types.worker.commands_runner import (
    ChatTaskMessage,
    ErrorResponse,
    FinishedResponse,
    GenerationResponse,
    InitializedResponse,
    PrintResponse,
    RunnerMessage,
    RunnerResponse,
    SetupMessage,
    TokenizedResponse,
)
from exo.shared.types.worker.common import RunnerError
from exo.shared.types.worker.shards import ShardMetadata
from exo.worker.runner.bootstrap import entrypoint
from exo.worker.runner.utils import (
    get_weights_size,
)

INITIALIZE_TIMEOUT = 400
PREFILL_TIMEOUT_SECONDS = 60
DECODE_TIMEOUT_SECONDS = 5


class RunnerSupervisor:
    def __init__(
        self,
        model_shard_meta: ShardMetadata,
        hosts: list[Host] | None,
        mlx_ibv_devices: list[list[str | None]] | None,
        mlx_ibv_coordinator: str | None,
        runner_process: Process,
        conn: Connection,
        read_queue: asyncio.Queue[RunnerResponse],
        err_path: str,
    ):
        self.model_shard_meta = model_shard_meta
        self.hosts = hosts
        self.mlx_ibv_devices = mlx_ibv_devices
        self.mlx_ibv_coordinator = mlx_ibv_coordinator
        self.runner_process = runner_process

        self.conn = AsyncConnection[RunnerMessage, RunnerResponse](conn)
        self._raw_conn = conn

        self.read_queue = read_queue
        self.read_task = asyncio.create_task(self._read_coro())

        self.err_path = err_path

    @classmethod
    async def create(
        cls,
        model_shard_meta: ShardMetadata,
        hosts: list[Host] | None = None,
        mlx_ibv_devices: list[list[str | None]] | None = None,
        mlx_ibv_coordinator: str | None = None,
        initialize_timeout: Optional[float] = None,
    ) -> "RunnerSupervisor":
        """
        Create and initialize a RunnerSupervisor instance.
        The .create() classmethod pattern is used to ensure the constructor is asynchronous.
        """
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=True)

        with tempfile.NamedTemporaryFile(
            prefix="child_stderr_", suffix=".log", delete=False
        ) as tmp:
            err_path = tmp.name

        runner_process = Process(
            target=entrypoint, args=(child_conn, err_path), daemon=False
        )
        runner_process.start()
        child_conn.close()

        read_queue = asyncio.Queue[RunnerResponse]()

        self = cls(
            model_shard_meta=model_shard_meta,
            hosts=hosts,
            mlx_ibv_devices=mlx_ibv_devices,
            mlx_ibv_coordinator=mlx_ibv_coordinator,
            runner_process=runner_process,
            read_queue=read_queue,
            conn=parent_conn,
            err_path=err_path,
        )

        logger.info(f"Initializing mlx instance with {model_shard_meta=}")
        await self.conn.send(
            SetupMessage(
                model_shard_meta=model_shard_meta,
                hosts=hosts,
                mlx_ibv_devices=mlx_ibv_devices,
                mlx_ibv_coordinator=mlx_ibv_coordinator,
            )
        )

        initialize_timeout = initialize_timeout or INITIALIZE_TIMEOUT
        response = await self._read_with_error_check(timeout=initialize_timeout)

        assert isinstance(response, InitializedResponse)
        logger.info(f"Runner initialized in {response.time_taken} seconds")

        return self

    async def _read_with_error_check(self, timeout: float) -> RunnerResponse | None:
        """
        Read from the queue with a timeout, but also check if the read_task has failed.
        """
        if self.read_task.done():
            e = self.read_task.exception()
            await self.astop()
            if e is not None:
                raise e
            else:
                return None

        queue_task = asyncio.create_task(self.read_queue.get())

        done, pending = await asyncio.wait(
            [queue_task, self.read_task],
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            if task is queue_task:
                task.cancel()

        if queue_task in done:
            return await queue_task

        if self.read_task in done:
            await self.astop()
            await self.read_task  # Re-raises any exception from read_task

            # This should never get hit.
            raise RunnerError(
                "RunnerStopped",
                "Runner read loop terminated unexpectedly before any response.",
                "",
            )

        # if we haven't read from the queue, we have timed out.
        await self.astop()  # TODO: This could be handled by the called or _read_with_error_check - as we don't want a false Timeout to bring the whole runner down.
        raise asyncio.TimeoutError()

    async def _read_coro(self):
        while True:
            try:
                response: RunnerResponse = await self.conn.recv()
            except EOFError as e_eof:
                e = await self._raise_crashed()
                if e is not None:
                    raise e from e_eof
                break

            match response:
                case PrintResponse():
                    # TODO: THIS IS A REALLY IMPORTANT LOG MESSAGE, AND SHOULD BE MADE PRETTIER
                    logger.info(f"{response.text}")
                case ErrorResponse():
                    raise RunnerError(
                        response.error_type, response.error_message, response.traceback
                    )
                case _:
                    await self.read_queue.put(response)

    async def stream_response(
        self,
        task: Task,
        request_started_callback: Callable[..., Coroutine[Any, Any, None]]
        | None = None,
    ) -> AsyncGenerator[GenerationChunk, None]:
        """
        Streams a chat request from the model.
        The request is pushed to the runner, and if the shard is the terminal shard, the response is streamed back to the worker.
        request_started_callback is called once the request is pushed to the runner, used to publish InferencePrepareCompleted and InferenceTriggerCompleted events.
        """
        if not self.runner_process.is_alive():
            raise RuntimeError("Runner process was found to be dead")

        task_params = task.task_params
        assert isinstance(
            task_params, ChatCompletionTaskParams
        )  # this is messy for now.
        await self.conn.send(
            ChatTaskMessage(
                task_data=task_params,
            ),
        )

        response = await self._read_with_error_check(5.0)
        assert isinstance(response, TokenizedResponse)

        if request_started_callback is not None:
            await request_started_callback()

        timeout = PREFILL_TIMEOUT_SECONDS

        logger.info(
            f"Starting chat completion with timeout {timeout}"
        )

        while True:
            try:
                response = await self._read_with_error_check(timeout)
            except asyncio.TimeoutError as e:
                logger.error(
                    f"Generation timed out during {'prefill' if timeout == PREFILL_TIMEOUT_SECONDS else 'decoding stage'}"
                )
                raise e

            match response:
                case GenerationResponse():
                    yield TokenChunk(
                        command_id=CommandId(task.command_id),
                        idx=response.token,
                        model=self.model_shard_meta.model_meta.model_id,
                        text=response.text,
                        token_id=response.token,
                        finish_reason=response.finish_reason,
                    )
                    timeout = DECODE_TIMEOUT_SECONDS
                case FinishedResponse():
                    break
                case _:
                    raise ValueError(f"Unexpected response type found: {response}")

    async def astop(self) -> None:
        # Cancel the stderr monitoring task
        async def await_task(task: asyncio.Task[Any]):
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        await await_task(self.read_task)

        self.runner_process.kill()

        with contextlib.suppress(Exception):
            self._raw_conn.close()

        # Wait to make sure that the model has been unloaded from memory
        async def wait_for_memory_release() -> None:
            required_memory_bytes = get_weights_size(self.model_shard_meta).in_bytes
            start_time = asyncio.get_event_loop().time()
            while True:
                available_memory_bytes = psutil.virtual_memory().available
                if available_memory_bytes >= required_memory_bytes:
                    break
                if asyncio.get_event_loop().time() - start_time > 30.0:
                    logger.warning(
                        "Runner memory not released after 30 seconds - exiting"
                    )
                    break
                await asyncio.sleep(0.1)

        await wait_for_memory_release()

    def __del__(self) -> None:
        if self.runner_process.is_alive():
            logger.warning(
                "RunnerSupervisor was not stopped cleanly before garbage collection. Force killing process tree."
            )
            # Can't use async in __del__, so use psutil directly
            try:
                pid = self.runner_process.pid
                if pid:
                    parent = psutil.Process(pid)
                    children = parent.children(recursive=True)
                    for child in reversed(children):
                        with contextlib.suppress(
                            psutil.NoSuchProcess, psutil.AccessDenied
                        ):
                            child.kill()
                    with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                        parent.kill()
            except Exception:
                with contextlib.suppress(ProcessLookupError):
                    self.runner_process.kill()

    async def _raise_crashed(self) -> Exception | None:
        await asyncio.sleep(0.1)

        rc = self.runner_process.exitcode
        if rc == 0:
            return None

        try:
            with open(self.err_path, "r", errors="replace") as f:
                captured = f.read()
        finally:
            with contextlib.suppress(OSError):
                os.unlink(self.err_path)

        # 2) Describe cause (signal vs exitcode)
        cause = f"exitcode={rc}"
        if isinstance(rc, int) and rc < 0:
            sig = -rc
            try:
                cause = f"signal={sig} ({signal.strsignal(sig)})"
            except Exception:
                cause = f"signal={sig}"

        logger.error(f"Runner terminated ({cause}).\n{captured}")

        return RunnerError(
            error_type="RunnerCrash",
            error_message=f"Runner terminated ({cause}).\n{captured}",
            traceback=traceback.format_exc(),
        )
