import asyncio
import contextlib
import traceback
from collections.abc import AsyncGenerator
from types import CoroutineType
from typing import Any, Callable, Optional

import psutil
from loguru import logger

from exo.shared.types.common import CommandId, Host
from exo.shared.types.events.chunks import GenerationChunk, TokenChunk
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
from exo.shared.types.worker.communication import (
    supervisor_read_response,
    supervisor_write_message,
)
from exo.shared.types.worker.shards import ShardMetadata
from exo.worker.runner.utils import (
    get_init_timeout,
    get_prefil_timeout,
    get_runner_command,
    get_token_generate_timeout,
    get_weights_size_kb,
    kill_process_tree,
)


class RunnerSupervisor:
    def __init__(
        self,
        model_shard_meta: ShardMetadata,
        hosts: list[Host],
        runner_process: asyncio.subprocess.Process,
        read_queue: asyncio.Queue[RunnerResponse],
        write_queue: asyncio.Queue[RunnerMessage],
        stderr_queue: asyncio.Queue[str],
    ):
        self.model_shard_meta = model_shard_meta
        self.hosts = hosts
        self.runner_process = runner_process

        self.read_queue = read_queue
        self.write_queue = write_queue
        self.stderr_queue = stderr_queue

        self.read_task = asyncio.create_task(self._read_coro())
        self.write_task = asyncio.create_task(self._write_coro())
        self.stderr_task = asyncio.create_task(self._watch_stderr())

    @classmethod
    async def create(
        cls,
        model_shard_meta: ShardMetadata,
        hosts: list[Host],
        initialize_timeout: Optional[float] = None,
    ) -> "RunnerSupervisor":
        """
        Create and initialize a RunnerSupervisor instance.
        The .create() classmethod pattern is used to ensure the constructor is asynchronous.
        """
        cmd: list[str] = get_runner_command()
        runner_process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        read_queue: asyncio.Queue[RunnerResponse] = asyncio.Queue()
        write_queue: asyncio.Queue[RunnerMessage] = asyncio.Queue()
        stderr_queue: asyncio.Queue[str] = asyncio.Queue()

        self = cls(
            model_shard_meta=model_shard_meta,
            hosts=hosts,
            runner_process=runner_process,
            read_queue=read_queue,
            write_queue=write_queue,
            stderr_queue=stderr_queue,
        )

        logger.info(f"Initializing mlx instance with {model_shard_meta=}")
        await self.write_queue.put(
            SetupMessage(
                model_shard_meta=model_shard_meta,
                hosts=hosts,
            )
        )

        if not initialize_timeout:
            initialize_timeout = get_init_timeout(model_shard_meta)

        response = await self._read_with_error_check(initialize_timeout)

        assert isinstance(response, InitializedResponse)
        logger.info(f"Runner initialized in {response.time_taken} seconds")

        return self

    async def _read_with_error_check(self, timeout: float) -> RunnerResponse:
        """
        Read from the queue with a timeout, but also check if the read_task has failed.
        """
        try:
            assert not self.read_task.done()        
        except AssertionError as e_assert:
            e = self.read_task.exception()
            assert e is not None
            raise e from e_assert

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
            response = await queue_task
            if isinstance(response, ErrorResponse):
                await self.astop()
                raise RunnerError(
                    response.error_type,
                    response.error_message,
                    response.traceback or "",
                )
            return response

        if self.read_task in done:
            try:
                await self.read_task  # Re-raises any exception from read_task
            except Exception:
                raise # bubble up exception
            raise RunnerError("RunnerStopped", "Runner read loop terminated unexpectedly before any response.", "")
        
        # if we haven't read from the queue, we have timed out.
        await self.astop() # TODO: This could be handled by the called or _read_with_error_check - as we don't want a false Timeout to bring the whole runner down.
        raise asyncio.TimeoutError()

    async def stream_response(
        self,
        task: Task,
        request_started_callback: Callable[..., CoroutineType[Any, Any, None]]
        | None = None,
    ) -> AsyncGenerator[GenerationChunk]:
        """
        Streams a chat request from the model.
        The request is pushed to the runner, and if the shard is the terminal shard, the response is streamed back to the worker.
        request_started_callback is called once the request is pushed to the runner, used to publish InferencePrepareCompleted and InferenceTriggerCompleted events.
        """
        if not self.healthy:
            raise RuntimeError("Runner process was found to be dead")

        task_params = task.task_params
        assert isinstance(
            task_params, ChatCompletionTaskParams
        )  # this is messy for now.
        await self.write_queue.put(
            ChatTaskMessage(
                task_data=task_params,
            ),
        )

        while True:
            try:
                response = await self._read_with_error_check(5.0)
            except asyncio.TimeoutError as e:
                logger.bind(user_facing=True).error(
                    "Generation timed out during tokenization"
                )
                raise e
            except asyncio.LimitOverrunError as e:
                raise RunnerError(
                    "IPCMessageTooLarge",
                    "The serialized prompt/response exceeded the IPC line limit. Switch to length-prefixed framing or reduce prompt size.",
                    ""
                ) from e


            match response:
                case TokenizedResponse():
                    prompt_tokens = response.prompt_tokens
                    break
                case ErrorResponse():
                    await self.astop()
                    raise RunnerError(
                        response.error_type, response.error_message, response.traceback
                    )
                case _:
                    raise ValueError(f"Unexpected response type found: {response}")

        if request_started_callback is not None:
            await request_started_callback()

        prefil_timeout = get_prefil_timeout(self.model_shard_meta, prompt_tokens=prompt_tokens)
        token_timeout = get_token_generate_timeout(self.model_shard_meta)
        timeout = prefil_timeout
        logger.bind(user_facing=True).info(
            f"Starting chat completion with timeout {timeout}"
        )

        while True:
            try:
                response = await self._read_with_error_check(timeout)
            except asyncio.TimeoutError as e:
                logger.bind(user_facing=True).error(
                    f"Generation timed out during {'prefil' if timeout == prefil_timeout else 'decoding stage'}"
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
                    timeout = token_timeout
                case FinishedResponse():
                    break
                case ErrorResponse():
                    await self.astop()
                    raise RunnerError(
                        response.error_type, response.error_message, response.traceback
                    )
                case _:
                    raise ValueError(f"Unexpected response type found: {response}")

    async def _write_coro(self):
        while True:
            message = await self.write_queue.get()
            await supervisor_write_message(self.runner_process, message)

    async def _read_coro(self):
        while True:
            try:
                response: RunnerResponse = await supervisor_read_response(
                    self.runner_process
                )
            except EOFError:
                e = await self._raise_crashed()
                if e:
                    # Runner process died unexpectedly (C++ crash)
                    raise e from EOFError # TODO: Do we just want to create an error and put it on the read_queue here?
                else:
                    continue

            match response:
                case PrintResponse():
                    # TODO: THIS IS A REALLY IMPORTANT LOG MESSAGE, AND SHOULD BE MADE PRETTIER
                    logger.bind(user_facing=True).info(f"{response.text}")
                case ErrorResponse():
                    ## Failure case #1: a crash happens Python, so it's neatly handled by passing an ErrorResponse with the details
                    await self.read_queue.put(response)
                case _:
                    await self.read_queue.put(response)

    async def astop(self) -> None:
        # Cancel the stderr monitoring task
        async def await_task(task: asyncio.Task[Any]):
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        await await_task(self.stderr_task)
        await await_task(self.read_task)
        await await_task(self.write_task)

        # Kill the process and all its children
        await kill_process_tree(self.runner_process)

        # Wait to make sure that the model has been unloaded from memory
        async def wait_for_memory_release() -> None:
            required_memory_bytes = get_weights_size_kb(self.model_shard_meta) * 1024
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
        if self.runner_process.returncode is None:
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

    @property
    def healthy(self) -> bool:
        return (
            self.runner_process.returncode is None
            and self.runner_process.stdin is not None
            and not self.runner_process.stdin.is_closing()
            and self.runner_process.stdout is not None
        )

    ## Failure case #2: a crash happens in MLX / C++ (eg segfault) that leads to error flushed to stderr and process dies
    async def _raise_crashed(self) -> Exception | None:
        if self.runner_process.returncode == 0:
            return None

        await self.astop()

        # Accumulate all stderr messages from the queue
        stderr_output = ""
        while not self.stderr_queue.empty():
            try:
                line = self.stderr_queue.get_nowait()
                stderr_output += f"{line}\n"
            except asyncio.QueueEmpty:
                break

        logger.bind(user_facing=True).error(
            f"Runner Error {self.runner_process.returncode}: {stderr_output}"
        )
        return RunnerError(
            error_type="MLXCrash",
            error_message=stderr_output,
            traceback=traceback.format_exc(),
        )

    async def _watch_stderr(self) -> None:
        assert self.runner_process.stderr is not None
        while True:
            try:
                line_bytes = await self.runner_process.stderr.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8").strip()

                await self.stderr_queue.put(line)
                logger.warning(f"Runner stderr read: {line}")
            except Exception as e:
                logger.warning(f"Error reading runner stderr: {e}")
                break
