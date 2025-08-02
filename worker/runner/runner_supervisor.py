import asyncio
import contextlib
import traceback
from collections.abc import AsyncGenerator
from logging import Logger
from types import CoroutineType
from typing import Any, Callable, Optional

import psutil

from shared.types.common import CommandId, Host
from shared.types.events.chunks import GenerationChunk, TokenChunk
from shared.types.tasks import ChatCompletionTaskParams, Task
from shared.types.worker.commands_runner import (
    ChatTaskMessage,
    ErrorResponse,
    FinishedResponse,
    GenerationResponse,
    InitializedResponse,
    PrintResponse,
    RunnerResponse,
    SetupMessage,
)
from shared.types.worker.common import RunnerError
from shared.types.worker.shards import ShardMetadata
from worker.runner.communication import (
    supervisor_read_response,
    supervisor_write_message,
)
from worker.runner.utils import (
    get_init_timeout,
    get_prefil_timeout,
    get_runner_command,
    get_token_generate_timeout,
    get_weights_size_kb,
)


class RunnerSupervisor:
    """
    RunnerSupervisor manages the lifecycle of a runner subprocess for model inference.
    Use the class method `create` to properly initialize an instance.
    """
    # TODO: Logger.

    def __init__(
        self,
        model_shard_meta: ShardMetadata,
        hosts: list[Host],
        runner_process: asyncio.subprocess.Process,
        logger: Logger,
    ):
        """Private constructor. Use RunnerSupervisor.create() instead."""
        self.model_shard_meta: ShardMetadata = model_shard_meta
        self.hosts: list[Host] = hosts
        self.runner_process: asyncio.subprocess.Process = runner_process
        self.running: bool = True
        self.stderr_task = asyncio.create_task(self._watch_stderr(logger))
        self.running_task: asyncio.Task[None] = asyncio.create_task(
            self._watch_runner()
        )
        self.logger = logger
        self.stderr_buffer: list[str] = []  # Accumulate stderr lines
        self.crash_detected: bool = False
        self.returncode: int | None = None
        self.stderr_outpu: str | None = None

    @classmethod
    async def create(
        cls,
        model_shard_meta: ShardMetadata,
        hosts: list[Host],
        logger: Logger,
        initialize_timeout: Optional[float] = None,
    ) -> "RunnerSupervisor":
        """
        Create and initialize a RunnerSupervisor instance.
        The .create() classmethod pattern is used to ensure the constructor is asynchronous.
        """
        cmd: list[str] = get_runner_command()
        runner_process: asyncio.subprocess.Process = (
            await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        )
        logger.info(f'initializing mlx instance with {model_shard_meta=}')
        await supervisor_write_message(
            runner_process,
            SetupMessage(
                model_shard_meta=model_shard_meta,
                hosts=hosts,
            ),
        )

        async def read_initialization_message() -> None:
            while True:
                line: RunnerResponse | None = await supervisor_read_response(
                    runner_process
                )
                if line is None:
                    continue
                elif isinstance(line, PrintResponse):
                    logger.info(line)
                    continue
                elif isinstance(line, ErrorResponse):
                    raise RunnerError(line.error_type, line.error_message, line.traceback or "")
                elif isinstance(line, InitializedResponse):
                    assert isinstance(line, InitializedResponse)
                    logger.info(f'Runner initialized in {line.time_taken} seconds')
                    break
                else:
                    raise AssertionError(f'Non-valid line read from runner during initialization: {line}')

        if not initialize_timeout:
            initialize_timeout = get_init_timeout(model_shard_meta)
        await asyncio.wait_for(read_initialization_message(), timeout=initialize_timeout)
        return cls(
            model_shard_meta=model_shard_meta,
            hosts=hosts,
            runner_process=runner_process,
            logger=logger,
        )

    async def astop(self) -> None:
        # Cancel the stderr monitoring task
        if not self.stderr_task.done():
            self.stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.stderr_task

        # Kill the process and all its children
        await self._kill_process_tree()
        
        # Wait to make sure that the model has been unloaded from memory
        async def wait_for_memory_release() -> None:
            required_memory_bytes = get_weights_size_kb(self.model_shard_meta) * 1024
            start_time = asyncio.get_event_loop().time()
            while True:
                available_memory_bytes = psutil.virtual_memory().available
                if available_memory_bytes >= required_memory_bytes:
                    break
                if asyncio.get_event_loop().time() - start_time > 30.0:
                    self.logger.warning("Timeout waiting for memory release after 30 seconds")
                    break
                await asyncio.sleep(0.1)

        await wait_for_memory_release()
        self.running = False

    async def _kill_process_tree(self) -> None:
        """Kill the process and all its children forcefully."""
        if self.runner_process.returncode is not None:
            return  # Process already dead
        
        try:
            # Get the main process
            pid = self.runner_process.pid
                
            # Find all child processes
            try:
                parent = psutil.Process(pid)
                children = parent.children(recursive=True)
                
                # Kill all children first (bottom-up)
                for child in reversed(children):
                    with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                        child.kill()  # SIGKILL
                
                # Kill the parent
                with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                    parent.kill()  # SIGKILL
                    
            except psutil.NoSuchProcess:
                # Process already gone, try subprocess kill anyway
                self.runner_process.kill()
            
            # Wait for the subprocess to exit
            try:
                await asyncio.wait_for(self.runner_process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                self.logger.error(f"Process {pid} did not exit after kill signal")
                
        except Exception as e:
            self.logger.error(f"Error killing process tree: {e}")

    async def _watch_runner(self) -> None:
        returncode = await self.runner_process.wait()
        self.running = False
        if returncode != 0:
            self.crash_detected = True
            self.returncode = returncode  # Will be picked up by _watch_stderr too

    async def _watch_stderr(self, logger: Logger) -> None:
        assert self.runner_process.stderr is not None
        while self.running:
            try:
                line_bytes = await self.runner_process.stderr.readline()
                if not line_bytes:
                    break  # EOF
                line = line_bytes.decode('utf-8').strip()
                self.stderr_buffer.append(line)
                logger.error(f"Runner stderr: {line}")
                # Detect common crash patterns (extend as needed, e.g., for OOM: "Killed" or "Out of memory")

                self.crash_detected = True
                self.stderr_output = "\n".join(self.stderr_buffer)
                logger.critical(f"Runner crash detected: {self.stderr_output}")
                # Don't raise here—let callers (e.g., stream_response) detect via healthy/returncode
            except Exception as e:
                logger.error(f"Error reading runner stderr: {e}")
                break

        # After EOF, inspect returncode for confirmation (Unix-like: negative == signal)
        returncode = self.runner_process.returncode
        if returncode is not None and returncode != 0:
            self.crash_detected = True
            self.returncode = returncode
            self.stderr_output = "\n".join(self.stderr_buffer)

    def _raise_if_crashed(self) -> None:
        if self.crash_detected:
            self.logger.error(f'Error {self.returncode}: {self.stderr_output}')
            raise RunnerError(
                error_type="RunnerCrash",
                error_message=self.stderr_output,
                traceback=traceback.format_exc(),
            )

    def __del__(self) -> None:
        if self.running:
            print(
                "Warning: RunnerSupervisor was not stopped cleanly before garbage collection. Force killing process tree."
            )
            # Can't use async in __del__, so use psutil directly
            try:
                pid = self.runner_process.pid
                if pid:
                    parent = psutil.Process(pid)
                    children = parent.children(recursive=True)
                    for child in reversed(children):
                        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                            child.kill()
                    with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                        parent.kill()
            except Exception:
                with contextlib.suppress(ProcessLookupError):
                    self.runner_process.kill()

    @property
    def healthy(self) -> bool:
        return (
            self.running
            and self.runner_process.returncode is None
            and self.runner_process.stdin is not None
            and not self.runner_process.stdin.is_closing()
            and self.runner_process.stdout is not None
        )

    async def stream_response(
        self,
        task: Task,
        request_started_callback: Callable[..., CoroutineType[Any, Any, None]] | None = None,  # fyi this is async now
    ) -> AsyncGenerator[GenerationChunk]:
        """
        Streams a chat request from the model.
        The request is pushed to the runner, and if the shard is the terminal shard, the response is streamed back to the worker.
        request_started_callback is called once the request is pushed to the runner, used to publish InferencePrepareCompleted and InferenceTriggerCompleted events.
        """
        if not self.healthy:
            raise RuntimeError("Runner process was found to be dead")
        task_params = task.task_params
        assert isinstance(task_params, ChatCompletionTaskParams)  # this is messy for now.
        await supervisor_write_message(
            proc=self.runner_process,
            message=ChatTaskMessage(
                task_data=task_params,
            ),
        )
        # This is easy for now. If we need more reliability, the runner can have a new 'ready' message type.
        if request_started_callback is not None:
            await request_started_callback()
        prefil_timeout = get_prefil_timeout(self.model_shard_meta)
        token_timeout = get_token_generate_timeout(self.model_shard_meta)
        timeout = prefil_timeout
        while True:
            try:
                line: RunnerResponse | None = await asyncio.wait_for(supervisor_read_response(
                    self.runner_process
                ), timeout=timeout)
                if line is None:
                    continue
            except (asyncio.TimeoutError, EOFError) as e:
                self._raise_if_crashed()
                raise RunnerError(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    traceback="",
                ) from e
            match line:
                case GenerationResponse():
                    yield TokenChunk(
                        command_id=CommandId(task.command_id),
                        idx=line.token,
                        model=self.model_shard_meta.model_meta.model_id,
                        text=line.text,
                        token_id=line.token,
                        finish_reason=line.finish_reason,
                    )
                    timeout = token_timeout
                case InitializedResponse():
                    raise ValueError('Initialized Response read during streaming flow')
                case FinishedResponse():
                    break
                case PrintResponse():
                    # print(f"runner printed: {line.text}")
                    self.logger.info(f"runner printed: {line.text}")
                case ErrorResponse():
                    await self.astop()
                    raise RunnerError(line.error_type, line.error_message, line.traceback or "")