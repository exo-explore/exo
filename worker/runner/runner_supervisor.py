import asyncio
import contextlib
import sys
from collections.abc import AsyncGenerator
from types import CoroutineType
from typing import Any, Callable

from shared.types.events import CommandId
from shared.types.events.chunks import GenerationChunk, TokenChunk
from shared.types.tasks import ChatCompletionTaskParams, Task
from shared.types.worker.commands_runner import (
    ChatTaskMessage,
    ErrorResponse,
    ExitMessage,
    FinishedResponse,
    GenerationResponse,
    PrintResponse,
    RunnerResponse,
    SetupMessage,
)
from shared.types.worker.mlx import Host
from shared.types.worker.shards import ShardMetadata
from worker.runner.communication import (
    supervisor_read_response,
    supervisor_write_message,
)
from worker.runner.utils import get_runner_command


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
    ):
        """Private constructor. Use RunnerSupervisor.create() instead."""
        self.model_shard_meta: ShardMetadata = model_shard_meta
        self.hosts: list[Host] = hosts
        self.runner_process: asyncio.subprocess.Process = runner_process
        self.running: bool = True

        self.running_task: asyncio.Task[None] = asyncio.create_task(
            self._watch_runner()
        )

    @classmethod
    async def create(
        cls,
        model_shard_meta: ShardMetadata,
        hosts: list[Host],
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
                stderr=sys.stderr,
            )
        )

        print(f'{model_shard_meta=}')
        await supervisor_write_message(
            runner_process,
            SetupMessage(
                model_shard_meta=model_shard_meta,
                hosts=hosts,
            ),
        )

        return cls(
            model_shard_meta=model_shard_meta,
            hosts=hosts,
            runner_process=runner_process,
        )

    async def astop(self) -> None:
        async def terminate() -> None:
            self.runner_process.terminate()
            _ = await self.runner_process.wait()

        if not self.healthy:
            print("Runner process is not healthy, killing...")
            await terminate()

        if self.runner_process.stdout is not None:
            while True:
                try:
                    line = await asyncio.wait_for(
                        self.runner_process.stdout.readline(), timeout=0.01
                    )
                    if not line:
                        break
                    print(f"Remaining stdout: {line.decode('utf-8').strip()}")
                except asyncio.TimeoutError:
                    break

        try:
            # Give the process a moment to exit gracefully
            await supervisor_write_message(
                proc=self.runner_process, message=ExitMessage()
            )
            _ = await asyncio.wait_for(self.runner_process.wait(), timeout=0.1)
        except asyncio.TimeoutError:
            print("Runner process did not terminate, killing...")
            await terminate()

        self.running = False

    async def _watch_runner(self) -> None:
        _ = await self.runner_process.wait()
        self.running = False

    def __del__(self) -> None:
        if not self.running:
            print(
                "Warning: RunnerSupervisor was not stopped cleanly before garbage collection. Force killing process."
            )

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
        request_started_callback: Callable[..., CoroutineType[Any, Any, None]] | None = None, # fyi this is async now
    ) -> AsyncGenerator[GenerationChunk]:
        """
        Streams a chat request from the model.
        The request is pushed to the runner, and if the shard is the terminal shard, the response is streamed back to the worker.
        request_started_callback is called once the request is pushed to the runner, used to publish InferencePrepareCompleted and InferenceTriggerCompleted events.
        """
        if not self.healthy:
            raise RuntimeError("Runner process was found to be dead")

        task_params = task.task_params
        assert isinstance(task_params, ChatCompletionTaskParams) # this is messy for now.
        await supervisor_write_message(
            proc=self.runner_process,
            message=ChatTaskMessage(
                task_data=task_params,
            ),
        )

        # This is easy for now. If we need more reliability, the runner can have a new 'ready' message type.
        if request_started_callback is not None:
            await request_started_callback()


        while True:
            line: RunnerResponse | None = await supervisor_read_response(
                self.runner_process
            )
            if line is None:
                continue
            else:
                match line:
                    case GenerationResponse(
                        text=text, token=token, finish_reason=finish_reason
                    ):
                        yield TokenChunk(
                            command_id=CommandId(task.task_id),
                            idx=token,
                            model=self.model_shard_meta.model_meta.model_id,
                            text=text,
                            token_id=token,
                            finish_reason=finish_reason,
                        )
                    case FinishedResponse():
                        break
                    case PrintResponse(text=text):
                        print(f"runner printed: {text}")
                    case ErrorResponse(
                        error_type=error_type,
                        error_message=error_message,
                        traceback=traceback,
                    ):
                        await self.astop()
                        raise Exception(error_type, error_message, traceback or "")
