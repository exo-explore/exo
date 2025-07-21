import asyncio
import os
from asyncio.queues import Queue
from functools import partial
from logging import Logger
from typing import AsyncGenerator, Optional

from pydantic import BaseModel, ConfigDict

from shared.types.common import NodeId
from shared.types.events.events import ChunkGenerated, InstanceId, RunnerStatusUpdated
from shared.types.events.registry import Event
from shared.types.state import State
from shared.types.worker.common import RunnerId
from shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadProgressData,
)
from shared.types.worker.mlx import Host
from shared.types.worker.ops import (
    AssignRunnerOp,
    DownloadOp,
    ExecuteTaskOp,
    RunnerDownOp,
    RunnerOp,
    RunnerOpType,
    RunnerUpOp,
    UnassignRunnerOp,
)
from shared.types.worker.runners import (
    DownloadingRunnerStatus,
    FailedRunnerStatus,
    LoadedRunnerStatus,
    ReadyRunnerStatus,
    RunnerStatus,
    RunnerStatusType,
    RunningRunnerStatus,
)
from shared.types.worker.shards import ShardMetadata
from worker.runner.runner_supervisor import RunnerSupervisor


class AssignedRunner(BaseModel):
    runner_id: RunnerId
    instance_id: InstanceId
    shard_metadata: ShardMetadata  # just data
    hosts: list[Host]

    status: RunnerStatus
    runner: Optional[RunnerSupervisor]  # set if the runner is 'up'

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def is_downloaded(self) -> bool:
        # TODO: Do this properly with huggingface validating each of the files.
        return os.path.exists(self.shard_metadata.model_path)

    def status_update_event(self) -> RunnerStatusUpdated:
        return RunnerStatusUpdated(
            runner_id=self.runner_id,
            runner_status=self.status,
        )

class Worker:
    def __init__(
        self,
        node_id: NodeId,
        initial_state: State,
        logger: Logger,
    ):
        self.node_id = node_id
        self.state = initial_state
        self.logger = logger

        self.assigned_runners: dict[RunnerId, AssignedRunner] = {}
        self._task: asyncio.Task[None] | None = None

    ## Worker lifecycle management
    @property
    def _is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self):
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        if not self._is_running:
            raise RuntimeError("Worker is not running")
            
        assert self._task is not None        

        self._task.cancel()

    ## Op Executors

    async def _execute_assign_op(
        self, op: AssignRunnerOp
    ) -> AsyncGenerator[Event, None]:
        '''
        Here, we are sure that the model is already downloaded.
        This op moves the runner from Assigned -> Ready state.
        '''
        self.assigned_runners[op.runner_id] = AssignedRunner(
            runner_id=op.runner_id,
            instance_id=op.instance_id,
            shard_metadata=op.shard_metadata,
            hosts=op.hosts,
            status=ReadyRunnerStatus(),
            runner=None,
        )

        yield self.assigned_runners[op.runner_id].status_update_event()

    async def _execute_unassign_op(
        self, op: UnassignRunnerOp
    ) -> AsyncGenerator[Event, None]:
        if op.runner_id not in self.assigned_runners:
            return        

        # We can try to do a graceful shutdown of the runner.
        runner: RunnerSupervisor | None = self.assigned_runners[op.runner_id].runner 
        if runner is not None:
            await runner.astop()
        
        # This is all we really need:
        del self.assigned_runners[op.runner_id]

        return
        yield

    async def _execute_runner_up_op(
        self, op: RunnerUpOp
    ) -> AsyncGenerator[Event, None]:
        assigned_runner = self.assigned_runners[op.runner_id]

        assigned_runner.runner = await RunnerSupervisor.create(
            model_shard_meta=assigned_runner.shard_metadata,
            hosts=assigned_runner.hosts,
        )

        if assigned_runner.runner.healthy:
            assigned_runner.status = LoadedRunnerStatus()
        else:
            assigned_runner.status = FailedRunnerStatus()
        yield self.assigned_runners[op.runner_id].status_update_event()

    async def _execute_runner_down_op(
        self, op: RunnerDownOp
    ) -> AsyncGenerator[Event, None]:
        assigned_runner = self.assigned_runners[op.runner_id]

        assert isinstance(assigned_runner.runner, RunnerSupervisor)
        await assigned_runner.runner.astop()
        assigned_runner.runner = None

        assigned_runner.status = ReadyRunnerStatus()
        yield assigned_runner.status_update_event()

    async def _execute_download_op(
        self, op: DownloadOp
    ) -> AsyncGenerator[Event, None]:
        '''
        The model needs assigning and then downloading.
        This op moves the runner from Assigned -> Downloading -> Ready state.
        '''
        initial_status = DownloadingRunnerStatus(
            download_progress=DownloadOngoing(
                node_id=self.node_id,
                download_progress=DownloadProgressData(
                    total_bytes=1, # tmp
                    downloaded_bytes=0
                )
            )
        )       

        self.assigned_runners[op.runner_id] = AssignedRunner(
            runner_id=op.runner_id,
            instance_id=op.instance_id,
            shard_metadata=op.shard_metadata,
            hosts=op.hosts,
            status=initial_status,
            runner=None,
        )
        assigned_runner: AssignedRunner = self.assigned_runners[op.runner_id]
        yield assigned_runner.status_update_event()

        # Download it!
        # TODO: we probably want download progress as part of a callback that gets passed to the downloader.        

        try:
            assert assigned_runner.is_downloaded
            assigned_runner.status = DownloadingRunnerStatus(
                download_progress=DownloadCompleted(
                    node_id=self.node_id,
                )
            )
        except Exception as e:
            assigned_runner.status = DownloadingRunnerStatus(
                download_progress=DownloadFailed(
                    node_id=self.node_id,
                    error_message=str(e)
                )
            )
        yield assigned_runner.status_update_event()

        assigned_runner.status = ReadyRunnerStatus()
        yield assigned_runner.status_update_event()

# Plan:
# First get a single inference running
# Then build boilerplate for passing callback when mlx is in the 'ready' state
# Then figure out if we can do what's needed with events. But this is a little challenging because it depends on Alex's code.
    async def _execute_chat_completion_op(
        self, op: ExecuteTaskOp
    ) -> AsyncGenerator[Event, None]:
        '''
        This is the entry point for a chat completion starting. 
        While there is only one execute function, it will get called in different ways for runner 0 and runner [1, 2, 3, ...].
        Runners [1, 2, 3, ...] will run this method when a task is in 'pending' state.
        Runner 0 will run this method when a task is in 'running' state.
        TODO: How do we handle the logic of ensuring that n-1 nodes have started their execution before allowing the 0'th runner to start?
        This is still a little unclear to me.
        '''
        assigned_runner = self.assigned_runners[op.runner_id]        

        async def inner_execute(queue: asyncio.Queue[Event]) -> None:
            assert assigned_runner.runner is not None
            assert assigned_runner.runner.healthy

            async def running_callback(queue: asyncio.Queue[Event]) -> None:
                # Called when the MLX process has been kicked off
                assigned_runner.status = RunningRunnerStatus()
                await queue.put(assigned_runner.status_update_event())
            
            try:
                async for chunk in assigned_runner.runner.stream_response(
                        task=op.task, 
                        request_started_callback=partial(running_callback, queue)):
                    await queue.put(ChunkGenerated(
                        task_id=op.task.task_id,
                        chunk=chunk
                    ))
        
                # After a successful inference:
                assigned_runner.status = LoadedRunnerStatus()
                await queue.put(assigned_runner.status_update_event())

            except Exception as e:
                # TODO: What log level?
                self.logger.log(2, f'Runner failed whilst running inference task. Task: {op.task}. Error: {e}')

                assigned_runner.runner = None
                assigned_runner.status = FailedRunnerStatus(error_message=str(e))
                await queue.put(assigned_runner.status_update_event())

        queue: Queue[Event] = asyncio.Queue()
        task = asyncio.create_task(inner_execute(queue))

        try:
            # Yield items from the queue
            while True:
                item: Event = await asyncio.wait_for(queue.get(), timeout=5)
                yield item
                if isinstance(item, RunnerStatusUpdated) and isinstance(
                    item.runner_status, (LoadedRunnerStatus, FailedRunnerStatus)
                ):
                    break
        finally:
            # Ensure the task is cleaned up
            await task


    ## Operation Planner

    async def _execute_op(self, op: RunnerOp) -> AsyncGenerator[Event, None]:
        ## It would be great if we can get rid of this async for ... yield pattern.
        match op.op_type:
            case RunnerOpType.ASSIGN_RUNNER:
                event_generator = self._execute_assign_op(op)
            case RunnerOpType.UNASSIGN_RUNNER:
                event_generator = self._execute_unassign_op(op)
            case RunnerOpType.RUNNER_UP:
                event_generator = self._execute_runner_up_op(op)
            case RunnerOpType.RUNNER_DOWN:
                event_generator = self._execute_runner_down_op(op)
            case RunnerOpType.DOWNLOAD:
                event_generator = self._execute_download_op(op)
            case RunnerOpType.CHAT_COMPLETION:
                event_generator = self._execute_chat_completion_op(op)

        async for event in event_generator:
            yield event

    ## Planning logic
    def plan(self, state: State) -> RunnerOp | None:
        # Compare state to worker 'mood'
        
        # First spin things down
        
        # Then spin things up

        # Then make sure things are downloading.
        for instance_id, instance in state.instances.items():
            # We should already have asserted that this runner exists
            # If it didn't exist then we return a assign_runner op.
            for node_id, runner_id in instance.instance_params.shard_assignments.node_to_runner.items():
                if node_id != self.node_id:
                    continue
                assert runner_id in self.assigned_runners

                runner = self.assigned_runners[runner_id]

                if not runner.is_downloaded:
                    if runner.status.runner_status == RunnerStatusType.Downloading:
                        return None
                    else:
                        return DownloadOp(
                            runner_id=runner_id,
                            instance_id=instance_id,
                            shard_metadata=instance.instance_params.shard_assignments.runner_to_shard[runner_id],
                            hosts=instance.instance_params.hosts
                        )




        # Finally, chat completion.
        return None


    # Handle state updates
    async def _loop(self):
        while True:
            state_copy = self.state.model_copy(deep=True)

            op: RunnerOp | None = self.plan(state_copy)            

            # Run the op, synchronously blocking for now.
            if op is not None:
                async for event in self._execute_op(op):
                    print(event)
                    # self.event_publisher(event)

            await asyncio.sleep(0.01)

    # TODO: Handle tail event log
    # TODO: Handle resource monitoring (write-only)

async def main():
    

    print("Hello from worker!")

if __name__ == "__main__":
    asyncio.run(main())
