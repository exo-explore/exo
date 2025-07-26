import asyncio
import logging
import os
from asyncio import Queue
from functools import partial
from typing import AsyncGenerator, Optional

from pydantic import BaseModel, ConfigDict

from shared.apply import apply
from shared.db.sqlite import AsyncSQLiteEventStorage
from shared.db.sqlite.event_log_manager import EventLogConfig, EventLogManager
from shared.node_id import get_node_id_keypair
from shared.types.common import NodeId
from shared.types.events import (
    ChunkGenerated,
    Event,
    InstanceId,
    NodePerformanceMeasured,
    RunnerDeleted,
    RunnerStatusUpdated,
    TaskStateUpdated,
)
from shared.types.profiling import NodePerformanceProfile
from shared.types.state import State
from shared.types.tasks import TaskStatus
from shared.types.worker.common import RunnerId
from shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadProgressData,
)
from shared.types.worker.instances import InstanceStatus
from shared.types.worker.mlx import Host
from shared.types.worker.ops import (
    AssignRunnerOp,
    DownloadOp,
    ExecuteTaskOp,
    RunnerDownOp,
    RunnerFailedOp,
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
from worker.download.download_utils import build_model_path
from worker.runner.runner_supervisor import RunnerSupervisor
from worker.utils.profile import start_polling_node_metrics


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
        return os.path.exists(build_model_path(self.shard_metadata.model_meta.model_id))

    def status_update_event(self) -> RunnerStatusUpdated:
        return RunnerStatusUpdated(
            runner_id=self.runner_id,
            runner_status=self.status,
        )

class Worker:
    def __init__(
        self,
        node_id: NodeId,
        logger: logging.Logger,
        worker_events: AsyncSQLiteEventStorage | None,
        global_events: AsyncSQLiteEventStorage | None,
    ):
        self.node_id: NodeId = node_id
        self.state: State = State()
        self.worker_events: AsyncSQLiteEventStorage | None = worker_events # worker_events is None in some tests.
        self.global_events: AsyncSQLiteEventStorage | None = global_events
        self.logger: logging.Logger = logger

        self.assigned_runners: dict[RunnerId, AssignedRunner] = {}
        self._task: asyncio.Task[None] | None = None

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
        yield RunnerDeleted(runner_id=op.runner_id)

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
        return

    async def _execute_runner_failed_op(
        self, op: RunnerFailedOp
    ) -> AsyncGenerator[Event, None]:
        '''
        We detected that this runner has failed. So we'll put it into 'failed' state now, triggering the rest of the instance to spin down.
        '''
        assigned_runner = self.assigned_runners[op.runner_id]

        assigned_runner.status = FailedRunnerStatus()
        yield self.assigned_runners[op.runner_id].status_update_event()

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


    async def _execute_task_op(
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

                if assigned_runner.shard_metadata.device_rank == 0:
                    await queue.put(TaskStateUpdated(
                        task_id=op.task.task_id,
                        task_status=TaskStatus.RUNNING,
                    ))

            try:
                async for chunk in assigned_runner.runner.stream_response(
                        task=op.task,
                        request_started_callback=partial(running_callback, queue)):
                    if assigned_runner.shard_metadata.device_rank == 0:
                        await queue.put(ChunkGenerated(
                            # todo: at some point we will no longer have a bijection between task_id and row_id. 
                            # So we probably want to store a mapping between these two in our Worker object.
                            command_id=chunk.command_id, 
                            chunk=chunk
                        ))

                if assigned_runner.shard_metadata.device_rank == 0:
                    await queue.put(TaskStateUpdated(
                        task_id=op.task.task_id,
                        task_status=TaskStatus.COMPLETE,
                    ))

                # After a successful inference:
                assigned_runner.status = LoadedRunnerStatus()
                await queue.put(assigned_runner.status_update_event())


            except Exception as e:
                # TODO: What log level?
                self.logger.log(2, f'Runner failed whilst running inference task. Task: {op.task}. Error: {e}')

                if assigned_runner.shard_metadata.device_rank == 0:
                    await queue.put(TaskStateUpdated(
                        task_id=op.task.task_id,
                        task_status=TaskStatus.FAILED,
                    ))

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
            case RunnerOpType.RUNNER_FAILED:
                event_generator = self._execute_runner_failed_op(op)
            case RunnerOpType.DOWNLOAD:
                event_generator = self._execute_download_op(op)
            case RunnerOpType.CHAT_COMPLETION:
                event_generator = self._execute_task_op(op)

        async for event in event_generator:
            yield event

    ## Planning logic
    def plan(self, state: State) -> RunnerOp | None:
        # Compare state to worker 'mood'

        # First, unassign assigned runners that are no longer in the state.
        for runner_id, _ in self.assigned_runners.items():
            runner_ids: list[RunnerId] = [
                runner_id
                for instance in state.instances.values()
                for runner_id in instance.shard_assignments.runner_to_shard
            ]
            if runner_id not in runner_ids:
                return UnassignRunnerOp(runner_id=runner_id)

        for runner_id, assigned_runner in self.assigned_runners.items():
            if assigned_runner.runner is not None and \
                not assigned_runner.runner.healthy and \
                not isinstance(assigned_runner.status, FailedRunnerStatus):
                return RunnerFailedOp(runner_id=runner_id)

        # Then spin down active runners
        for _instance_id, instance in state.instances.items():
            for node_id, runner_id in instance.shard_assignments.node_to_runner.items():
                if node_id != self.node_id:
                    continue

                # We spin down a runner if it's meant to be inactive and it's Loaded.
                if runner_id in self.assigned_runners and \
                    isinstance(self.assigned_runners[runner_id].status, LoadedRunnerStatus) and \
                    instance.instance_type == InstanceStatus.INACTIVE:
                    return RunnerDownOp(runner_id=runner_id)

        # If we are part of an instance that has a dead node - and we aren't the dead node - we should spin down
        # TODO: We need to limit number of retries if we keep failing.
        for _instance_id, instance in state.instances.items():
            if self.node_id in instance.shard_assignments.node_to_runner and \
                instance.shard_assignments.node_to_runner[self.node_id] in self.assigned_runners and \
                not isinstance(self.assigned_runners[instance.shard_assignments.node_to_runner[self.node_id]].status, ReadyRunnerStatus): # make sure that our runner has not already been spun down into ready state
                other_node_in_instance_has_failed = False
                for runner_id in instance.shard_assignments.runner_to_shard:
                    if runner_id in state.runners and \
                        isinstance(state.runners[runner_id], FailedRunnerStatus) and \
                        runner_id not in self.assigned_runners:
                        other_node_in_instance_has_failed= True

                if other_node_in_instance_has_failed:
                    # Spin down *our* runner
                    return RunnerDownOp(runner_id=instance.shard_assignments.node_to_runner[self.node_id])

        # If we are failed - and *all of the other nodes have spun down* - then we can spin down too.
        for _instance_id, instance in state.instances.items():
            if self.node_id in instance.shard_assignments.node_to_runner and \
                instance.shard_assignments.node_to_runner[self.node_id] in state.runners and \
                isinstance(self.assigned_runners[instance.shard_assignments.node_to_runner[self.node_id]].status, FailedRunnerStatus):

                num_spundown_nodes = 0
                for runner_id in instance.shard_assignments.runner_to_shard:
                    if isinstance(state.runners[runner_id], ReadyRunnerStatus) and \
                        runner_id not in self.assigned_runners:
                        num_spundown_nodes += 1
                    # Suggested:
                    # if runner_id in state.runners and isinstance(state.runners[runner_id], ReadyRunnerStatus):
                    #     if runner_id != instance.shard_assignments.node_to_runner[self.node_id]:
                    #         num_spundown_nodes += 1

                if num_spundown_nodes == next(iter(instance.shard_assignments.runner_to_shard.values())).world_size - 1:
                    # All the other nodes are spun down - so now we can spin down too.
                    # This also catches the case of 1-node. If there's one node in the instance then we should spin down straight away
                    return RunnerDownOp(runner_id=instance.shard_assignments.node_to_runner[self.node_id])

        # Then assign runners we do want
        for instance_id, instance in state.instances.items():
            for node_id, runner_id in instance.shard_assignments.node_to_runner.items():
                if node_id != self.node_id:
                    continue

                if runner_id not in self.assigned_runners:
                    return AssignRunnerOp(
                        runner_id=runner_id,
                        instance_id=instance_id,
                        shard_metadata=instance.shard_assignments.runner_to_shard[runner_id],
                        hosts=instance.hosts
                    )

        # Then make sure things are downloading.
        for instance_id, instance in state.instances.items():
            # We should already have asserted that this runner exists
            # If it didn't exist then we return a assign_runner op.
            for node_id, runner_id in instance.shard_assignments.node_to_runner.items():
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
                            shard_metadata=instance.shard_assignments.runner_to_shard[runner_id],
                            hosts=instance.hosts
                        )

        # Then spin up 'ready' runners that should be active
        for _instance_id, instance in state.instances.items():
            if self.node_id in instance.shard_assignments.node_to_runner and \
                self.assigned_runners[instance.shard_assignments.node_to_runner[self.node_id]].runner is None and \
                instance.instance_type == InstanceStatus.ACTIVE:

                # We are part of this instance, we want it up but it hasn't been spun up yet.
                # Need to assert all other runners are ready before we can spin up.
                ready_to_spin = True
                for runner_id in instance.shard_assignments.node_to_runner.values():
                    if runner_id in state.runners and state.runners[runner_id].runner_status != RunnerStatusType.Ready:
                        ready_to_spin = False

                if ready_to_spin:
                    return RunnerUpOp(runner_id=instance.shard_assignments.node_to_runner[self.node_id])

        # Then make sure things are running based on tasks.
        for instance_id, instance in state.instances.items():
            for node_id, runner_id in instance.shard_assignments.node_to_runner.items():
                if node_id != self.node_id:
                    continue
                assert runner_id in self.assigned_runners
                runner = self.assigned_runners[runner_id]
                if runner.status.runner_status != RunnerStatusType.Loaded:
                    continue # The only previous state to get to Running is from Loaded

                for _, task in state.tasks.items():
                    if task.instance_id == instance_id and task.task_status == TaskStatus.PENDING:
                        if (runner.shard_metadata.device_rank >= 1 or runner.shard_metadata.world_size == 1):
                            return ExecuteTaskOp(runner_id=runner_id, task=task)
                        else:
                            # We already know our own status is Loaded. We are rank 0,
                            # so let's check that all the other runners are running - ready for us to fire the prompt.
                            running_runner_count = 0
                            for other_runner_id, other_runner_status in state.runners.items():
                                if other_runner_id in instance.shard_assignments.node_to_runner.values() and \
                                    isinstance(other_runner_status, RunningRunnerStatus):
                                    running_runner_count += 1

                            if running_runner_count == runner.shard_metadata.world_size - 1:
                                return ExecuteTaskOp(runner_id=runner_id, task=task)

        return None


    async def event_publisher(self, event: Event) -> None:
        assert self.worker_events is not None
        await self.worker_events.append_events([event], self.node_id)

    # Handle state updates
    async def run(self):
        assert self.global_events is not None

        while True:
            _rank = list(self.assigned_runners.values())[0].shard_metadata.device_rank if self.assigned_runners else None
            # 1. get latest events
            events = await self.global_events.get_events_since(self.state.last_event_applied_idx)

            # 2. for each event, apply it to the state and run sagas
            for event_from_log in events:
                self.state = apply(self.state, event_from_log)

            # 3. based on the updated state, we plan & execute an operation.
            op: RunnerOp | None = self.plan(self.state)

            # run the op, synchronously blocking for now
            if op is not None:
                async for event in self._execute_op(op):
                    await self.event_publisher(event)

            await asyncio.sleep(0.01)
            self.logger.info(f"state: {self.state}")


async def main():
    node_id_keypair = get_node_id_keypair()
    node_id = NodeId(node_id_keypair.to_peer_id().to_base58())
    logger: logging.Logger = logging.getLogger('worker_logger')
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

    event_log_manager = EventLogManager(EventLogConfig(), logger)
    await event_log_manager.initialize()
    
    # TODO: add profiling etc to resource monitor
    async def resource_monitor_callback(node_performance_profile: NodePerformanceProfile) -> None:
        await event_log_manager.worker_events.append_events(
            [NodePerformanceMeasured(node_id=node_id, node_profile=node_performance_profile)], origin=node_id
        )
    asyncio.create_task(start_polling_node_metrics(callback=resource_monitor_callback))

    worker = Worker(node_id, logger, event_log_manager.worker_events, event_log_manager.global_events)

    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
