from collections import defaultdict
from datetime import datetime, timezone

import anyio
from anyio import fail_after
from loguru import logger

from exo.download.download_utils import resolve_model_in_path
from exo.shared.apply import apply
from exo.shared.models.model_cards import ModelId
from exo.shared.types.api import ImageEditsTaskParams
from exo.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
    StartDownload,
)
from exo.shared.types.common import CommandId, NodeId, SystemId
from exo.shared.types.events import (
    Event,
    IndexedEvent,
    InputChunkReceived,
    NodeDownloadProgress,
    NodeGatheredInfo,
    TaskCreated,
    TaskStatusUpdated,
    TopologyEdgeCreated,
    TopologyEdgeDeleted,
)
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    CancelTask,
    CreateRunner,
    DownloadModel,
    ImageEdits,
    Shutdown,
    Task,
    TaskStatus,
)
from exo.shared.types.topology import Connection, SocketConnection
from exo.shared.types.worker.downloads import DownloadCompleted
from exo.shared.types.worker.runners import RunnerId
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.info_gatherer.info_gatherer import GatheredInfo, InfoGatherer
from exo.utils.info_gatherer.net_profile import check_reachable
from exo.utils.keyed_backoff import KeyedBackoff
from exo.utils.task_group import TaskGroup
from exo.worker.plan import plan
from exo.worker.runner.runner_supervisor import RunnerSupervisor


class Worker:
    def __init__(
        self,
        node_id: NodeId,
        *,
        event_receiver: Receiver[IndexedEvent],
        event_sender: Sender[Event],
        # This is for requesting updates. It doesn't need to be a general command sender right now,
        # but I think it's the correct way to be thinking about commands
        command_sender: Sender[ForwarderCommand],
        download_command_sender: Sender[ForwarderDownloadCommand],
    ):
        self.node_id: NodeId = node_id
        self.event_receiver = event_receiver
        self.event_sender = event_sender
        self.command_sender = command_sender
        self.download_command_sender = download_command_sender

        self.state: State = State()
        self.runners: dict[RunnerId, RunnerSupervisor] = {}
        self._tg: TaskGroup = TaskGroup()

        self._system_id = SystemId()

        # Buffer for input image chunks (for image editing)
        self.input_chunk_buffer: dict[CommandId, dict[int, str]] = {}
        self.input_chunk_counts: dict[CommandId, int] = {}
        self._max_chunk_buffers: int = 100

        self._download_backoff: KeyedBackoff[ModelId] = KeyedBackoff(base=0.5, cap=10.0)
        self._state_version: int = 0
        self._state_changed: anyio.Event = anyio.Event()

    async def run(self):
        logger.info("Starting Worker")

        info_send, info_recv = channel[GatheredInfo]()
        info_gatherer: InfoGatherer = InfoGatherer(info_send)

        try:
            async with self._tg as tg:
                tg.start_soon(info_gatherer.run)
                tg.start_soon(self._forward_info, info_recv)
                tg.start_soon(self.plan_step)
                tg.start_soon(self._event_applier)
                tg.start_soon(self._poll_connection_updates)
        except BaseException:
            # Worker was shut down via self._tg.cancel_tasks() — swallow the CancelledError
            # so it doesn't propagate into the Node's TaskGroup and kill the new Worker.
            pass
        finally:
            # Actual shutdown code - waits for all tasks to complete before executing.
            logger.info("Stopping Worker")
            self.event_sender.close()
            self.command_sender.close()
            self.download_command_sender.close()
            for runner in self.runners.values():
                runner.shutdown()

    async def _forward_info(self, recv: Receiver[GatheredInfo]):
        with recv as info_stream:
            async for info in info_stream:
                await self.event_sender.send(
                    NodeGatheredInfo(
                        node_id=self.node_id,
                        when=str(datetime.now(tz=timezone.utc)),
                        info=info,
                    )
                )

    async def _event_applier(self):
        with self.event_receiver as events:
            async for event in events:
                # 2. for each event, apply it to the state
                self.state = apply(self.state, event=event)
                self._state_version += 1
                self._state_changed.set()
                self._state_changed = anyio.Event()
                event = event.event

                # Buffer input image chunks for image editing
                if isinstance(event, InputChunkReceived):
                    cmd_id = event.command_id
                    if cmd_id not in self.input_chunk_buffer:
                        # Evict oldest buffers if too many accumulate (leak prevention)
                        if len(self.input_chunk_buffer) >= self._max_chunk_buffers:
                            oldest_key = next(iter(self.input_chunk_buffer))
                            self.input_chunk_buffer.pop(oldest_key, None)
                            self.input_chunk_counts.pop(oldest_key, None)
                            logger.warning(
                                f"Evicted stale chunk buffer for {oldest_key}"
                            )
                        self.input_chunk_buffer[cmd_id] = {}
                        self.input_chunk_counts[cmd_id] = event.chunk.total_chunks

                    self.input_chunk_buffer[cmd_id][event.chunk.chunk_index] = (
                        event.chunk.data
                    )

    async def plan_step(self):
        while True:
            event = self._state_changed
            state_changed = False
            with anyio.move_on_after(1.0):
                await event.wait()
                state_changed = True
            task: Task | None = plan(
                self.node_id,
                self.runners,
                self.state.downloads,
                self.state.instances,
                self.state.runners,
                self.state.tasks,
                self.input_chunk_buffer,
                self.input_chunk_counts,
            )
            if task is None:
                if not state_changed:
                    # No state change and no work — true idle, yield before retrying
                    await anyio.sleep(0.5)
                continue

            # Gate DownloadModel on backoff BEFORE emitting TaskCreated
            # to prevent flooding the event log with useless events
            if isinstance(task, DownloadModel):
                model_id = task.shard_metadata.model_card.model_id
                if not self._download_backoff.should_proceed(model_id):
                    continue

            logger.info(f"Worker plan: {task.__class__.__name__}")
            if not task.task_status:
                logger.warning(f"Task {task.task_id} has no task_status, skipping")
                continue
            await self.event_sender.send(TaskCreated(task_id=task.task_id, task=task))

            # lets not kill the worker if a runner is unresponsive
            match task:
                case CreateRunner():
                    self._create_supervisor(task)
                    await self.event_sender.send(
                        TaskStatusUpdated(
                            task_id=task.task_id, task_status=TaskStatus.Complete
                        )
                    )
                case DownloadModel(shard_metadata=shard):
                    model_id = shard.model_card.model_id
                    self._download_backoff.record_attempt(model_id)

                    found_path = resolve_model_in_path(model_id)
                    if found_path is not None:
                        logger.info(
                            f"Model {model_id} found in EXO_MODELS_PATH at {found_path}"
                        )
                        self._download_backoff.reset(model_id)
                        await self.event_sender.send(
                            NodeDownloadProgress(
                                download_progress=DownloadCompleted(
                                    node_id=self.node_id,
                                    shard_metadata=shard,
                                    model_directory=str(found_path),
                                    total=shard.model_card.storage_size,
                                    read_only=True,
                                )
                            )
                        )
                        await self.event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id,
                                task_status=TaskStatus.Complete,
                            )
                        )
                    else:
                        await self.download_command_sender.send(
                            ForwarderDownloadCommand(
                                origin=self._system_id,
                                command=StartDownload(
                                    target_node_id=self.node_id,
                                    shard_metadata=shard,
                                ),
                            )
                        )
                        await self.event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id,
                                task_status=TaskStatus.Running,
                            )
                        )
                case Shutdown(runner_id=runner_id):
                    runner = self.runners.pop(runner_id, None)
                    if runner is None:
                        logger.warning(f"Shutdown: runner {runner_id} already removed")
                        await self.event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id, task_status=TaskStatus.Complete
                            )
                        )
                        continue
                    try:
                        with fail_after(3):
                            await runner.start_task(task)
                    except TimeoutError:
                        await self.event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id, task_status=TaskStatus.TimedOut
                            )
                        )
                    finally:
                        runner.shutdown()
                case CancelTask(
                    cancelled_task_id=cancelled_task_id, runner_id=runner_id
                ):
                    if (cancel_runner := self.runners.get(runner_id)) is not None:
                        await cancel_runner.cancel_task(cancelled_task_id)
                    else:
                        logger.warning(f"CancelTask: runner {runner_id} not found")
                    await self.event_sender.send(
                        TaskStatusUpdated(
                            task_id=task.task_id, task_status=TaskStatus.Complete
                        )
                    )
                case ImageEdits() if task.task_params.total_input_chunks > 0:
                    # Assemble image from chunks and inject into task
                    cmd_id = task.command_id
                    chunks = self.input_chunk_buffer.get(cmd_id, {})
                    assembled = "".join(chunks[i] for i in sorted(chunks.keys()))
                    logger.info(
                        f"Assembled input image from {len(chunks)} chunks, "
                        f"total size: {len(assembled)} bytes"
                    )
                    # Create modified task with assembled image data
                    modified_task = ImageEdits(
                        task_id=task.task_id,
                        command_id=task.command_id,
                        instance_id=task.instance_id,
                        task_status=task.task_status,
                        task_params=ImageEditsTaskParams(
                            image_data=assembled,
                            total_input_chunks=task.task_params.total_input_chunks,
                            prompt=task.task_params.prompt,
                            model=task.task_params.model,
                            n=task.task_params.n,
                            quality=task.task_params.quality,
                            output_format=task.task_params.output_format,
                            response_format=task.task_params.response_format,
                            size=task.task_params.size,
                            image_strength=task.task_params.image_strength,
                            bench=task.task_params.bench,
                            stream=task.task_params.stream,
                            partial_images=task.task_params.partial_images,
                            advanced_params=task.task_params.advanced_params,
                        ),
                    )
                    # Cleanup buffers
                    if cmd_id in self.input_chunk_buffer:
                        del self.input_chunk_buffer[cmd_id]
                    if cmd_id in self.input_chunk_counts:
                        del self.input_chunk_counts[cmd_id]
                    await self._start_runner_task(modified_task)
                case task:
                    await self._start_runner_task(task)

    def shutdown(self):
        self._tg.cancel_tasks()

    async def _start_runner_task(self, task: Task):
        if (instance := self.state.instances.get(task.instance_id)) is not None:
            runner_id = instance.shard_assignments.node_to_runner.get(self.node_id)
            if runner_id is None:
                logger.warning(
                    f"No runner mapping for node {self.node_id} in instance {task.instance_id}"
                )
                return
            runner = self.runners.get(runner_id)
            if runner is None:
                logger.warning(f"Runner {runner_id} not found for task {task.task_id}")
                return
            await runner.start_task(task)

    def _create_supervisor(self, task: CreateRunner) -> RunnerSupervisor:
        runner_id = task.bound_instance.bound_runner_id
        if existing := self.runners.pop(runner_id, None):
            existing.shutdown()
        runner = RunnerSupervisor.create(
            bound_instance=task.bound_instance,
            event_sender=self.event_sender.clone(),
        )
        self.runners[runner_id] = runner
        self._tg.start_soon(runner.run)
        return runner

    async def _poll_connection_updates(self):
        while True:
            edges = set(
                conn.edge for conn in self.state.topology.out_edges(self.node_id)
            )
            conns: defaultdict[NodeId, set[str]] = defaultdict(set)
            async for ip, nid in check_reachable(
                self.state.topology,
                self.node_id,
                self.state.node_network,
            ):
                if ip in conns[nid]:
                    continue
                conns[nid].add(ip)
                edge = SocketConnection(
                    # nonsense multiaddr
                    sink_multiaddr=Multiaddr(address=f"/ip4/{ip}/tcp/52415")
                    if "." in ip
                    # nonsense multiaddr
                    else Multiaddr(address=f"/ip6/{ip}/tcp/52415"),
                )
                if edge not in edges:
                    logger.debug(f"ping discovered {edge=}")
                    await self.event_sender.send(
                        TopologyEdgeCreated(
                            conn=Connection(source=self.node_id, sink=nid, edge=edge)
                        )
                    )

            for conn in self.state.topology.out_edges(self.node_id):
                if not isinstance(conn.edge, SocketConnection):
                    continue
                # ignore mDNS discovered connections
                if conn.edge.sink_multiaddr.port != 52415:
                    continue
                if (
                    conn.sink not in conns
                    or conn.edge.sink_multiaddr.ip_address not in conns[conn.sink]
                ):
                    logger.debug(f"ping failed to discover {conn=}")
                    await self.event_sender.send(TopologyEdgeDeleted(conn=conn))

            await anyio.sleep(30)
