import hashlib
from collections import defaultdict
from datetime import datetime, timezone

import anyio
from anyio import fail_after, to_thread
from loguru import logger

from exo.api.types import ImageEditsTaskParams
from exo.download.download_utils import is_read_only_model_dir, resolve_existing_model
from exo.routing.event_router import (
    EventRouterBrokenResourceError,
    EventRouterClosedResourceError,
    NodeEventReceiver,
)
from exo.shared.apply import apply
from exo.shared.constants import EXO_MAX_INSTANCE_RETRIES
from exo.shared.models.model_cards import ModelId, card_cache
from exo.shared.types.chunks import ErrorChunk, InputImageChunk
from exo.shared.types.commands import (
    DeleteInstance,
    ForwarderCommand,
    ForwarderDownloadCommand,
    StartDownload,
)
from exo.shared.types.common import CommandId, NodeId, SystemId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    InputChunkReceived,
    InstanceDeleted,
    NodeDownloadProgress,
    NodeGatheredInfo,
    StateSnapshot,
    TaskCreated,
    TaskDeleted,
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
    LoadModel,
    Shutdown,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.text_generation import Base64Image, Base64ImageHash
from exo.shared.types.topology import Connection, SocketConnection
from exo.shared.types.worker.downloads import DownloadCompleted
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runners import RunnerId
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.info_gatherer.info_gatherer import GatheredInfo, InfoGatherer
from exo.utils.info_gatherer.net_profile import check_reachable
from exo.utils.keyed_backoff import KeyedBackoff
from exo.utils.task_group import TaskGroup
from exo.worker.plan import plan
from exo.worker.runner.supervisor import RunnerSupervisor


class Worker:
    def __init__(
        self,
        node_id: NodeId,
        *,
        event_receiver: NodeEventReceiver,
        event_sender: Sender[Event],
        # This is for requesting updates. It doesn't need to be a general command sender right now,
        # but I think it's the correct way to be thinking about commands
        command_sender: Sender[ForwarderCommand],
        download_command_sender: Sender[ForwarderDownloadCommand],
        api_port: int,
    ):
        self.node_id: NodeId = node_id
        self.event_receiver = event_receiver
        self.event_sender = event_sender
        self.command_sender = command_sender
        self.download_command_sender = download_command_sender
        self.api_port = api_port

        self.state: State = State()
        self.runners: dict[RunnerId, RunnerSupervisor] = {}
        self._tg: TaskGroup = TaskGroup()

        self._system_id = SystemId()

        # Images a request sends ahead of itself, kept per request until it runs here or
        # turns out not to need this node
        self.input_chunk_buffer: dict[CommandId, dict[int, InputImageChunk]] = {}
        self.input_chunk_counts: dict[CommandId, int] = {}
        self.input_images: dict[CommandId, dict[Base64ImageHash, Base64Image]] = {}
        self._tasks_with_inputs: dict[TaskId, CommandId] = {}
        self._inputs_lost: set[TaskId] = set()

        self._download_backoff: KeyedBackoff[ModelId] = KeyedBackoff(base=0.5, cap=10.0)
        self._instance_backoff: KeyedBackoff[InstanceId] = KeyedBackoff(
            base=0.5, cap=10.0
        )
        self._stopped: anyio.Event = anyio.Event()

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
                tg.start_soon(self._reconcile_custom_cards)
        except* (EventRouterBrokenResourceError, EventRouterClosedResourceError):
            # Event router has been closed (try-star syntax handles error groups)
            pass
        finally:
            # Actual shutdown code - waits for all tasks to complete before executing.
            logger.info("Stopping Worker")
            self.event_sender.close()
            self.command_sender.close()
            self.download_command_sender.close()
            for runner in self.runners.values():
                runner.shutdown()
            self._stopped.set()

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
            async for update in events:
                if isinstance(update, StateSnapshot):
                    self.state = update.state
                    continue
                # 2. for each event, apply it to the state
                self.state = apply(self.state, event=update)
                event = update.event

                if isinstance(event, InstanceDeleted):
                    self._instance_backoff.reset(event.instance_id)

                # Buffer input image chunks for image editing
                if isinstance(event, InputChunkReceived):
                    cmd_id = event.command_id
                    if cmd_id not in self.input_chunk_buffer:
                        self.input_chunk_buffer[cmd_id] = {}
                        self.input_chunk_counts[cmd_id] = event.chunk.total_chunks

                    self.input_chunk_buffer[cmd_id][event.chunk.chunk_index] = (
                        event.chunk
                    )
                    if (
                        len(self.input_chunk_buffer[cmd_id])
                        == self.input_chunk_counts[cmd_id]
                    ):
                        per_image: defaultdict[int, list[InputImageChunk]] = (
                            defaultdict(list)
                        )
                        for chunk in self.input_chunk_buffer[cmd_id].values():
                            per_image[chunk.image_index].append(chunk)
                        for chunks_for_image in per_image.values():
                            sorted_chunks = sorted(
                                chunks_for_image, key=lambda c: c.chunk_index
                            )
                            img = Base64Image("".join(c.data for c in sorted_chunks))
                            self.input_images.setdefault(cmd_id, {})[
                                Base64ImageHash(
                                    hashlib.sha256(img.encode("ascii")).hexdigest()
                                )
                            ] = img

                if isinstance(event, TaskCreated) and isinstance(
                    event.task, (TextGeneration, ImageEdits)
                ):
                    self._keep_inputs_if_needed_here(event.task)

                if isinstance(event, TaskDeleted):
                    self._inputs_lost.discard(event.task_id)
                    if (
                        cmd_id := self._tasks_with_inputs.pop(event.task_id, None)
                    ) is not None:
                        self._drop_inputs(cmd_id)

    def _keep_inputs_if_needed_here(self, task: TextGeneration | ImageEdits) -> None:
        """Every node receives every request's images; only the nodes that run it keep them."""
        instance = self.state.instances.get(task.instance_id)
        if (
            instance is None
            or self.node_id not in instance.shard_assignments.node_to_runner
        ):
            self._drop_inputs(task.command_id)
        elif task.command_id in self.input_chunk_buffer:
            self._tasks_with_inputs[task.task_id] = task.command_id

    def _drop_inputs(self, command_id: CommandId) -> None:
        self.input_chunk_buffer.pop(command_id, None)
        self.input_chunk_counts.pop(command_id, None)
        self.input_images.pop(command_id, None)

    async def _report_lost_inputs(self, task: TextGeneration | ImageEdits) -> None:
        """Fail a request whose images never reached this node.

        A request's images always arrive before the request itself, so if they are missing
        once it is here, they are not coming: this node caught up from a state snapshot part
        way through them. Waiting would hold the request, and the model's instance, forever.
        """
        if task.task_id in self._inputs_lost:
            return
        self._inputs_lost.add(task.task_id)
        self._drop_inputs(task.command_id)
        logger.warning(f"Images for request {task.command_id} never reached this node")
        await self.event_sender.send(
            ChunkGenerated(
                command_id=task.command_id,
                chunk=ErrorChunk(
                    model=ModelId(task.task_params.model),
                    error_message=(
                        "The request's images didn't reach every device running the "
                        "model. Please send it again."
                    ),
                ),
            )
        )
        await self.event_sender.send(
            TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Failed)
        )

    async def _reconcile_custom_cards(self) -> None:
        while True:
            await anyio.sleep(1)
            target = dict(self.state.custom_model_cards)
            for model_id, card in target.items():
                if card_cache.get(model_id) == card:
                    continue
                await card_cache.save(card)

            for card in await card_cache.list_all():
                if card.model_id not in target:
                    await card_cache.pop(card.model_id)

    async def plan_step(self):
        while True:
            await anyio.sleep(0.1)
            task: Task | None = plan(
                self.node_id,
                self.runners,
                self.state.downloads,
                self.state.instances,
                self.state.runners,
                self.state.tasks,
                self._instance_backoff,
                self._download_backoff,
            )
            if task is None:
                continue

            if isinstance(task, CreateRunner):
                iid = task.instance_id
                if self._instance_backoff.attempts(iid) >= EXO_MAX_INSTANCE_RETRIES:
                    logger.warning(
                        f"Instance {iid} exceeded {EXO_MAX_INSTANCE_RETRIES} retries, requesting deletion"
                    )
                    await self.command_sender.send(
                        ForwarderCommand(
                            origin=self._system_id,
                            command=DeleteInstance(instance_id=iid),
                        )
                    )
                    continue

            logger.info(f"Worker plan: {task.__class__.__name__}")
            assert task.task_status
            await self.event_sender.send(TaskCreated(task_id=task.task_id, task=task))

            # lets not kill the worker if a runner is unresponsive
            match task:
                case CreateRunner():
                    await self._create_supervisor(task)
                    self._instance_backoff.record_attempt(task.instance_id)
                    await self.event_sender.send(
                        TaskStatusUpdated(
                            task_id=task.task_id, task_status=TaskStatus.Complete
                        )
                    )
                case DownloadModel(shard_metadata=shard):
                    model_id = shard.model_card.model_id
                    self._download_backoff.record_attempt(model_id)

                    found_path = await to_thread.run_sync(
                        resolve_existing_model, model_id, shard.model_card
                    )
                    if found_path is not None:
                        logger.info(f"Model {model_id} found at {found_path}")
                        await self.event_sender.send(
                            NodeDownloadProgress(
                                download_progress=DownloadCompleted(
                                    node_id=self.node_id,
                                    shard_metadata=shard,
                                    model_directory=str(found_path),
                                    total=shard.model_card.storage_size,
                                    read_only=is_read_only_model_dir(found_path),
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
                    runner = self.runners.pop(runner_id)
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
                    await self.runners[runner_id].cancel_task(cancelled_task_id)
                    await self.event_sender.send(
                        TaskStatusUpdated(
                            task_id=task.task_id, task_status=TaskStatus.Complete
                        )
                    )
                case ImageEdits() if (
                    task.task_params.total_input_chunks > 0
                    and len(self.input_chunk_buffer.get(task.command_id, {}))
                    < task.task_params.total_input_chunks
                ):
                    await self._report_lost_inputs(task)
                case ImageEdits() if task.task_params.total_input_chunks > 0:
                    # Assemble image from chunks and inject into task
                    cmd_id = task.command_id
                    chunks = self.input_chunk_buffer.get(cmd_id, {})
                    assembled = "".join(chunks[i].data for i in range(len(chunks)))
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
                    self._drop_inputs(cmd_id)
                    await self._start_runner_task(modified_task)

                case TextGeneration() if task.task_params.image_hashes and not all(
                    h in self.input_images.get(task.command_id, {})
                    for h in task.task_params.image_hashes.values()
                ):
                    await self._report_lost_inputs(task)
                case TextGeneration() if task.task_params.image_hashes:
                    cmd_id = task.command_id
                    images = self.input_images[cmd_id]
                    resolved_images = [
                        images[h]
                        for _, h in sorted(task.task_params.image_hashes.items())
                    ]
                    modified_task = task.model_copy(
                        update={
                            "task_params": task.task_params.model_copy(
                                update={"images": resolved_images}
                            )
                        }
                    )
                    self._drop_inputs(cmd_id)
                    await self._start_runner_task(modified_task)
                case LoadModel(instance_id=instance_id):
                    if (instance := self.state.instances.get(instance_id)) is not None:
                        model_id = instance.shard_assignments.model_id
                        self._download_backoff.reset(model_id)

                    await self._start_runner_task(task)
                case task:
                    await self._start_runner_task(task)

    async def shutdown(self):
        self._tg.cancel_tasks()
        await self._stopped.wait()

    async def _start_runner_task(self, task: Task):
        if (instance := self.state.instances.get(task.instance_id)) is not None:
            await self.runners[
                instance.shard_assignments.node_to_runner[self.node_id]
            ].start_task(task)

    async def _create_supervisor(self, task: CreateRunner) -> RunnerSupervisor:
        """Creates and stores a new AssignedRunner with initial downloading status."""
        runner = await RunnerSupervisor.create(
            bound_instance=task.bound_instance,
            event_sender=self.event_sender.clone(),
        )
        self.runners[task.bound_instance.bound_runner_id] = runner
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
                api_port=self.api_port,
            ):
                if ip in conns[nid]:
                    continue
                conns[nid].add(ip)
                edge = SocketConnection(
                    # nonsense multiaddr
                    sink_multiaddr=Multiaddr(address=f"/ip4/{ip}/tcp/{self.api_port}")
                    if "." in ip
                    # nonsense multiaddr
                    else Multiaddr(address=f"/ip6/{ip}/tcp/{self.api_port}"),
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
                if conn.edge.sink_multiaddr.port != self.api_port:
                    continue
                if (
                    conn.sink not in conns
                    or conn.edge.sink_multiaddr.ip_address not in conns[conn.sink]
                ):
                    logger.debug(f"ping failed to discover {conn=}")
                    await self.event_sender.send(TopologyEdgeDeleted(conn=conn))

            await anyio.sleep(10)
