import hashlib
from collections import defaultdict
from datetime import datetime, timezone

import anyio
from anyio import fail_after, to_thread
from loguru import logger

from exo.api.types import ImageEditsTaskParams
from exo.download.download_utils import is_read_only_model_dir, resolve_existing_model
from exo.routing.event_router import EventRouter
from exo.routing.snapshot_receiver import SnapshotReceiver
from exo.shared.apply import apply
from exo.shared.constants import EXO_MAX_INSTANCE_RETRIES
from exo.shared.models.model_cards import (
    ModelCard,
    ModelId,
    add_to_card_cache,
    delete_custom_card,
)
from exo.shared.types.chunks import InputImageChunk
from exo.shared.types.commands import (
    DeleteInstance,
    ForwarderCommand,
    ForwarderDownloadCommand,
    RequestSnapshot,
    StartDownload,
)
from exo.shared.types.common import CommandId, NodeId, SessionId, SystemId
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
    TransientEvent,
)
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.snapshots import SnapshotChunk
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    CancelTask,
    CreateRunner,
    DownloadModel,
    ImageEdits,
    LoadModel,
    Shutdown,
    Task,
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

_SNAPSHOT_FETCH_TIMEOUT_SECONDS = 30


class Worker:
    def __init__(
        self,
        node_id: NodeId,
        session_id: SessionId,
        *,
        event_router: EventRouter,
        event_receiver: Receiver[IndexedEvent],
        event_sender: Sender[Event],
        transient_event_receiver: Receiver[TransientEvent],
        transient_event_sender: Sender[TransientEvent],
        snapshot_chunk_receiver: Receiver[SnapshotChunk],
        # This is for requesting updates. It doesn't need to be a general command sender right now,
        # but I think it's the correct way to be thinking about commands
        command_sender: Sender[ForwarderCommand],
        download_command_sender: Sender[ForwarderDownloadCommand],
        api_port: int,
    ):
        self.node_id: NodeId = node_id
        self.session_id: SessionId = session_id
        self.event_router = event_router
        self.event_receiver = event_receiver
        self.event_sender = event_sender
        self.transient_event_receiver = transient_event_receiver
        self.transient_event_sender = transient_event_sender
        self.snapshot_chunk_receiver = snapshot_chunk_receiver
        self.command_sender = command_sender
        self.download_command_sender = download_command_sender
        self.api_port = api_port

        self.state: State = State()
        self.runners: dict[RunnerId, RunnerSupervisor] = {}
        self._tg: TaskGroup = TaskGroup()

        self._system_id = SystemId()

        # Buffer for input image chunks (for image editing)
        self.input_chunk_buffer: dict[CommandId, dict[int, InputImageChunk]] = {}
        self.input_chunk_counts: dict[CommandId, int] = {}
        self.image_cache: dict[Base64ImageHash, Base64Image] = {}

        self._download_backoff: KeyedBackoff[ModelId] = KeyedBackoff(base=0.5, cap=10.0)
        self._instance_backoff: KeyedBackoff[InstanceId] = KeyedBackoff(
            base=0.5, cap=10.0
        )
        # Tracks the on-disk copies of custom model cards we've written. The
        # reconciliation loop diffs this against state.custom_model_cards.
        self._synced_custom_cards: dict[ModelId, ModelCard] = {}
        self._stopped: anyio.Event = anyio.Event()

    async def run(self):
        logger.info("Starting Worker")

        info_send, info_recv = channel[GatheredInfo]()
        info_gatherer: InfoGatherer = InfoGatherer(info_send)

        try:
            async with self._tg as tg:
                # Snapshot fetch runs inside the task group so shutdown()
                # during the fetch (e.g. a master re-election arriving early)
                # cancels cleanly. Events are queued by the EventRouter
                # while we wait; if the master has no snapshot we fall
                # through to the NACK-based event-log replay.
                tg.start_soon(self._bootstrap_then_run, info_gatherer, info_recv)
        finally:
            # Actual shutdown code - waits for all tasks to complete before executing.
            logger.info("Stopping Worker")
            self.event_sender.close()
            self.transient_event_sender.close()
            self.snapshot_chunk_receiver.close()
            self.command_sender.close()
            self.download_command_sender.close()
            for runner in self.runners.values():
                runner.shutdown()
            self._stopped.set()

    async def _bootstrap_then_run(
        self, info_gatherer: InfoGatherer, info_recv: Receiver[GatheredInfo]
    ) -> None:
        await self._fetch_snapshot()
        self._tg.start_soon(info_gatherer.run)
        self._tg.start_soon(self._forward_info, info_recv)
        self._tg.start_soon(self.plan_step)
        self._tg.start_soon(self._event_applier)
        self._tg.start_soon(self._transient_event_handler)
        self._tg.start_soon(self._reconcile_instance_backoff)
        self._tg.start_soon(self._reconcile_custom_cards)
        self._tg.start_soon(self._poll_connection_updates)

    async def _fetch_snapshot(self) -> None:
        """Request a snapshot from the master and apply it before draining events.

        We expect this to take well under a second on a healthy cluster. If
        nothing arrives within `_SNAPSHOT_FETCH_TIMEOUT_SECONDS`, we proceed
        with an empty state — the existing NACK loop will replay the entire
        event log, which is slow but always correct.
        """
        receiver = SnapshotReceiver(self.node_id, self.session_id)
        await self.command_sender.send(
            ForwarderCommand(
                origin=self._system_id,
                command=RequestSnapshot(requester_node_id=self.node_id),
            )
        )

        with anyio.move_on_after(_SNAPSHOT_FETCH_TIMEOUT_SECONDS):
            with self.snapshot_chunk_receiver as chunks:
                async for chunk in chunks:
                    received = receiver.ingest(chunk)
                    if received is not None:
                        self.state = received.state
                        self.event_router.set_buffer_start(
                            received.last_event_applied_idx + 1
                        )
                        logger.info(
                            f"Worker bootstrapped from snapshot at idx "
                            f"{received.last_event_applied_idx}"
                        )
                        return
        logger.info(
            "No snapshot received before timeout; falling back to full event-log replay"
        )

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
                # Events queued before our snapshot was applied are no-ops:
                # the snapshot already folded them in.
                if event.idx <= self.state.last_event_applied_idx:
                    continue
                self.state = apply(self.state, event=event)

    async def _transient_event_handler(self):
        with self.transient_event_receiver as events:
            async for event in events:
                if isinstance(event, InputChunkReceived):
                    self._absorb_input_chunk(event)

    async def _reconcile_instance_backoff(self):
        """Drop backoff entries for instances no longer in state.

        Reconciling from state (rather than reacting to InstanceDeleted) keeps
        worker behaviour correct after snapshot-based catch-up, which may not
        replay the deletion event at all.
        """
        while True:
            await anyio.sleep(1)
            live_instances = set(self.state.instances.keys())
            for iid in self._instance_backoff.tracked_keys():
                if iid not in live_instances:
                    self._instance_backoff.reset(iid)

    async def _reconcile_custom_cards(self):
        """Make the on-disk custom model cards match `state.custom_model_cards`.

        Adds and removes are driven by state diffs rather than event reactions
        so that joining via snapshot (which skips historical events) still
        leaves disk in sync.
        """
        while True:
            await anyio.sleep(1)
            target = dict(self.state.custom_model_cards)
            for model_id, card in target.items():
                if self._synced_custom_cards.get(model_id) == card:
                    continue
                try:
                    await card.save_to_custom_dir()
                    add_to_card_cache(card)
                    self._synced_custom_cards[model_id] = card
                except OSError as e:
                    logger.opt(exception=e).warning(
                        f"Failed to write custom model card {model_id}; will retry"
                    )
            for model_id in list(self._synced_custom_cards):
                if model_id in target:
                    continue
                try:
                    await delete_custom_card(model_id)
                    self._synced_custom_cards.pop(model_id, None)
                except OSError as e:
                    logger.opt(exception=e).warning(
                        f"Failed to delete custom model card {model_id}; will retry"
                    )

    def _absorb_input_chunk(self, event: InputChunkReceived) -> None:
        """Buffer an image-upload chunk; once all chunks for a command have
        arrived, reassemble each image and cache it by hash."""
        cmd_id = event.command_id
        if cmd_id not in self.input_chunk_buffer:
            self.input_chunk_buffer[cmd_id] = {}
            self.input_chunk_counts[cmd_id] = event.chunk.total_chunks

        self.input_chunk_buffer[cmd_id][event.chunk.chunk_index] = event.chunk

        if len(self.input_chunk_buffer[cmd_id]) != self.input_chunk_counts[cmd_id]:
            return

        per_image: defaultdict[int, list[InputImageChunk]] = defaultdict(list)
        for chunk in self.input_chunk_buffer[cmd_id].values():
            per_image[chunk.image_index].append(chunk)
        for chunks_for_image in per_image.values():
            sorted_chunks = sorted(chunks_for_image, key=lambda c: c.chunk_index)
            img = Base64Image("".join(c.data for c in sorted_chunks))
            digest = Base64ImageHash(hashlib.sha256(img.encode("ascii")).hexdigest())
            self.image_cache[digest] = img

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
                self.input_chunk_buffer,
                self.image_cache,
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
                    self._create_supervisor(task)
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
                    # Cleanup buffers
                    if cmd_id in self.input_chunk_buffer:
                        del self.input_chunk_buffer[cmd_id]
                    if cmd_id in self.input_chunk_counts:
                        del self.input_chunk_counts[cmd_id]
                    await self._start_runner_task(modified_task)

                case TextGeneration() if task.task_params.image_hashes:
                    cmd_id = task.command_id
                    resolved_images = [
                        self.image_cache[h]
                        for _, h in sorted(task.task_params.image_hashes.items())
                    ]
                    modified_task = task.model_copy(
                        update={
                            "task_params": task.task_params.model_copy(
                                update={"images": resolved_images}
                            )
                        }
                    )
                    if cmd_id in self.input_chunk_buffer:
                        del self.input_chunk_buffer[cmd_id]
                    if cmd_id in self.input_chunk_counts:
                        del self.input_chunk_counts[cmd_id]
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

    def _create_supervisor(self, task: CreateRunner) -> RunnerSupervisor:
        """Creates and stores a new AssignedRunner with initial downloading status."""
        runner = RunnerSupervisor.create(
            bound_instance=task.bound_instance,
            event_sender=self.event_sender.clone(),
            transient_event_sender=self.transient_event_sender.clone(),
        )
        self.runners[task.bound_instance.bound_runner_id] = runner
        self._tg.start_soon(runner.run)
        return runner

    async def _poll_connection_updates(self):
        # Poll cadence trades off cluster-view freshness against ping load.
        # 2s makes topology edge changes feel snappy on the dashboard while
        # still being well above the per-probe timeout (5s) so we don't
        # stack overlapping probes when peers are slow.
        poll_interval_seconds = 2.0
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

            await anyio.sleep(poll_interval_seconds)
