from collections import defaultdict
from datetime import datetime, timezone

import anyio
from anyio import fail_after, to_thread
from exo_rs import LVAggregator, LVPublisher, Mailbox, SessionHandle
from loguru import logger
from pydantic import TypeAdapter, ValidationError

from exo.download.download_utils import is_read_only_model_dir, resolve_existing_model
from exo.routing.event_router import (
    EventRouterBrokenResourceError,
    EventRouterClosedResourceError,
)
from exo.shared.apply import apply
from exo.shared.constants import EXO_MAX_INSTANCE_RETRIES
from exo.shared.models import model_cards
from exo.shared.models.model_cards import ModelCard, ModelId
from exo.shared.types.commands import (
    DeleteInstance,
    ForwarderCommand,
    ForwarderDownloadCommand,
    JoinInstance,
    LeaveInstance,
    Mail,
    StartDownload,
)
from exo.shared.types.common import NodeId, SystemId
from exo.shared.types.events import (
    Event,
    IndexedEvent,
    InstanceDeleted,
    NodeDownloadProgress,
    NodeGatheredInfo,
    TaskCreated,
    TaskStatusUpdated,
)
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    CancelTask,
    CreateRunner,
    DownloadModel,
    ForgetInstance,
    LoadModel,
    Shutdown,
    Task,
    TaskStatus,
)
from exo.shared.types.topology import SocketConnection, SocketConnections
from exo.shared.types.worker.downloads import DownloadCompleted
from exo.shared.types.worker.instances import Instance, InstanceId
from exo.shared.types.worker.runners import RunnerId
from exo.utils.channels import Receiver, Sender
from exo.utils.info_gatherer.info_gatherer import GatheredInfo, InfoGatherer
from exo.utils.info_gatherer.net_profile import check_reachable
from exo.utils.keyed_backoff import KeyedBackoff
from exo.utils.task_group import TaskGroup
from exo.worker.plan import plan
from exo.worker.runner.supervisor import RunnerSupervisor

PRIMARY_RUNNER_MISSING_TIMEOUT_SECONDS = 10.0


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
        session_handle: SessionHandle,
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

        self._download_backoff: KeyedBackoff[ModelId] = KeyedBackoff(base=0.5, cap=10.0)
        self._instance_backoff: KeyedBackoff[InstanceId] = KeyedBackoff(
            base=0.5, cap=10.0
        )
        self._stopped: anyio.Event = anyio.Event()
        self._sh: SessionHandle = session_handle
        self.aggregator: LVAggregator = session_handle.last_value_aggregator(
            "node_metrics"
        )
        self.mailbox: Mailbox = session_handle.mailbox(node_id)
        self.desired_instances: dict[InstanceId, Instance] = {}
        self._primary_runner_missing_since: dict[InstanceId, float] = {}
        self._primary_desired_instance_publishers: dict[InstanceId, LVPublisher] = {}
        self._socket_connections_publisher: LVPublisher = (
            session_handle.last_value_publisher(
                f"node_metrics/{self.node_id}/socket_connections"
            )
        )

    async def run(self):
        logger.info("Starting Worker")

        info_gatherer: InfoGatherer = InfoGatherer(self._sh, self.node_id)

        try:
            async with self._tg as tg:
                tg.start_soon(info_gatherer.run)
                tg.start_soon(self.plan_step)
                tg.start_soon(self._event_applier)
                tg.start_soon(self._poll_connection_updates)
                tg.start_soon(self._reconcile_custom_cards)
                tg.start_soon(self._listen_to_mailbox)
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
            with anyio.CancelScope(shield=True):
                for publisher in self._primary_desired_instance_publishers.values():
                    await publisher.delete()
                await self._socket_connections_publisher.delete()
            self._stopped.set()

    async def _listen_to_mailbox(self):
        ta = TypeAdapter[Mail](Mail)
        while (mail := await self.mailbox.recv()) is not None:
            try:
                mail = ta.validate_json(mail)
            except ValidationError:
                logger.warning(f"discarding corrupt mail {mail}")
                continue

            match mail:
                case JoinInstance(instance=instance):
                    self.desired_instances[instance.instance_id] = instance
                    if instance.primary_output_node() == self.node_id:
                        await self._publish_primary_desired_instance(instance)
                case LeaveInstance(instance_id=instance_id):
                    self.desired_instances.pop(instance_id, None)
                    await self._delete_primary_desired_instance(instance_id)

    async def _publish_primary_desired_instance(self, instance: Instance) -> None:
        publisher = self._primary_desired_instance_publishers.get(instance.instance_id)
        if publisher is None:
            publisher = self._sh.last_value_publisher(
                f"node_metrics/{self.node_id}/desired_instances/{instance.instance_id}"
            )
            self._primary_desired_instance_publishers[instance.instance_id] = publisher
        await publisher.put(instance.model_dump_json())

    async def _delete_primary_desired_instance(self, instance_id: InstanceId) -> None:
        publisher = self._primary_desired_instance_publishers.pop(instance_id, None)
        if publisher is not None:
            await publisher.delete()

    async def _forget_desired_instance_locally(self, instance_id: InstanceId) -> None:
        self.desired_instances.pop(instance_id, None)
        self._primary_runner_missing_since.pop(instance_id, None)
        await self._delete_primary_desired_instance(instance_id)

    def _update_primary_runner_missing_since(
        self, live_runner_ids: set[RunnerId]
    ) -> None:
        now = anyio.current_time()
        for instance_id, instance in list(self.desired_instances.items()):
            primary_runner_id = instance.shard_assignments.shards[
                instance.shard_assignments.primary_output_node
            ].runner_id
            if primary_runner_id in live_runner_ids:
                self._primary_runner_missing_since.pop(instance_id, None)
                continue

            self._primary_runner_missing_since.setdefault(instance_id, now)

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
                event = event.event

                if isinstance(event, InstanceDeleted):
                    self._instance_backoff.reset(event.instance_id)

    async def _reconcile_custom_cards(self) -> None:
        storage = self._sh.storage_interface()
        while True:
            await anyio.sleep(10)
            target: list[ModelId] = []
            for _, value in (await storage.dump("custom_model_cards/")).items():
                try:
                    card = ModelCard.model_validate_json(value)
                except ValidationError:
                    continue
                target.append(card.model_id)
                if model_cards.card_cache.get(card.model_id) == card:
                    continue
                logger.info(f"Registered new custom model card for {card.model_id}")
                await model_cards.card_cache.save(card)

            for card in await model_cards.card_cache.list_all():
                if card.model_id not in target:
                    await model_cards.card_cache.delete(card.model_id)

    async def plan_step(self):
        while True:
            await anyio.sleep(0.1)
            state = self.state.with_aggregator(self.aggregator)
            live_runner_ids = set(state.runners) | set(self.runners)
            now = anyio.current_time()
            self._update_primary_runner_missing_since(live_runner_ids)
            task: Task | None = plan(
                self.node_id,
                self.runners,
                state.downloads,  # comes from with_agg
                self.desired_instances,  # comes from mailbox
                state.runners,  # comes from with_agg
                self._instance_backoff,
                self._download_backoff,
                live_runner_ids=live_runner_ids,
                primary_runner_missing_since=self._primary_runner_missing_since,
                now=now,
                primary_runner_missing_timeout_seconds=PRIMARY_RUNNER_MISSING_TIMEOUT_SECONDS,
            )
            if task is None:
                continue

            if isinstance(task, ForgetInstance):
                logger.warning(
                    f"Instance {task.instance_id} primary runner missing for "
                    f"{PRIMARY_RUNNER_MISSING_TIMEOUT_SECONDS:g}s; forgetting locally"
                )
                await self._forget_desired_instance_locally(task.instance_id)
                continue

            if isinstance(task, CreateRunner):
                iid = task.instance_id
                if self._instance_backoff.attempts(iid) >= EXO_MAX_INSTANCE_RETRIES:
                    logger.warning(
                        f"Instance {iid} exceeded {EXO_MAX_INSTANCE_RETRIES} retries, requesting deletion"
                    )
                    await self._forget_desired_instance_locally(iid)
                    await self.command_sender.send(
                        ForwarderCommand(
                            origin=self._system_id,
                            command=DeleteInstance(),
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
        if (instance := self.desired_instances.get(task.instance_id)) is not None:
            for rid in instance.runners_for(self.node_id):
                await self.runners[rid].start_task(task)

    async def _create_supervisor(self, task: CreateRunner) -> RunnerSupervisor:
        """Creates and stores a new AssignedRunner with initial downloading status."""
        task_responder = (
            self._sh.task_responder(task.instance_id)
            if task.bound_instance.is_primary_output_node()
            else None
        )
        runner = await RunnerSupervisor.create(
            bound_instance=task.bound_instance,
            event_sender=self.event_sender.clone(),
            task_assignment_subscriber=self._sh.last_value_subscriber(
                f"task_assignments/{task.instance_id}/*"
            ),
            runner_status_publisher=self._sh.last_value_publisher(
                f"node_metrics/{self.node_id}/runners/{task.bound_instance.bound_runner_id}/status"
            ),
            task_responder=task_responder,
        )
        self.runners[task.bound_instance.bound_runner_id] = runner
        self._tg.start_soon(runner.run)
        return runner

    async def _poll_connection_updates(self):
        while True:
            state = self.state.with_aggregator(self.aggregator)
            conns: defaultdict[NodeId, set[str]] = defaultdict(set)
            async for ip, nid in check_reachable(
                state.topology,
                self.node_id,
                state.node_network,
                api_port=self.api_port,
            ):
                if ip in conns[nid]:
                    continue
                conns[nid].add(ip)

            socket_connections = SocketConnections(
                connections={
                    nid: [
                        SocketConnection(
                            # nonsense multiaddr
                            sink_multiaddr=Multiaddr(
                                address=f"/ip4/{ip}/tcp/{self.api_port}"
                            )
                            if "." in ip
                            # nonsense multiaddr
                            else Multiaddr(address=f"/ip6/{ip}/tcp/{self.api_port}"),
                        )
                        for ip in sorted(ips)
                    ]
                    for nid, ips in conns.items()
                }
            )
            await self._socket_connections_publisher.put(
                socket_connections.model_dump_json()
            )

            await anyio.sleep(10)
