from dataclasses import dataclass
from typing import Callable

from anyio import fail_after

from exo.routing.topics import ConnectionMessage, ForwarderCommand, ForwarderEvent
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.common import NodeId
from exo.shared.types.events import ChunkGenerated, Event, TaskStateUpdated
from exo.shared.types.tasks import TaskId, TaskStatus
from exo.utils.channels import Receiver, Sender, channel
from exo.worker.download.shard_downloader import NoopShardDownloader, ShardDownloader
from exo.worker.main import Worker


@dataclass
class WorkerMailbox:
    sender: Sender[ForwarderEvent]
    receiver: Receiver[ForwarderEvent]
    counter: int = 0

    async def append_events(self, events: list[Event], *, origin: NodeId):
        for event in events:
            await self.sender.send(
                ForwarderEvent(
                    origin=origin,
                    event=event,
                    origin_idx=self.counter,
                )
            )
            self.counter += 1

    async def receive(self) -> ForwarderEvent:
        return await self.receiver.receive()

    def collect(self) -> list[ForwarderEvent]:
        # Clear out the test mailboxes currently held events
        return self.receiver.collect()


def create_worker_void_mailbox(
    node_id: NodeId, shard_downloader: ShardDownloader | None = None
) -> Worker:
    if shard_downloader is None:
        shard_downloader = NoopShardDownloader()
    return Worker(
        node_id,
        shard_downloader=shard_downloader,
        initial_connection_messages=[],
        connection_message_receiver=channel[ConnectionMessage]()[1],
        global_event_receiver=channel[ForwarderEvent]()[1],
        local_event_sender=channel[ForwarderEvent]()[0],
        command_sender=channel[ForwarderCommand]()[0],
    )


def create_worker_and_mailbox(
    node_id: NodeId, shard_downloader: ShardDownloader | None = None
) -> tuple[Worker, WorkerMailbox]:
    if shard_downloader is None:
        shard_downloader = NoopShardDownloader()

    lsend, receiver = channel[ForwarderEvent]()
    sender, grecv = channel[ForwarderEvent]()
    worker = Worker(
        node_id,
        shard_downloader=shard_downloader,
        initial_connection_messages=[],
        connection_message_receiver=channel[ConnectionMessage]()[1],
        global_event_receiver=grecv,
        local_event_sender=lsend,
        command_sender=channel[ForwarderCommand]()[0],
    )
    return worker, WorkerMailbox(sender, receiver)


def create_worker_with_old_mailbox(
    node_id: NodeId,
    mailbox: WorkerMailbox,
    shard_downloader: ShardDownloader | None = None,
) -> Worker:
    if shard_downloader is None:
        shard_downloader = NoopShardDownloader()
    # This function is subtly complex, come talk to Evan if you want to know what it's actually doing.
    worker = Worker(
        node_id,
        shard_downloader=shard_downloader,
        initial_connection_messages=[],
        connection_message_receiver=channel[ConnectionMessage]()[1],
        global_event_receiver=mailbox.sender.clone_receiver(),
        local_event_sender=mailbox.receiver.clone_sender(),
        command_sender=channel[ForwarderCommand]()[0],
    )
    return worker


async def read_streaming_response(
    global_event_receiver: WorkerMailbox, filter_task: TaskId | None = None
) -> tuple[bool, bool, str, int]:
    # Read off all events - these should be our GenerationChunk events
    seen_task_started = 0
    seen_task_finished = 0
    response_string = ""
    finish_reason: str | None = None
    token_count = 0
    extra_events: list[Event] = []

    event = (await global_event_receiver.receive()).event
    extra_events.append(event)

    from loguru import logger

    logger.info("STARTING READ")

    with fail_after(10.0):
        if filter_task:
            while not (
                isinstance(event, TaskStateUpdated)
                and event.task_status == TaskStatus.Running
                and event.task_id == filter_task
            ):
                event = (await global_event_receiver.receive()).event
                extra_events.append(event)

        for event in extra_events:
            if isinstance(event, TaskStateUpdated):
                if event.task_status == TaskStatus.Running:
                    seen_task_started += 1
                if event.task_status == TaskStatus.Complete:
                    seen_task_finished += 1
            if isinstance(event, ChunkGenerated) and isinstance(
                event.chunk, TokenChunk
            ):
                response_string += event.chunk.text
                token_count += 1
                if event.chunk.finish_reason:
                    finish_reason = event.chunk.finish_reason

        while not seen_task_finished:
            event = (await global_event_receiver.receive()).event
            if isinstance(event, TaskStateUpdated):
                if event.task_status == TaskStatus.Running:
                    seen_task_started += 1
                if event.task_status == TaskStatus.Complete:
                    seen_task_finished += 1
            if isinstance(event, ChunkGenerated) and isinstance(
                event.chunk, TokenChunk
            ):
                response_string += event.chunk.text
                token_count += 1
                if event.chunk.finish_reason:
                    finish_reason = event.chunk.finish_reason

    logger.info(f"finish reason {finish_reason}")

    return seen_task_started == 1, seen_task_finished == 1, response_string, token_count


async def until_event_with_timeout[T](
    global_event_receiver: WorkerMailbox,
    event_type: type[T],
    multiplicity: int = 1,
    condition: Callable[[T], bool] = lambda x: True,
    timeout: float = 30.0,
) -> None:
    times_seen = 0

    with fail_after(timeout):
        while times_seen < multiplicity:
            event = (await global_event_receiver.receive()).event
            if isinstance(event, event_type):
                print(f"Wow! We got a {event}")
                print(
                    f"But condition? {condition(event) if isinstance(event, event_type) else False}"
                )
            if event and isinstance(event, event_type) and condition(event):
                times_seen += 1
