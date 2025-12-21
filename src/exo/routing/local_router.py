"""
Local-only Router for single-node operation without libp2p networking.

This is useful for:
- Testing on platforms where libp2p has permission issues (e.g., proot/Android)
- Single-node development and debugging
- Running inference without distributed networking

Set EXO_LOCAL_MODE=1 to use this router instead of the networked Router.
"""

from copy import copy
from itertools import count
from math import inf
from typing import cast

from anyio import (
    BrokenResourceError,
    ClosedResourceError,
    create_task_group,
    sleep_forever,
)
from anyio.abc import TaskGroup
from loguru import logger

from exo.utils.channels import Receiver, Sender, channel
from exo.utils.pydantic_ext import CamelCaseModel

from .topics import TypedTopic


class LocalTopicRouter[T: CamelCaseModel]:
    """A topic router that only routes messages locally (no network)."""

    def __init__(
        self,
        topic: TypedTopic[T],
        max_buffer_size: float = inf,
    ):
        self.topic: TypedTopic[T] = topic
        self.senders: set[Sender[T]] = set()
        send, recv = channel[T]()
        self.receiver: Receiver[T] = recv
        self._sender: Sender[T] = send

    async def run(self):
        logger.debug(f"Local Topic Router {self.topic} ready")
        with self.receiver as items:
            async for item in items:
                # In local mode, we just publish to all local senders
                await self.publish(item)

    async def shutdown(self):
        logger.debug(f"Shutting down Local Topic Router {self.topic}")
        for sender in self.senders:
            sender.close()
        self._sender.close()
        self.receiver.close()

    async def publish(self, item: T):
        """Publish item T to all local senders."""
        to_clear: set[Sender[T]] = set()
        for sender in copy(self.senders):
            try:
                await sender.send(item)
            except (ClosedResourceError, BrokenResourceError):
                to_clear.add(sender)
        self.senders -= to_clear

    async def publish_bytes(self, data: bytes):
        await self.publish(self.topic.deserialize(data))

    def new_sender(self) -> Sender[T]:
        return self._sender.clone()


class LocalRouter:
    """
    A local-only router that doesn't require libp2p networking.
    
    This allows exo to run in single-node mode on platforms where
    libp2p has permission issues (like proot on Android).
    """

    @classmethod
    def create(cls) -> "LocalRouter":
        logger.warning("Running in LOCAL MODE - no distributed networking")
        return cls()

    def __init__(self):
        self.topic_routers: dict[str, LocalTopicRouter[CamelCaseModel]] = {}
        self._id_count = count()
        self._tg: TaskGroup | None = None

    async def register_topic[T: CamelCaseModel](self, topic: TypedTopic[T]):
        assert self._tg is None, "Attempted to register topic after setup time"
        router = LocalTopicRouter[T](topic)
        self.topic_routers[topic.topic] = cast(LocalTopicRouter[CamelCaseModel], router)
        logger.debug(f"Registered local topic: {topic.topic}")

    def sender[T: CamelCaseModel](self, topic: TypedTopic[T]) -> Sender[T]:
        router = self.topic_routers.get(topic.topic, None)
        assert router is not None
        assert router.topic == topic
        sender = cast(LocalTopicRouter[T], router).new_sender()
        return sender

    def receiver[T: CamelCaseModel](self, topic: TypedTopic[T]) -> Receiver[T]:
        router = self.topic_routers.get(topic.topic, None)
        assert router is not None
        assert router.topic == topic
        assert router.topic.model_type == topic.model_type

        send, recv = channel[T]()
        router.senders.add(cast(Sender[CamelCaseModel], send))

        return recv

    async def run(self):
        logger.debug("Starting Local Router (no networking)")
        async with create_task_group() as tg:
            self._tg = tg
            for topic in self.topic_routers:
                router = self.topic_routers[topic]
                tg.start_soon(router.run)
            # Local router just waits - no network tasks
            await sleep_forever()

    async def shutdown(self):
        logger.debug("Shutting down Local Router")
        if not self._tg:
            return
        self._tg.cancel_scope.cancel()

