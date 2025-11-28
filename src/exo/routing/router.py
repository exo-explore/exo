from copy import copy
from itertools import count
from math import inf
from os import PathLike
from pathlib import Path
from typing import cast

import anyio
from anyio import (
    BrokenResourceError,
    ClosedResourceError,
    create_task_group,
    sleep_forever,
)
from anyio.abc import TaskGroup
from exo_pyo3_bindings import (
    Keypair,
    RustNetworkingHandle,
    RustReceiver,
    RustSender,
)
from filelock import FileLock
from loguru import logger

from exo import __version__
from exo.shared.constants import EXO_NODE_ID_KEYPAIR
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.pydantic_ext import CamelCaseModel

from .connection_message import ConnectionMessage
from .topics import CONNECTION_MESSAGES, PublishPolicy, TypedTopic


# A significant current limitation of the TopicRouter is that it is not capable
# of preventing feedback, as it does not ask for a system id so cannot tell
# which message is coming/going to which system.
# This is currently only relevant for Election
class TopicRouter[T: CamelCaseModel]:
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
        self.networking_sender: RustSender | None = None
        self.networking_receiver: RustReceiver | None = None

        self._tg: TaskGroup = create_task_group()

    async def run(self):
        async with self._tg as tg:
            tg.start_soon(self.receive_loop)            

    async def receive_loop(self):
        logger.debug(f"Topic Router {self.topic} ready to send")
        with self.receiver as items:
            async for item in items:
                # Check if we should send to network
                if self.topic.publish_policy is PublishPolicy.Always and self.networking_sender is not None:
                    await self._send_out(item)
                # Then publish to all senders
                await self.publish(item)
        logger.debug(f"Shut down Topic Router {self.topic}")

    async def net_receive_loop(self):
        assert self.networking_receiver is not None
        while True:
            item = self.topic.deserialize(await self.networking_receiver.receive())
            logger.debug(f"Got {item} from network")
            await self.publish(item)

    def subscribe_with(self, net_send: RustSender, net_recv: RustReceiver):
        self.networking_sender = net_send
        self.networking_receiver = net_recv
        self._tg.start_soon(self.net_receive_loop)

    async def shutdown(self):
        # Close all the things!
        for sender in self.senders:
            sender.close()
        self._sender.close()
        self.receiver.close()

    async def publish(self, item: T):
        """
        Publish item T on this topic to all senders.
        NB: this sends to ALL receivers, potentially including receivers held by the object doing the sending.
        You should handle your own output if you hold a sender + receiver pair.
        """
        to_clear: set[Sender[T]] = set()
        for sender in copy(self.senders):
            try:
                await sender.send(item)
            except (ClosedResourceError, BrokenResourceError):
                to_clear.add(sender)
        self.senders -= to_clear

    def new_sender(self) -> Sender[T]:
        return self._sender.clone()

    async def _send_out(self, item: T):
        assert self.networking_sender is not None
        logger.trace(f"TopicRouter {self.topic.topic} sending {item}")
        await self.networking_sender.send(self.topic.serialize(item))


class Router:
    @classmethod
    async def create(cls, identity: Keypair) -> "Router":
        return cls(handle=await RustNetworkingHandle.create(identity, __version__))

    def __init__(self, handle: RustNetworkingHandle):
        self.topic_routers: dict[str, TopicRouter[CamelCaseModel]] = {}
        self._unsubbed: list[str] = []
        self._net: RustNetworkingHandle = handle
        self._id_count = count()
        self._tg: TaskGroup | None = None

    async def register_topic[T: CamelCaseModel](self, topic: TypedTopic[T]):
        assert self._tg is None, "Attempted to register topic after setup time"
        router = TopicRouter[T](topic)
        self.topic_routers[topic.topic] = cast(TopicRouter[CamelCaseModel], router)
        self._unsubbed.append(topic.topic)

    def sender[T: CamelCaseModel](self, topic: TypedTopic[T]) -> Sender[T]:
        router = self.topic_routers.get(topic.topic, None)
        # There's gotta be a way to do this without THIS many asserts
        assert router is not None
        assert router.topic == topic
        sender = cast(TopicRouter[T], router).new_sender()
        return sender

    def receiver[T: CamelCaseModel](self, topic: TypedTopic[T]) -> Receiver[T]:
        router = self.topic_routers.get(topic.topic, None)
        # There's gotta be a way to do this without THIS many asserts

        assert router is not None
        assert router.topic == topic
        assert router.topic.model_type == topic.model_type

        send, recv = channel[T]()
        router.senders.add(cast(Sender[CamelCaseModel], send))

        return recv

    async def run(self):
        logger.debug("Starting Router")
        async with create_task_group() as tg:
            self._tg = tg
            for topic in self.topic_routers:
                router = self.topic_routers[topic]
                tg.start_soon(router.run)
            tg.start_soon(self._networking_recv_connection_messages)
            # Router only shuts down if you cancel it.
            await sleep_forever()

    async def shutdown(self):
        logger.debug("Shutting down Router")
        if not self._tg:
            return
        self._tg.cancel_scope.cancel()

    async def _networking_recv_connection_messages(self):
        recv = await self._net.get_connection_receiver()
        while True:
            message = await recv.receive()
            await anyio.sleep(0.2)
            logger.trace(
                f"Received message on connection_messages with payload {message}"
            )
            to_clear: list[str] = []
            for topic in self._unsubbed:
                try:
                    rsend, rrecv = await self._net.subscribe(topic)
                    logger.info(f"Subscribed to peer on {topic}")
                    to_clear.append(topic)
                    self.topic_routers[topic].subscribe_with(rsend, rrecv)
                # TODO: real error
                except RuntimeError:
                    pass
            if to_clear:
                assert to_clear == self._unsubbed
                self._unsubbed = [i for i in self._unsubbed if i not in to_clear]

            if CONNECTION_MESSAGES.topic in self.topic_routers:
                router = self.topic_routers[CONNECTION_MESSAGES.topic]
                assert router.topic.model_type == ConnectionMessage
                router = cast(TopicRouter[ConnectionMessage], router)
                await router.publish(ConnectionMessage.from_rust(message))


def get_node_id_keypair(
    path: str | bytes | PathLike[str] | PathLike[bytes] = EXO_NODE_ID_KEYPAIR,
) -> Keypair:
    """
    Obtains the :class:`Keypair` associated with this node-ID.
    Obtain the :class:`PeerId` by from it.
    """

    def lock_path(path: str | bytes | PathLike[str] | PathLike[bytes]) -> Path:
        return Path(str(path) + ".lock")

    # operate with cross-process lock to avoid race conditions
    with FileLock(lock_path(path)):
        with open(path, "a+b") as f:  # opens in append-mode => starts at EOF
            # if non-zero EOF, then file exists => use to get node-ID
            if f.tell() != 0:
                f.seek(0)  # go to start & read postcard-encoded bytes
                postcard_encoded = f.read()

                try:  # if decoded successfully, save & return
                    return Keypair.from_postcard_encoding(postcard_encoded)
                except ValueError as e:  # on runtime error, assume corrupt file
                    logger.warning(f"Encountered error when trying to get keypair: {e}")

        # if no valid credentials, create new ones and persist
        with open(path, "w+b") as f:
            keypair = Keypair.generate_ed25519()
            f.write(keypair.to_postcard_encoding())
            return keypair
