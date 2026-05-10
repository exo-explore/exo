from collections.abc import Sequence
from copy import copy
from itertools import count
from math import inf
from os import PathLike
from pathlib import Path
from typing import cast

from anyio import (
    BrokenResourceError,
    ClosedResourceError,
    current_time,
    move_on_after,
    sleep_forever,
)
from exo_pyo3_bindings import (
    AllQueuesFullError,
    Keypair,
    MessageTooLargeError,
    NetworkingHandle,
    NoPeersSubscribedToTopicError,
    PyFromSwarm,
)
from filelock import FileLock
from loguru import logger

from exo.shared.constants import EXO_LEGACY_NODE_ID_KEYPAIR, EXO_NODE_ID_KEYPAIR
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.pydantic_ext import FrozenModel
from exo.utils.task_group import TaskGroup

from .connection_message import ConnectionMessage
from .topics import CONNECTION_MESSAGES, PublishPolicy, TypedTopic


# A significant current limitation of the TopicRouter is that it is not capable
# of preventing feedback, as it does not ask for a system id so cannot tell
# which message is coming/going to which system.
# This is currently only relevant for Election
class TopicRouter[T: FrozenModel]:
    def __init__(
        self,
        topic: TypedTopic[T],
        networking_sender: Sender[tuple[str, bytes]],
        max_buffer_size: float = inf,
    ):
        self.topic: TypedTopic[T] = topic
        self.senders: set[Sender[T]] = set()
        send, recv = channel[T]()
        self.receiver: Receiver[T] = recv
        self._sender: Sender[T] = send
        self.networking_sender: Sender[tuple[str, bytes]] = networking_sender

    async def run(self):
        logger.debug(f"Topic Router {self.topic} ready to send")
        with self.receiver as items:
            async for item in items:
                # Check if we should send to network
                if (
                    len(self.senders) == 0
                    and self.topic.publish_policy is PublishPolicy.Minimal
                ):
                    await self._send_out(item)
                    continue
                if self.topic.publish_policy is PublishPolicy.Always:
                    await self._send_out(item)
                # Then publish to all senders
                await self.publish(item)

    async def shutdown(self):
        logger.debug(f"Shutting down Topic Router {self.topic}")
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

    async def publish_bytes(self, data: bytes):
        await self.publish(self.topic.deserialize(data))

    def new_sender(self) -> Sender[T]:
        return self._sender.clone()

    async def _send_out(self, item: T):
        logger.trace(f"TopicRouter {self.topic.topic} sending {item}")
        await self.networking_sender.send(
            (str(self.topic.topic), self.topic.serialize(item))
        )


class Router:
    @classmethod
    def create(
        cls,
        identity: Keypair,
        bootstrap_peers: Sequence[str] = (),
        listen_port: int = 0,
    ) -> "Router":
        return cls(
            handle=NetworkingHandle(identity, list(bootstrap_peers), listen_port)
        )

    def __init__(self, handle: NetworkingHandle):
        self.topic_routers: dict[str, TopicRouter[FrozenModel]] = {}
        send, recv = channel[tuple[str, bytes]]()
        self.networking_receiver: Receiver[tuple[str, bytes]] = recv
        self._net: NetworkingHandle = handle
        self._tmp_networking_sender: Sender[tuple[str, bytes]] | None = send
        self._id_count = count()
        self._tg: TaskGroup = TaskGroup()
        self._publish_failure_counts: dict[str, int] = {}
        self._publish_failure_first_seen: dict[str, float] = {}

    async def register_topic[T: FrozenModel](self, topic: TypedTopic[T]):
        send = self._tmp_networking_sender
        if send:
            self._tmp_networking_sender = None
        else:
            send = self.networking_receiver.clone_sender()
        router = TopicRouter[T](topic, send)
        self.topic_routers[topic.topic] = cast(TopicRouter[FrozenModel], router)
        if self._tg.is_running():
            await self._networking_subscribe(topic.topic)

    def sender[T: FrozenModel](self, topic: TypedTopic[T]) -> Sender[T]:
        router = self.topic_routers.get(topic.topic, None)
        # There's gotta be a way to do this without THIS many asserts
        assert router is not None
        assert router.topic == topic
        sender = cast(TopicRouter[T], router).new_sender()
        return sender

    def receiver[T: FrozenModel](self, topic: TypedTopic[T]) -> Receiver[T]:
        router = self.topic_routers.get(topic.topic, None)
        # There's gotta be a way to do this without THIS many asserts

        assert router is not None
        assert router.topic == topic
        assert router.topic.model_type == topic.model_type

        send, recv = channel[T]()
        router.senders.add(cast(Sender[FrozenModel], send))

        return recv

    async def run(self):
        logger.debug("Starting Router")
        try:
            async with self._tg as tg:
                for topic in self.topic_routers:
                    router = self.topic_routers[topic]
                    tg.start_soon(router.run)
                tg.start_soon(self._networking_recv)
                tg.start_soon(self._networking_publish)
                # subscribe to pending topics
                for topic in self.topic_routers:
                    await self._networking_subscribe(topic)
                # Router only shuts down if you cancel it.
                await sleep_forever()
        finally:
            with move_on_after(1, shield=True):
                for topic in self.topic_routers:
                    await self._networking_unsubscribe(str(topic))

    async def shutdown(self):
        logger.debug("Shutting down Router")
        self._tg.cancel_tasks()

    async def _networking_subscribe(self, topic: str):
        await self._net.gossipsub_subscribe(topic)
        logger.info(f"Subscribed to {topic}")

    async def _networking_unsubscribe(self, topic: str):
        await self._net.gossipsub_unsubscribe(topic)
        logger.info(f"Unsubscribed from {topic}")

    async def _networking_recv(self):
        try:
            while True:
                from_swarm = await self._net.recv()
                logger.debug(from_swarm)
                match from_swarm:
                    case PyFromSwarm.Message(origin, topic, data):
                        logger.trace(
                            f"Received message on {topic} from {origin} with payload {data}"
                        )
                        if topic not in self.topic_routers:
                            logger.warning(
                                f"Received message on unknown or inactive topic {topic}"
                            )
                            continue
                        router = self.topic_routers[topic]
                        await router.publish_bytes(data)
                    case PyFromSwarm.Connection():
                        message = ConnectionMessage.from_update(from_swarm)
                        logger.trace(
                            f"Received message on connection_messages with payload {message}"
                        )
                        if CONNECTION_MESSAGES.topic in self.topic_routers:
                            router = self.topic_routers[CONNECTION_MESSAGES.topic]
                            assert router.topic.model_type == ConnectionMessage
                            router = cast(TopicRouter[ConnectionMessage], router)
                            await router.publish(message)
                    case _:
                        logger.critical(
                            "failed to exhaustively check FromSwarm messages - logic error"
                        )
        except Exception as exception:
            logger.opt(exception=exception).error(
                "Gossipsub receive loop terminated unexpectedly"
            )
            raise

    async def _networking_publish(self):
        with self.networking_receiver as networked_items:
            async for topic, data in networked_items:
                try:
                    logger.trace(f"Sending message on {topic} with payload {data}")
                    if len(data) > 1024 * 1024:
                        logger.warning(
                            "Sending overlarge payload, network performance may be "
                            f"temporarily degraded topic={topic} payload_bytes={len(data)}"
                        )
                    await self._net.gossipsub_publish(topic, data)
                    self._clear_publish_failures(topic)
                except NoPeersSubscribedToTopicError:
                    self._record_publish_failure(
                        topic=topic,
                        payload_bytes=len(data),
                        reason="no_peers_subscribed",
                        log_level="DEBUG",
                    )
                except AllQueuesFullError:
                    self._record_publish_failure(
                        topic=topic,
                        payload_bytes=len(data),
                        reason="all_peer_queues_full",
                        log_level="WARNING",
                    )
                except MessageTooLargeError:
                    self._record_publish_failure(
                        topic=topic,
                        payload_bytes=len(data),
                        reason="message_too_large",
                        log_level="WARNING",
                    )

    def _record_publish_failure(
        self, *, topic: str, payload_bytes: int, reason: str, log_level: str
    ) -> None:
        key = f"{topic}:{reason}"
        count = self._publish_failure_counts.get(key, 0) + 1
        self._publish_failure_counts[key] = count
        first_seen = self._publish_failure_first_seen.setdefault(key, current_time())
        elapsed_seconds = current_time() - first_seen
        if count == 1 or count % 10 == 0:
            logger.log(
                log_level,
                "Gossipsub publish failed "
                f"topic={topic} reason={reason} payload_bytes={payload_bytes} "
                f"consecutive_failures={count} "
                f"failure_window_seconds={elapsed_seconds:.3f}",
            )

    def _clear_publish_failures(self, topic: str) -> None:
        cleared = [
            key for key in self._publish_failure_counts if key.startswith(f"{topic}:")
        ]
        for key in cleared:
            count = self._publish_failure_counts.pop(key)
            first_seen = self._publish_failure_first_seen.pop(key, current_time())
            logger.info(
                "Gossipsub publish recovered "
                f"topic={topic} reason={key.removeprefix(f'{topic}:')} "
                f"previous_failures={count} "
                f"failure_window_seconds={current_time() - first_seen:.3f}"
            )


def get_node_id_keypair(
    path: str | bytes | PathLike[str] | PathLike[bytes] = EXO_NODE_ID_KEYPAIR,
    legacy_path: str | bytes | PathLike[str] | PathLike[bytes] | None = (
        EXO_LEGACY_NODE_ID_KEYPAIR
    ),
    process_scope: int | str | None = None,
) -> Keypair:
    """
    Obtains the :class:`Keypair` associated with this node-ID.
    Obtain the :class:`PeerId` by from it.

    Codex P1 (PR #16 round-(N+2), router.py:297): when ``process_scope``
    is provided, the on-disk keypair filename is suffixed with the
    scope (typically the libp2p / peer-download port the caller has
    chosen). This preserves *per-process* node identity isolation
    when multiple exo processes run on the same host -- the new
    same-host multi-node workflow added in this PR (distinct
    peer-download ports per process) needs each process to have a
    distinct ``NodeId`` so peer discovery's ``peer_node_id ==
    node_id`` self-skip and routing's unique-node-id assumptions
    hold. Single-process deployments leave ``process_scope=None``
    and continue using the shared persistent keypair file.

    On first call after the upgrade, if the new ``path`` (config dir)
    has no keypair yet but the legacy cache-dir ``legacy_path`` does,
    the legacy file is moved to ``path`` so the node retains its
    identity across the relocation. Migration is best-effort: if
    moving fails (e.g. cross-device link errors on Linux when
    ``XDG_*`` dirs span filesystems), the legacy bytes are copied
    instead. Either way, the legacy file is removed once the new
    location holds a valid keypair so subsequent calls do not need
    to re-check. Codex P2 (PR #16 round-(N+2), router.py:322): the
    migration is performed INSIDE the file lock so two concurrent
    processes can't both pass the existence check and then race
    each other into divergent in-memory vs. on-disk identities.
    Codex P1 (PR #16 round-(N+13), router.py:359): when callers
    pass distinct ``process_scope`` values, the per-scope lock
    above does NOT serialize legacy adoption across scopes, so a
    second lock keyed on the (unscoped) legacy path is acquired
    before invoking the migrator -- otherwise the cross-device
    byte-copy fallback can produce duplicate ``NodeId``s.
    """
    base_path = Path(str(path))
    resolved_path = (
        _scoped_keypair_path(base_path, process_scope)
        if process_scope is not None
        else base_path
    )

    # The legacy cache file pre-dates the per-process scoping change
    # so it is intentionally NOT scope-suffixed. We migrate it as a
    # one-shot identity adoption for whichever process happens to
    # boot first; subsequent processes (with different scopes) will
    # observe the legacy file already gone and start with fresh
    # keypairs, which is exactly what per-process isolation requires.
    resolved_legacy: Path | None = (
        Path(str(legacy_path)) if legacy_path is not None else None
    )

    def lock_path(p: str | bytes | PathLike[str] | PathLike[bytes]) -> Path:
        return Path(str(p) + ".lock")

    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    # operate with cross-process lock to avoid race conditions.
    # The migration MUST run inside this lock so two processes that
    # boot simultaneously can't both pass the migrator's existence
    # check, race the keypair generation, and end up with the same
    # on-disk file but divergent in-memory identities.
    with FileLock(lock_path(resolved_path)):
        if resolved_legacy is not None:
            # Codex P1 (PR #16 round-(N+13), router.py:359):
            # serialize legacy adoption across ALL ``process_scope``
            # values. The outer ``resolved_path`` lock is per-scope,
            # so two same-host processes with different scopes
            # acquire DIFFERENT lock files and can each enter
            # ``_migrate_legacy_node_id_keypair`` concurrently. In
            # the cross-device fallback path -- where ``replace()``
            # raises ``OSError`` and the migrator falls back to a
            # ``read_bytes`` + ``write_bytes`` + ``unlink``
            # sequence -- both processes can read the same legacy
            # keypair before either unlinks it, then each writes
            # those bytes into its own scoped file. Result: two
            # nodes claiming the same ``NodeId`` despite distinct
            # scopes, breaking routing's unique-identity and
            # election's tiebreaker invariants. A lock keyed on the
            # legacy path (which is intentionally NOT scope-suffixed
            # because it pre-dates scoping) serializes migration so
            # exactly one scope wins legacy adoption and any
            # concurrent peers observe the file already gone and
            # generate fresh keypairs -- the documented "first
            # process boots wins" semantic. Released immediately
            # after migration so unrelated keypair I/O on other
            # scopes isn't blocked on identity housekeeping.
            with FileLock(lock_path(resolved_legacy)):
                _migrate_legacy_node_id_keypair(resolved_path, resolved_legacy)

        with open(resolved_path, "a+b") as f:  # opens in append-mode => starts at EOF
            # if non-zero EOF, then file exists => use to get node-ID
            if f.tell() != 0:
                f.seek(0)  # go to start & read protobuf-encoded bytes
                protobuf_encoded = f.read()

                try:  # if decoded successfully, save & return
                    return Keypair.from_bytes(protobuf_encoded)
                except ValueError as e:  # on runtime error, assume corrupt file
                    logger.warning(f"Encountered error when trying to get keypair: {e}")

        # if no valid credentials, create new ones and persist
        with open(resolved_path, "w+b") as f:
            keypair = Keypair.generate()
            f.write(keypair.to_bytes())
            return keypair


def _scoped_keypair_path(base: Path, scope: int | str) -> Path:
    """Return ``base`` with the process scope inserted before the
    suffix (e.g. ``node_id.keypair`` + scope ``52415`` ->
    ``node_id.52415.keypair``).

    We insert the scope as a stem-suffix rather than as a directory
    so concurrent processes on the same host share the parent dir
    (and the file lock's inode-level coordination still works for
    legacy-migration safety) while their identity files remain
    distinct. Scope is rendered with ``str()`` so callers can pass
    a port number, a UUID, a hostname, etc.
    """
    suffix = base.suffix or ".keypair"
    stem = base.stem if base.suffix else base.name
    return base.parent / f"{stem}.{scope}{suffix}"


def _migrate_legacy_node_id_keypair(
    new_path: Path,
    legacy_path: Path,
) -> None:
    """One-shot migrator for the cache→config relocation of the
    node-ID keypair (Codex P1 PR #16 round 5).

    Idempotent and best-effort: only acts when ``new_path`` is
    absent and ``legacy_path`` exists. Falls back to byte copy if
    ``rename`` fails (cross-device, permissions, etc.). On any
    exception we log and bail -- the caller will then generate a
    fresh keypair, which is suboptimal but better than crashing
    startup over identity-file housekeeping.
    """
    try:
        if new_path.exists() or not legacy_path.exists():
            return
        # Ensure the destination directory exists for either the
        # ``replace`` (which silently no-ops on missing parent on some
        # platforms but raises ``ENOENT`` on others) or the byte-copy
        # fallback. ``get_node_id_keypair`` already creates this dir
        # for the same reason; doing it again here keeps the migrator
        # safely callable from tests in isolation.
        new_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            legacy_path.replace(new_path)
        except OSError as rename_err:
            logger.debug(
                f"Cross-device rename of legacy keypair failed ({rename_err}); "
                "falling back to byte copy."
            )
            new_path.write_bytes(legacy_path.read_bytes())
            legacy_path.unlink(missing_ok=True)
        logger.info(
            f"Migrated node-ID keypair from legacy cache path {legacy_path} "
            f"to persistent config path {new_path}."
        )
    except Exception as e:
        logger.warning(
            f"Failed to migrate legacy node-ID keypair from {legacy_path} "
            f"to {new_path}: {e}. The node will generate a new identity; "
            "manually copy the file if cluster membership matters."
        )
