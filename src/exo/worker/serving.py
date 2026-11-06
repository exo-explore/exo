"""The output node's side of serving a request, without the master.

An API node sends a request to the instance's output node: `serve` hands the task to
every rank of the instance (this node's runner, and each other node's through `deliver`), says
Accepted once every rank holds it, and streams the request's output back. The ranks agree among
themselves to run a task once every rank holds it (mx_all_gather_tasks), so Accepted means the
request will run.

A rank that holds a task waits in the ranks' agreement for every other rank to hold it too, and
an idle rank never joins: a task only some ranks hold would stall the instance. So whenever a
request doesn't run to its end, every rank is given the task and its cancellation, until each has
taken them, answers that it has no ready runner (its instance is being started again, which
drops the task on every rank), or is gone. A rank given both drops the task with the others.
"""

import contextlib
from collections import defaultdict
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Final, final

import anyio
import httpx
from anyio import BrokenResourceError, ClosedResourceError, EndOfStream
from loguru import logger

from exo.routing.membership import Reachability
from exo.shared import serving
from exo.shared.model_readiness import (
    NotReady,
    instance_is_ready,
    instance_readiness,
)
from exo.shared.models.model_cards import ModelId
from exo.shared.request_errors import (
    DEVICE_DISCONNECTED,
    DEVICE_UNREACHABLE,
    INSTANCE_GONE_BEFORE_START,
    NOT_TAKEN,
)
from exo.shared.serving import STREAM_HEARTBEAT_INTERVAL, WEDGE_SECONDS
from exo.shared.types.chunks import Chunk, ErrorChunk
from exo.shared.types.common import NodeId
from exo.shared.types.serving import (
    Accepted,
    Delivery,
    Refused,
    ServeFrame,
    ServeRequest,
)
from exo.shared.types.state import State
from exo.shared.types.tasks import GenerationTask, TaskId
from exo.shared.types.worker.instances import Instance, InstanceId
from exo.shared.types.worker.runners import RunnerId, RunnerReady, RunnerRunning
from exo.utils.channels import Sender, channel
from exo.utils.task_group import TaskGroup
from exo.worker.runner.supervisor import RunnerSupervisor

SERVE_PATH: Final = "/_exo/serve"
DELIVER_PATH: Final = "/_exo/deliver"
# How long to wait for a TCP connection to a node's address: httpx's default, as the
# reachability probe uses. An address that doesn't answer is skipped for the next one.
CONNECT_TIMEOUT: Final = 5.0
# How long an idle connection to a node is kept for the next request: under the 5 s after which
# the node's server (hypercorn's keep_alive_timeout) closes it, so a request is never sent on a
# connection the server is closing
KEEPALIVE_EXPIRY: Final = 2.0
# The connections to other nodes: as many as there are requests and deliveries in flight. A cap
# would make the request after it wait, with no answer, for one of the others to end.
CONNECTION_LIMITS: Final = httpx.Limits(
    max_connections=None, keepalive_expiry=KEEPALIVE_EXPIRY
)
# A rank's answer to a delivery it can't take because it has no ready runner for the instance
# (status DECLINED_STATUS, with this header). The only answer that ends the attempts to give a
# rank a task's cancellation: any other one is tried again.
DECLINED_STATUS: Final = 409
DECLINED_HEADER: Final = "x-exo-no-ready-runner"
DECLINED_HEADER_VALUE: Final = "1"
# How long a rank may go on not taking a request's cancellation (refusing it for any other
# reason than having no ready runner, or not answering) before this node fails its own runner.
# The ranks that took the request hold it until every rank has its cancellation, so the
# instance can't serve; failing this runner starts every runner of the instance again (as the
# stall watchdog does after 90 s for a stuck text runner, while image instances have none),
# which drops the request on every rank. The wedge bound: a rank that doesn't take a delivery
# within it hasn't, and this one has had as long to take the cancellation.
CANCEL_REFUSED_LIMIT: Final = WEDGE_SECONDS


@final
@dataclass(frozen=True)
class Declined:
    """The rank answered that it has no ready runner for the instance: its runner is gone or is
    being started again, and with it every runner of the instance"""

    reason: str


@final
@dataclass(frozen=True)
class Undelivered:
    """The rank didn't take it for any other reason: no answer, no address that connects, or
    any other answer (refused by its member check, a node of another version, an error)"""

    reason: str


# None: the node took the delivery
DeliveryOutcome = Declined | Undelivered | None


class RefusalError(Exception):
    """A request that wasn't taken, so nothing of it runs, or a delivery a rank can't take as
    it has no ready runner for the instance; the message is one sentence for the user"""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


@final
@dataclass(eq=False)
class _Delivery:
    """How each rank of the instance took a request's task"""

    nodes: frozenset[NodeId]
    outcomes: dict[NodeId, DeliveryOutcome] = field(default_factory=dict)
    done: anyio.Event = field(default_factory=anyio.Event)
    # Cancelled when the request is ended while it is still being delivered
    scope: anyio.CancelScope = field(default_factory=anyio.CancelScope)

    @property
    def everywhere(self) -> bool:
        return self.outcomes.keys() == self.nodes and all(
            outcome is None for outcome in self.outcomes.values()
        )


@final
@dataclass(eq=False)
class _Open:
    """A request streaming from this node"""

    task: GenerationTask
    output: Sender[Chunk]
    delivery: _Delivery
    # Why it was ended from outside (its instance deleted, a node of it unreachable)
    why: str | None = None

    def end(self, error_message: str) -> None:
        """End the stream, telling its client why. A request still being delivered is refused
        at once."""
        self.why = error_message
        self.delivery.scope.cancel()
        with contextlib.suppress(ClosedResourceError, BrokenResourceError):
            self.output.send_nowait(
                ErrorChunk(
                    model=ModelId(self.task.task_params.model),
                    error_message=error_message,
                )
            )
        self.output.close()


@final
class ServingHost:
    def __init__(
        self,
        node_id: NodeId,
        *,
        state: Callable[[], State],
        runners: Callable[[], Mapping[RunnerId, RunnerSupervisor]],
        reachability: Reachability,
        api_port: int,
        # How a delivery reaches another node. Over HTTP, unless a test wires nodes together
        # in-process.
        deliver_to: Callable[[NodeId, Delivery], Awaitable[DeliveryOutcome]]
        | None = None,
    ):
        self.node_id = node_id
        self._deliver_to = deliver_to or self._deliver_over_http
        self._state = state
        self._runners = runners
        self._reachability = reachability
        self._api_port = api_port
        self._tg = TaskGroup()
        self._client: httpx.AsyncClient | None = None
        # Image ranks don't agree on the order of their tasks: they run them as they arrive.
        # So an image request is handed to every rank (and, if not every rank took it, its
        # cancellation too) before the next one is. Text ranks agree on the order themselves.
        self._image_locks: defaultdict[InstanceId, anyio.Lock] = defaultdict(anyio.Lock)
        self._open: dict[TaskId, _Open] = {}
        # Instances with a request whose cancellation a rank hasn't taken yet: they can't serve
        # until it does, and each request delivered meanwhile would only add another
        self._unsettled: defaultdict[InstanceId, int] = defaultdict(int)

    async def run(self) -> None:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(WEDGE_SECONDS, connect=CONNECT_TIMEOUT),
            limits=CONNECTION_LIMITS,
            trust_env=False,
        ) as client:
            self._client = client
            async with self._tg as tg:
                tg.start_soon(self._end_requests_of_unreachable_nodes)
                await anyio.sleep_forever()

    def open_requests(self) -> list[GenerationTask]:
        return [entry.task for entry in self._open.values()]

    def runner_tasks(self) -> dict[str, list[str]]:
        """The requests each of this node's runners holds"""
        return {
            str(runner_id): [str(task_id) for task_id in runner.in_progress]
            for runner_id, runner in self._runners().items()
        }

    # --- the output node -------------------------------------------------------------------

    async def serve(
        self, request: ServeRequest
    ) -> AsyncGenerator[ServeFrame | None, None]:
        """A request's stream: heartbeats (None) while every rank is given the task, then
        Accepted or Refused, then the output. Close it to cancel the request."""
        task = request.task
        try:
            instance, runner, peers = self._check(request)
        except RefusalError as refusal:
            logger.info(f"Refusing request {task.command_id}: {refusal.message}")
            yield Refused(message=refusal.message)
            return
        send, receive = channel[Chunk]()
        delivery = _Delivery(nodes=frozenset([self.node_id, *peers]))
        entry = _Open(task=task, output=send, delivery=delivery)
        self._open[task.task_id] = entry
        self._tg.start_soon(
            self._deliver_everywhere, instance, runner, peers, task, send, delivery
        )
        try:
            while not delivery.done.is_set():
                with anyio.move_on_after(STREAM_HEARTBEAT_INTERVAL):
                    await delivery.done.wait()
                if not delivery.done.is_set():
                    yield None
            if not delivery.everywhere:
                yield Refused(message=entry.why or _why_not_taken(delivery))
                return
            yield Accepted()
            while True:
                chunk: Chunk | None = None
                with anyio.move_on_after(STREAM_HEARTBEAT_INTERVAL):
                    try:
                        chunk = await receive.receive()
                    except EndOfStream:
                        return
                yield chunk
        finally:
            self._open.pop(task.task_id, None)
            receive.close()
            # A request still being delivered stops there: the ranks it reached are then given
            # its cancellation (_deliver_everywhere)
            delivery.scope.cancel()
            # Not while this node shuts down: its runners stop, and with them the request
            with contextlib.suppress(RuntimeError):
                self._tg.start_soon(self._finish, instance, runner, task, delivery)

    def _check(
        self, request: ServeRequest
    ) -> tuple[Instance, RunnerSupervisor, list[NodeId]]:
        task = request.task
        state = self._state()
        if request.target != self.node_id:
            raise RefusalError(DEVICE_DISCONNECTED)
        instance = state.instances.get(task.instance_id)
        if instance is None:
            raise RefusalError(INSTANCE_GONE_BEFORE_START)
        if (
            serving.output_node(instance, images=serving.is_image_task(task))
            != self.node_id
        ):
            raise RefusalError(DEVICE_DISCONNECTED)
        nodes = list(instance.shard_assignments.node_to_runner)
        if not all(self._reachability.is_reachable(node) for node in nodes):
            raise RefusalError(DEVICE_UNREACHABLE)
        if self._unsettled[instance.instance_id]:
            raise RefusalError(NOT_TAKEN)
        runner = self._local_runner(instance)
        if not instance_is_ready(instance, state.runners) or not (
            runner is not None and _ready(runner)
        ):
            not_ready = instance_readiness(state, instance)
            raise RefusalError(
                not_ready.message
                if not_ready is not None
                else INSTANCE_GONE_BEFORE_START
            )
        assert runner is not None
        # A task is served once: one being served, or that this node's runner was already
        # given, is another copy of a request whose own stream has it
        if task.task_id in self._open or runner.has_task(task.task_id):
            raise RefusalError(NOT_TAKEN)
        return instance, runner, [node for node in nodes if node != self.node_id]

    async def _deliver_everywhere(
        self,
        instance: Instance,
        runner: RunnerSupervisor,
        peers: list[NodeId],
        task: GenerationTask,
        output: Sender[Chunk],
        delivery: _Delivery,
    ) -> None:
        """Give every rank the task; if not every rank took it, give every rank it reached its
        cancellation too"""

        async def here() -> None:
            # Not cancelled half way: the runner would hold a task it was never sent
            with anyio.CancelScope(shield=True):
                await runner.start_task(task, output)
            delivery.outcomes[self.node_id] = (
                None
                if runner.has_task(task.task_id)
                else Declined("its runner stopped")
            )

        async def there(peer: NodeId) -> None:
            delivery.outcomes[peer] = Undelivered(
                f"nothing within {WEDGE_SECONDS:.0f}s"
            )
            with anyio.move_on_after(WEDGE_SECONDS):
                delivery.outcomes[peer] = await self._deliver_to(
                    peer, Delivery(task=task)
                )

        # Image ranks take requests in the order they arrive: one image request at a time is
        # handed to every rank, with its cancellation if not every rank took it. Waiting for
        # that turn is part of the delivery: a request ended meanwhile reaches no rank, and one
        # whose turn doesn't come within WEDGE_SECONDS (a rank that still hasn't taken the one before
        # it) is refused.
        order = (
            self._image_locks[instance.instance_id]
            if serving.is_image_task(task)
            else None
        )
        turn = False
        try:
            with delivery.scope:
                if order is not None:
                    with anyio.move_on_after(WEDGE_SECONDS):
                        await order.acquire()
                        turn = True
                if order is None or turn:
                    async with anyio.create_task_group() as tg:
                        tg.start_soon(here)
                        for peer in peers:
                            tg.start_soon(there, peer)
        except Exception as error:
            # The request is refused (not every rank took it); this node's other requests
            # carry on
            logger.opt(exception=error).error(f"Delivering {task.command_id} failed")
        finally:
            delivery.done.set()
        try:
            if delivery.everywhere:
                return
            logger.warning(
                f"Request {task.command_id} wasn't taken by every rank: "
                f"{delivery.outcomes}"
            )
            # Every rank it may have reached (a rank whose answer didn't come may still take
            # it) holds it until every rank holds it and its cancellation; the instance takes
            # no other request until then
            if delivery.outcomes:
                self._unsettled[instance.instance_id] += 1
                try:
                    await self._cancel_everywhere(instance, runner, task)
                finally:
                    self._unsettled[instance.instance_id] -= 1
        except Exception as error:
            logger.opt(exception=error).error(f"Cancelling {task.command_id} failed")
        finally:
            if order is not None and turn:
                order.release()

    async def _finish(
        self,
        instance: Instance,
        runner: RunnerSupervisor,
        task: GenerationTask,
        delivery: _Delivery,
    ) -> None:
        """After a request's stream ends: if its runners still run it, cancel it on every rank.
        Not needed if it ran to its end (every rank finished it together), if its runner or
        instance is gone (the instance's runners stop), or if not every rank took it (its
        delivery cancels it)."""
        await delivery.done.wait()
        if (
            not delivery.everywhere
            or task.task_id not in runner.in_progress
            or instance.instance_id not in self._state().instances
            or self._local_runner(instance) is not runner
        ):
            return
        logger.info(f"Cancelling request {task.command_id} on every rank")
        try:
            await self._cancel_everywhere(instance, runner, task)
        except Exception as error:
            # This node's other requests carry on; the instance's runners are torn down if the
            # request is stuck on them
            logger.opt(exception=error).error(f"Cancelling {task.command_id} failed")

    async def _cancel_everywhere(
        self, instance: Instance, runner: RunnerSupervisor, task: GenerationTask
    ) -> None:
        """Give every rank the task and its cancellation"""
        cancelled = Delivery(task=task, cancelled=True)
        async with anyio.create_task_group() as tg:
            for node in instance.shard_assignments.node_to_runner:
                if node != self.node_id:
                    tg.start_soon(self._cancel_on, node, instance, runner, cancelled)
            # This node's runner can't take it only if it has stopped, and its instance's
            # runners with it
            with contextlib.suppress(RefusalError):
                await self.deliver(cancelled)

    async def _cancel_on(
        self,
        peer: NodeId,
        instance: Instance,
        runner: RunnerSupervisor,
        cancelled: Delivery,
    ) -> None:
        """Give a rank the task and its cancellation until it takes them or answers that it
        has no ready runner (its runners are being started again, which drops every task),
        while the instance is there and this node's runner is the one that took the task. A
        rank that hasn't taken it within CANCEL_REFUSED_LIMIT (each attempt is cut short to
        fit) fails this node's runner. Meanwhile the instance refuses new requests."""
        command_id = cancelled.task.command_id
        deadline = anyio.current_time() + CANCEL_REFUSED_LIMIT
        unsettled = False
        try:
            while True:
                left = max(0.0, deadline - anyio.current_time())
                outcome: DeliveryOutcome = Undelivered("no answer in time")
                with anyio.move_on_after(min(WEDGE_SECONDS, left)):
                    outcome = await self._deliver_to(peer, cancelled)
                if not isinstance(outcome, Undelivered):
                    return
                if (
                    instance.instance_id not in self._state().instances
                    or self._local_runner(instance) is not runner
                    or not _ready(runner)
                ):
                    logger.info(
                        f"Request {command_id}: no more need to cancel it on {peer}"
                    )
                    return
                if not unsettled:
                    unsettled = True
                    self._unsettled[instance.instance_id] += 1
                    logger.warning(
                        f"Request {command_id}: {peer} didn't take its cancellation "
                        f"({outcome.reason}); trying again every "
                        f"{STREAM_HEARTBEAT_INTERVAL:.0f}s"
                    )
                if anyio.current_time() >= deadline:
                    reason = (
                        f"{peer} didn't take the cancellation of request {command_id} "
                        f"for {CANCEL_REFUSED_LIMIT:.0f}s ({outcome.reason})"
                    )
                    logger.error(f"Failing this device's runner: {reason}")
                    await runner.fail(reason)
                    return
                await anyio.sleep(
                    min(STREAM_HEARTBEAT_INTERVAL, deadline - anyio.current_time())
                )
        finally:
            if unsettled:
                self._unsettled[instance.instance_id] -= 1

    async def _deliver_over_http(
        self, peer: NodeId, delivery: Delivery
    ) -> DeliveryOutcome:
        assert self._client is not None
        body = delivery.model_dump_json()
        urls = serving.node_urls(self._state(), self.node_id, peer, self._api_port)
        outcome: DeliveryOutcome = Undelivered("no address answered")
        for url in urls:
            try:
                response = await self._post_delivery(f"{url}{DELIVER_PATH}", body)
            except (httpx.ConnectError, httpx.ConnectTimeout):
                continue
            except httpx.HTTPError as error:
                return Undelivered(type(error).__name__)
            if response.status_code == 200:
                return None
            if DECLINED_HEADER in response.headers:
                return Declined(response.text[:200])
            # Not the rank's answer (refused by its member check, another version of exo, an
            # error): another of its addresses may do
            outcome = Undelivered(
                f"{url} answered {response.status_code}: {response.text[:200]}"
            )
        return outcome

    async def _post_delivery(self, url: str, body: str) -> httpx.Response:
        """POST a delivery; once more on a fresh connection if the node closed the one it went
        on without answering (a server that was frozen past its keep-alive timeout closes idle
        connections as it wakes). Taking a delivery twice is harmless."""
        assert self._client is not None
        headers = {"content-type": "application/json"}
        try:
            return await self._client.post(url, content=body, headers=headers)
        except httpx.RemoteProtocolError:
            return await self._client.post(url, content=body, headers=headers)

    def end_instance(self, instance_id: InstanceId, error_message: str) -> None:
        """End the requests streaming from an instance that was deleted"""
        for entry in list(self._open.values()):
            if entry.task.instance_id == instance_id:
                entry.end(error_message)

    async def _end_requests_of_unreachable_nodes(self) -> None:
        """End the requests of an instance one of whose nodes can't be reached now: their
        output can't be made (the device may be back in a moment, or be removed once it is
        gone). The API node isn't consulted: a request is cancelled when its stream closes,
        however that comes about, and nothing else cancels it."""
        with self._reachability.watch() as unreachable:
            async for node_id in unreachable:
                for entry in list(self._open.values()):
                    instance = self._state().instances.get(entry.task.instance_id)
                    if (
                        instance is not None
                        and node_id in instance.shard_assignments.node_to_runner
                    ):
                        entry.end(DEVICE_UNREACHABLE)

    # --- every rank ------------------------------------------------------------------------

    async def deliver(self, delivery: Delivery) -> None:
        """Take a request's task for this node's runner (and its cancellation). Taking the
        same delivery twice is harmless. Raises RefusalError if this node has no ready runner
        for the request's instance."""
        task = delivery.task
        instance = self._state().instances.get(task.instance_id)
        runner = self._local_runner(instance) if instance is not None else None
        if runner is None:
            raise RefusalError("this device has no runner for the request's instance")
        if not runner.has_task(task.task_id):
            if not _ready(runner):
                raise RefusalError("this device's runner for the model isn't ready")
            await runner.start_task(task)
            if not runner.has_task(task.task_id):
                raise RefusalError("this device's runner for the model has stopped")
        if delivery.cancelled:
            await runner.cancel_task(task.task_id)

    def _local_runner(self, instance: Instance) -> RunnerSupervisor | None:
        runner_id = instance.shard_assignments.node_to_runner.get(self.node_id)
        return self._runners().get(runner_id) if runner_id is not None else None


# A rank that declined has no ready runner for the instance: it is being started again
_RESTARTING: Final = NotReady(
    status="it is being started again", retry_after_seconds=5
).message


def _why_not_taken(delivery: _Delivery) -> str:
    """A request every rank that answered declined is waiting for the instance to be ready
    again (the API node waits for that); one a rank didn't answer is not taken"""
    failures = [
        outcome for outcome in delivery.outcomes.values() if outcome is not None
    ]
    if failures and all(isinstance(outcome, Declined) for outcome in failures):
        return _RESTARTING
    return NOT_TAKEN


def _ready(runner: RunnerSupervisor) -> bool:
    """A generation task given to a runner in any other state stops it (runner.py)"""
    return isinstance(runner.status, (RunnerReady, RunnerRunning))
