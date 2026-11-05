# pyright: reportPrivateUsage=false
"""An in-process cluster for testing serving without the master.

Real ServingHosts, ServingClients and RunnerSupervisors; the runner processes are silent stand-ins,
and the ranks' agreement is simulated the way mx_all_gather_tasks behaves: a task runs once every
rank holds it, and a cancellation takes effect once every rank holds the task and its
cancellation. Nodes are wired together in-process; a test can freeze a node (it stops answering
and stops streaming, as a sleeping Mac) or kill it (its connections break).
"""

import contextlib
import gc
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import cast, final

import anyio
import httpx
from anyio.abc import TaskGroup
from anyio.abc import TaskStatus as Started
from loguru import logger

from exo.api.serving_client import ServingClient, UnroutableError
from exo.api.types import ImageGenerationTaskParams
from exo.routing.membership import Reachability
from exo.shared.models.model_cards import ModelId
from exo.shared.topology import Topology
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerEvent,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.profiling import NetworkInterfaceInfo, NodeNetworkInfo
from exo.shared.types.serving import Delivery, ServeFrame, ServeRequest
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    GenerationTask,
    ImageGeneration,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import BoundInstance, Instance, InstanceId
from exo.shared.types.worker.runners import RunnerId, RunnerReady
from exo.utils.async_process import AsyncProcess
from exo.utils.channels import (
    MpReceiver,
    MpSender,
    Receiver,
    Sender,
    channel,
    mp_channel,
)
from exo.worker.runner.bootstrap import RunnerTerminationError
from exo.worker.runner.supervisor import RunnerStdioHandler, RunnerSupervisor
from exo.worker.serving import (
    Declined,
    DeliveryOutcome,
    RefusalError,
    ServingHost,
    Undelivered,
)
from exo.worker.tests.unittests.conftest import (
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)

MODEL = ModelId("mlx-community/test-model")


class _SilentProcess:
    """A runner process that is alive and says nothing of its own, until it is stopped"""

    def __init__(self) -> None:
        _, self.stdout = channel[bytes]()
        _, self.stderr = channel[bytes]()
        self.exitcode: int | None = None

    def is_alive(self) -> bool:
        return self.exitcode is None

    async def run(self, *, task_status: Started[None]) -> None:
        task_status.started()
        await anyio.sleep_forever()

    async def stop(self) -> None:
        # As a process stopped with SIGTERM
        if self.exitcode is None:
            self.exitcode = -15


class FakeReachability(Reachability):
    """Which nodes the cluster's nodes reach; `lose` makes one unreachable for all of them"""

    def __init__(self, nodes: set[NodeId]) -> None:
        self.reachable = set(nodes)
        self._watchers: list[Sender[NodeId]] = []

    def is_reachable(self, node: NodeId) -> bool:
        return node in self.reachable

    def watch(self) -> Receiver[NodeId]:
        send, receive = channel[NodeId]()
        self._watchers.append(send)
        return receive

    def lose(self, node_id: NodeId) -> None:
        self.reachable.discard(node_id)
        for watcher in self._watchers:
            with contextlib.suppress(
                anyio.ClosedResourceError, anyio.BrokenResourceError
            ):
                watcher.send_nowait(node_id)


@final
@dataclass(eq=False)
class Rank:
    rank: int
    node_id: NodeId
    supervisor: RunnerSupervisor
    events: MpSender[RunnerEvent | RunnerTerminationError]
    tasks: MpReceiver[Task]
    cancels: MpReceiver[TaskId]
    received: list[TaskId] = field(default_factory=list)
    cancels_received: list[TaskId] = field(default_factory=list)
    # Tasks and cancellations, in the order the runner read them
    arrivals: list[tuple[str, TaskId]] = field(default_factory=list)


@final
class Agreement:
    """The ranks' agreement, as mx_all_gather_tasks does it"""

    def __init__(self, ranks: list[Rank], tokens: int, token_interval: float) -> None:
        self.ranks = ranks
        self.tokens = tokens
        self.token_interval = token_interval
        self.held: defaultdict[TaskId, set[int]] = defaultdict(set)
        self.cancelled: defaultdict[TaskId, set[int]] = defaultdict(set)
        self.tasks: dict[TaskId, GenerationTask] = {}
        self.started: set[TaskId] = set()
        self.ended: set[TaskId] = set()
        self.tg: TaskGroup | None = None

    def everyone(self, ranks: set[int]) -> bool:
        return len(ranks) == len(self.ranks)

    def hold(self, rank: int, task: GenerationTask) -> None:
        self.held[task.task_id].add(rank)
        self.tasks[task.task_id] = task
        self._advance(task.task_id)

    def cancel(self, rank: int, task_id: TaskId) -> None:
        if rank in self.held[task_id]:
            self.cancelled[task_id].add(rank)
        self._advance(task_id)

    def _advance(self, task_id: TaskId) -> None:
        if task_id in self.ended or not self.everyone(self.held[task_id]):
            return
        if self.everyone(self.cancelled[task_id]):
            self._end(task_id, TaskStatus.Cancelled)
        elif task_id not in self.started:
            self.started.add(task_id)
            assert self.tg is not None
            self.tg.start_soon(self._generate, self.tasks[task_id])

    def _end(self, task_id: TaskId, status: TaskStatus) -> None:
        if task_id in self.ended:
            return
        self.ended.add(task_id)
        for rank in self.ranks:
            # A runner its supervisor stopped hears nothing more
            with contextlib.suppress(anyio.ClosedResourceError):
                rank.events.send(TaskStatusUpdated(task_id=task_id, task_status=status))

    async def _generate(self, task: GenerationTask) -> None:
        # The rank that emits the output: the last pipeline stage for images
        output = self.ranks[-1 if isinstance(task, ImageGeneration) else 0]
        for index in range(self.tokens):
            await anyio.sleep(self.token_interval)
            if task.task_id in self.ended:
                return
            try:
                output.events.send(
                    ChunkGenerated(
                        command_id=task.command_id,
                        chunk=TokenChunk(
                            model=MODEL,
                            text=f"t{index} ",
                            token_id=index,
                            usage=None,
                            finish_reason="stop" if index == self.tokens - 1 else None,
                        ),
                    )
                )
            except anyio.ClosedResourceError:
                return  # its supervisor stopped it
        self._end(task.task_id, TaskStatus.Complete)


@final
class ServingCluster:
    """N ranks of one text instance on N nodes, and API nodes to send requests through"""

    def __init__(
        self,
        world_size: int,
        *,
        api_nodes: tuple[NodeId, ...],
        tokens: int,
        token_interval: float,
    ) -> None:
        self.rank_nodes = [NodeId(f"rank{index}") for index in range(world_size)]
        self.api_nodes = api_nodes
        nodes = set(self.rank_nodes) | set(api_nodes)
        runners = [RunnerId(f"runner{index}") for index in range(world_size)]
        self.instance: Instance = get_mlx_ring_instance(
            instance_id=InstanceId("instance"),
            model_id=MODEL,
            node_to_runner=dict(zip(self.rank_nodes, runners, strict=True)),
            runner_to_shard={
                runner: get_pipeline_shard_metadata(MODEL, index, world_size)
                for index, runner in enumerate(runners)
            },
        )
        topology = Topology()
        for node in sorted(nodes):
            topology.add_node(node)
        self.state = State(
            instances={self.instance.instance_id: self.instance},
            runners={runner: RunnerReady() for runner in runners},
            topology=topology,
            node_network={
                node: NodeNetworkInfo(
                    interfaces=[
                        NetworkInterfaceInfo(
                            name="en0", ip_address=f"10.0.0.{index + 1}"
                        )
                    ]
                )
                for index, node in enumerate(sorted(nodes))
            },
        )
        self.reachability = FakeReachability(nodes)
        self.awake: dict[NodeId, anyio.Event] = {}
        for node in nodes:
            self.awake[node] = anyio.Event()
            self.awake[node].set()
        self.killed: set[NodeId] = set()
        # Nodes whose answer to a delivery is neither "taken" nor "no ready runner" (a member
        # check that doesn't know the sender, say): the delivery doesn't reach them. See refuse.
        self.refusing: dict[NodeId, str] = {}
        self._refusals_left: dict[NodeId, int] = {}
        self.ranks: list[Rank] = []
        self.hosts: dict[NodeId, ServingHost] = {}
        self.clients: dict[NodeId, ServingClient] = {}
        self.tokens = tokens
        self.token_interval = token_interval
        self.agreement: Agreement | None = None
        self._tg: TaskGroup | None = None

    # --- faults ----------------------------------------------------------------------------

    def freeze(self, node: NodeId) -> None:
        self.awake[node] = anyio.Event()

    def wake(self, node: NodeId) -> None:
        self.awake[node].set()

    def refuse(self, node: NodeId, answer: str, times: int | None = None) -> None:
        """Answer the next `times` deliveries to the node (every one, if None) this way"""
        self.refusing[node] = answer
        if times is not None:
            self._refusals_left[node] = times

    def kill(self, node: NodeId) -> None:
        self.killed.add(node)
        self.awake[node].set()

    # --- wiring ----------------------------------------------------------------------------

    async def _deliver(self, peer: NodeId, delivery: Delivery) -> DeliveryOutcome:
        """A delivery over a connection that holds it while the peer is frozen: the peer takes it
        when it wakes, whether or not the sender still waits (as bytes in a socket buffer)."""
        if peer in self.killed:
            return Undelivered("killed")
        if (answer := self.refusing.get(peer)) is not None:
            if peer in self._refusals_left:
                self._refusals_left[peer] -= 1
                if self._refusals_left[peer] == 0:
                    del self._refusals_left[peer]
                    del self.refusing[peer]
            return Undelivered(answer)
        done = anyio.Event()
        outcome: list[DeliveryOutcome] = []

        async def arrive() -> None:
            await self.awake[peer].wait()
            if peer in self.killed:
                outcome.append(Undelivered("killed"))
            else:
                try:
                    await self.hosts[peer].deliver(delivery)
                    outcome.append(None)
                except RefusalError as refusal:
                    outcome.append(Declined(refusal.message))
            done.set()

        assert self._tg is not None
        self._tg.start_soon(arrive)
        await done.wait()
        return outcome[0]

    def _frames_from(
        self, request: ServeRequest
    ) -> contextlib.AbstractAsyncContextManager[AsyncIterator[ServeFrame | None]]:
        @contextlib.asynccontextmanager
        async def opened() -> AsyncIterator[AsyncIterator[ServeFrame | None]]:
            if request.target in self.killed:
                raise UnroutableError("killed")
            frames = self.hosts[request.target].serve(request)

            async def through() -> AsyncIterator[ServeFrame | None]:
                async for frame in frames:
                    await self.awake[request.target].wait()
                    if request.target in self.killed:
                        raise httpx.ReadError("connection reset")
                    yield frame

            async with contextlib.aclosing(frames):
                yield through()

        return opened()

    async def _run_rank(self, rank: Rank, agreement: Agreement) -> None:
        async def tasks() -> None:
            while True:
                task = await rank.tasks.receive_async()
                rank.received.append(task.task_id)
                rank.arrivals.append(("task", task.task_id))
                rank.events.send(TaskAcknowledged(task_id=task.task_id))
                if isinstance(task, GenerationTask):
                    agreement.hold(rank.rank, task)

        async def cancels() -> None:
            while True:
                task_id = await rank.cancels.receive_async()
                rank.cancels_received.append(task_id)
                rank.arrivals.append(("cancel", task_id))
                agreement.cancel(rank.rank, task_id)

        # A runner whose supervisor stopped it reads nothing more
        with contextlib.suppress(anyio.ClosedResourceError, anyio.EndOfStream):
            async with anyio.create_task_group() as tg:
                tg.start_soon(tasks)
                tg.start_soon(cancels)

    async def run(self, *, task_status: Started[None]) -> None:
        async with anyio.create_task_group() as tg:
            self._tg = tg
            state = lambda: self.state  # noqa: E731
            for index, node in enumerate(self.rank_nodes):
                runner_id = self.instance.shard_assignments.node_to_runner[node]
                bound = BoundInstance(
                    instance=self.instance,
                    bound_runner_id=runner_id,
                    bound_node_id=node,
                )
                event_sender, _ = channel[Event]()
                task_sender, task_receiver = mp_channel[Task]()
                cancel_sender, cancel_receiver = mp_channel[TaskId]()
                events, event_receiver = mp_channel[
                    RunnerEvent | RunnerTerminationError
                ]()
                process = cast(AsyncProcess, cast(object, _SilentProcess()))
                supervisor = RunnerSupervisor(
                    shard_metadata=bound.bound_shard,
                    bound_instance=bound,
                    runner_process=process,
                    _runner_stdio_handler=await RunnerStdioHandler.create(
                        stdout_rx=process.stdout, stderr_rx=process.stderr
                    ),
                    _ev_recv=event_receiver,
                    _task_sender=task_sender,
                    _event_sender=event_sender,
                    _cancel_sender=cancel_sender,
                )
                supervisor.status = RunnerReady()
                self.ranks.append(
                    Rank(
                        index, node, supervisor, events, task_receiver, cancel_receiver
                    )
                )
                self.hosts[node] = ServingHost(
                    node,
                    state=state,
                    runners=lambda supervisor=supervisor, runner_id=runner_id: {
                        runner_id: supervisor
                    },
                    reachability=self.reachability,
                    api_port=52415,
                    deliver_to=self._deliver,
                )
                tg.start_soon(supervisor.run)
                tg.start_soon(self.hosts[node].run)
            agreement = Agreement(self.ranks, self.tokens, self.token_interval)
            agreement.tg = tg
            self.agreement = agreement
            for rank in self.ranks:
                tg.start_soon(self._run_rank, rank, agreement)
            for node in self.api_nodes:
                self.clients[node] = ServingClient(
                    node,
                    api_port=52415,
                    host=self.hosts.get(node),
                    state=state,
                    reachability=self.reachability,
                    frames_from=self._frames_from,
                )
                tg.start_soon(self.clients[node].run)
            await anyio.sleep(0.05)
            task_status.started()
            await anyio.sleep_forever()

    # --- requests --------------------------------------------------------------------------

    def task(self, prompt: str = "hello") -> TextGeneration:
        return TextGeneration(
            command_id=CommandId(),
            instance_id=self.instance.instance_id,
            task_params=TextGenerationTaskParams(
                model=MODEL,
                input=[InputMessage(role="user", content=InputMessageContent(prompt))],
            ),
        )

    def image_task(self, prompt: str = "a cat") -> ImageGeneration:
        """An image request: its output node is the last rank (the last pipeline stage)"""
        return ImageGeneration(
            command_id=CommandId(),
            instance_id=self.instance.instance_id,
            task_params=ImageGenerationTaskParams(prompt=prompt, model=MODEL),
        )

    def in_progress(self) -> dict[NodeId, list[TaskId]]:
        return {rank.node_id: list(rank.supervisor.in_progress) for rank in self.ranks}

    async def settled(
        self, check: Callable[[], bool], timeout: float = 5.0, what: str = ""
    ) -> None:
        with anyio.fail_after(timeout):
            while not check():
                await anyio.sleep(0.01)
        logger.debug(f"settled: {what}")


@contextlib.asynccontextmanager
async def serving_cluster(
    world_size: int = 3,
    *,
    api_nodes: tuple[NodeId, ...] = (NodeId("api"),),
    tokens: int = 20,
    token_interval: float = 0.01,
) -> AsyncIterator[ServingCluster]:
    cluster = ServingCluster(
        world_size, api_nodes=api_nodes, tokens=tokens, token_interval=token_interval
    )
    async with anyio.create_task_group() as tg:
        await tg.start(cluster.run)
        try:
            yield cluster
        finally:
            tg.cancel_scope.cancel()
    # Each runner channel holds POSIX semaphores until it is collected; the system allows only so
    # many (10,000 on macOS, shared by every process), so free them now rather than at some later
    # collection
    del cluster
    gc.collect()
