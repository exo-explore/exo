import argparse
import multiprocessing as mp
import os
import resource
import signal
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Self

import anyio
from loguru import logger
from pydantic import PositiveInt

import exo.routing.topics as topics
from exo.api.main import API
from exo.download.coordinator import DownloadCoordinator
from exo.download.impl_shard_downloader import exo_shard_downloader
from exo.master.main import Master
from exo.routing.event_router import EventRouter
from exo.routing.router import Router, get_node_id_keypair
from exo.shared.constants import EXO_LOG
from exo.shared.election import Election, ElectionResult
from exo.shared.logging import logger_cleanup, logger_set_context, logger_setup
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.state import State
from exo.utils.channels import Receiver, channel
from exo.utils.pydantic_ext import FrozenModel
from exo.utils.task_group import TaskGroup
from exo.worker.main import Worker


@dataclass
class Node:
    router: Router
    event_router: EventRouter
    download_coordinator: DownloadCoordinator | None
    worker: Worker | None
    election: Election  # Every node participates in election, as we do want a node to become master even if it isn't a master candidate if no master candidates are present.
    election_result_receiver: Receiver[ElectionResult]
    master: Master | None
    api: API | None

    node_id: NodeId
    offline: bool
    _api_port: int
    _tg: TaskGroup = field(init=False, default_factory=TaskGroup)

    @classmethod
    async def create(cls, args: "Args") -> Self:
        keypair = get_node_id_keypair()
        node_id = NodeId(keypair.to_node_id())
        session_id = SessionId(master_node_id=node_id, election_clock=0)
        router = Router.create(
            keypair,
            bootstrap_peers=args.bootstrap_peers,
            listen_port=args.libp2p_port,
        )
        await router.register_topic(topics.GLOBAL_EVENTS)
        await router.register_topic(topics.LOCAL_EVENTS)
        await router.register_topic(topics.COMMANDS)
        await router.register_topic(topics.ELECTION_MESSAGES)
        await router.register_topic(topics.CONNECTION_MESSAGES)
        await router.register_topic(topics.DOWNLOAD_COMMANDS)
        event_router = EventRouter(
            session_id,
            command_sender=router.sender(topics.COMMANDS),
            external_outbound=router.sender(topics.LOCAL_EVENTS),
            external_inbound=router.receiver(topics.GLOBAL_EVENTS),
        )

        logger.info(f"Starting node {node_id}")

        # Create DownloadCoordinator (unless --no-downloads)
        if not args.no_downloads:
            download_coordinator = DownloadCoordinator(
                node_id,
                exo_shard_downloader(offline=args.offline),
                event_sender=event_router.sender(),
                download_command_receiver=router.receiver(topics.DOWNLOAD_COMMANDS),
                offline=args.offline,
            )
        else:
            download_coordinator = None

        if args.spawn_api:
            api = API(
                node_id,
                port=args.api_port,
                event_receiver=event_router.receiver(),
                command_sender=router.sender(topics.COMMANDS),
                download_command_sender=router.sender(topics.DOWNLOAD_COMMANDS),
                election_receiver=router.receiver(topics.ELECTION_MESSAGES),
            )
        else:
            api = None

        if not args.no_worker:
            worker = Worker(
                node_id,
                event_receiver=event_router.receiver(),
                event_sender=event_router.sender(),
                command_sender=router.sender(topics.COMMANDS),
                download_command_sender=router.sender(topics.DOWNLOAD_COMMANDS),
                api_port=args.api_port,
            )
        else:
            worker = None

        # We start every node with a master
        master = Master(
            node_id,
            session_id,
            event_sender=event_router.sender(),
            global_event_sender=router.sender(topics.GLOBAL_EVENTS),
            local_event_receiver=router.receiver(topics.LOCAL_EVENTS),
            command_receiver=router.receiver(topics.COMMANDS),
            download_command_sender=router.sender(topics.DOWNLOAD_COMMANDS),
        )

        er_send, er_recv = channel[ElectionResult]()
        election = Election(
            node_id,
            # If someone manages to assemble 1 MILLION devices into an exo cluster then. well done. good job champ.
            seniority=1_000_000 if args.force_master else 0,
            # nb: this DOES feedback right now. i have thoughts on how to address this,
            # but ultimately it seems not worth the complexity
            election_message_sender=router.sender(topics.ELECTION_MESSAGES),
            election_message_receiver=router.receiver(topics.ELECTION_MESSAGES),
            connection_message_receiver=router.receiver(topics.CONNECTION_MESSAGES),
            command_receiver=router.receiver(topics.COMMANDS),
            election_result_sender=er_send,
        )

        self = cls(
            router,
            event_router,
            download_coordinator,
            worker,
            election,
            er_recv,
            master,
            api,
            node_id,
            args.offline,
            args.api_port,
        )
        logger_set_context(
            node_id=node_id, role="master" if args.force_master else "node"
        )
        logger.info(
            f"Node components created node_id={node_id} api_port={args.api_port} "
            f"libp2p_port={args.libp2p_port} bootstrap_peers={args.bootstrap_peers}"
        )
        return self

    async def run(self):
        async with self._tg as tg:
            signal.signal(signal.SIGINT, lambda _, __: self.shutdown())
            signal.signal(signal.SIGTERM, lambda _, __: self.shutdown())
            tg.start_soon(self.router.run)
            tg.start_soon(self.event_router.run)
            tg.start_soon(self.election.run)
            if self.download_coordinator:
                tg.start_soon(self.download_coordinator.run)
            if self.worker:
                tg.start_soon(self.worker.run)
            if self.master:
                tg.start_soon(self.master.run)
            if self.api:
                tg.start_soon(self.api.run)
            tg.start_soon(self._elect_loop)
            tg.start_soon(self._diagnostic_snapshot_loop)

    def shutdown(self):
        # if this is our second call to shutdown, just sys.exit
        if self._tg.cancel_called():
            import sys

            sys.exit(1)
        self._tg.cancel_tasks()

    async def _elect_loop(self):
        with self.election_result_receiver as results:
            async for result in results:
                # This function continues to have a lot of very specific entangled logic
                # At least it's somewhat contained

                # I don't like this duplication, but it's manageable for now.
                # TODO: This function needs refactoring generally

                # Ok:
                # On new master:
                # - Elect master locally if necessary
                # - Shutdown and re-create the worker
                # - Shut down and re-create the API

                if result.is_new_master:
                    await anyio.sleep(0)
                    self.event_router.shutdown()
                    self.event_router = EventRouter(
                        result.session_id,
                        self.router.sender(topics.COMMANDS),
                        self.router.receiver(topics.GLOBAL_EVENTS),
                        self.router.sender(topics.LOCAL_EVENTS),
                    )

                if (
                    result.session_id.master_node_id == self.node_id
                    and self.master is not None
                ):
                    logger_set_context(role="master", session_id=str(result.session_id))
                    logger.info("Node elected Master")
                elif (
                    result.session_id.master_node_id == self.node_id
                    and self.master is None
                ):
                    logger_set_context(role="master", session_id=str(result.session_id))
                    logger.info("Node elected Master - promoting self")
                    self.master = Master(
                        self.node_id,
                        result.session_id,
                        event_sender=self.event_router.sender(),
                        global_event_sender=self.router.sender(topics.GLOBAL_EVENTS),
                        local_event_receiver=self.router.receiver(topics.LOCAL_EVENTS),
                        command_receiver=self.router.receiver(topics.COMMANDS),
                        download_command_sender=self.router.sender(
                            topics.DOWNLOAD_COMMANDS
                        ),
                    )
                    self._tg.start_soon(self.master.run)
                elif (
                    result.session_id.master_node_id != self.node_id
                    and self.master is not None
                ):
                    logger_set_context(role="worker", session_id=str(result.session_id))
                    logger.info(
                        f"Node {result.session_id.master_node_id} elected master - demoting self"
                    )
                    await self.master.shutdown()
                    self.master = None
                else:
                    logger_set_context(role="worker", session_id=str(result.session_id))
                    logger.info(
                        f"Node {result.session_id.master_node_id} elected master"
                    )
                if result.is_new_master:
                    if self.download_coordinator:
                        await self.download_coordinator.shutdown()
                        self.download_coordinator = DownloadCoordinator(
                            self.node_id,
                            exo_shard_downloader(offline=self.offline),
                            event_sender=self.event_router.sender(),
                            download_command_receiver=self.router.receiver(
                                topics.DOWNLOAD_COMMANDS
                            ),
                            offline=self.offline,
                        )
                        self._tg.start_soon(self.download_coordinator.run)
                    if self.worker:
                        await self.worker.shutdown()
                        # TODO: add profiling etc to resource monitor
                        self.worker = Worker(
                            self.node_id,
                            event_receiver=self.event_router.receiver(),
                            event_sender=self.event_router.sender(),
                            command_sender=self.router.sender(topics.COMMANDS),
                            download_command_sender=self.router.sender(
                                topics.DOWNLOAD_COMMANDS
                            ),
                            api_port=self._api_port,
                        )
                        self._tg.start_soon(self.worker.run)
                    if self.api:
                        self.api.reset(result.won_clock, self.event_router.receiver())
                    self._tg.start_soon(self.event_router.run)
                else:
                    if self.api:
                        self.api.unpause(result.won_clock)

    async def _diagnostic_snapshot_loop(self) -> None:
        interval_value = os.getenv("EXO_DIAGNOSTIC_SNAPSHOT_SECONDS", "15")
        try:
            interval_seconds = float(interval_value)
        except ValueError:
            logger.warning(
                "Invalid EXO_DIAGNOSTIC_SNAPSHOT_SECONDS value "
                f"{interval_value!r}; using default 15s"
            )
            interval_seconds = 15.0
        if interval_seconds <= 0:
            logger.info("Cluster diagnostic snapshots disabled")
            return
        while True:
            await anyio.sleep(interval_seconds)
            self._log_diagnostic_snapshot()

    def _log_diagnostic_snapshot(self) -> None:
        state_source = "none"
        state: State | None = None
        if self.master is not None:
            state_source = "master"
            state = self.master.state
        elif self.worker is not None:
            state_source = "worker"
            state = self.worker.state

        if state is None:
            logger.info("Cluster diagnostic snapshot state_source=none")
            return

        node_names = self._topology_node_names(state)
        runner_states = [
            f"{runner_id}:{type(runner_status).__name__}"
            for runner_id, runner_status in state.runners.items()
        ]
        instance_models = [
            (
                f"{instance_id}:"
                f"{instance.shard_assignments.model_id}:"
                f"{len(instance.shard_assignments.node_to_runner)}-node"
            )
            for instance_id, instance in state.instances.items()
        ]
        last_seen_ages = self._last_seen_ages(state)
        local_runner_processes = (
            len(self.worker.runners) if self.worker is not None else 0
        )
        outbound_events = len(self.event_router.out_for_delivery)
        logger.info(
            "Cluster diagnostic snapshot "
            f"state_source={state_source} "
            f"last_event_applied_idx={state.last_event_applied_idx} "
            f"topology_nodes={node_names} "
            f"last_seen_ages_seconds={last_seen_ages} "
            f"state_runners={runner_states} "
            f"state_instances={instance_models} "
            f"local_runner_processes={local_runner_processes} "
            f"out_for_delivery={outbound_events}"
        )

    def _topology_node_names(self, state: State) -> list[str]:
        names: list[str] = []
        for node_id in state.topology.list_nodes():
            identity = state.node_identities.get(node_id)
            names.append(
                identity.friendly_name if identity is not None else str(node_id)
            )
        return names

    def _last_seen_ages(self, state: State) -> dict[str, float]:
        now = datetime.now(tz=timezone.utc)
        ages: dict[str, float] = {}
        for node_id, last_seen in state.last_seen.items():
            identity = state.node_identities.get(node_id)
            name = identity.friendly_name if identity is not None else str(node_id)
            ages[name] = round((now - last_seen).total_seconds(), 3)
        return ages


def main():
    args = Args.parse()
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(max(soft, 65535), hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))

    mp.set_start_method("spawn", force=True)
    # TODO: Refactor the current verbosity system
    logger_setup(EXO_LOG, args.verbosity)
    logger_set_context(git_commit=_git_commit())
    logger.info(f"{'=' * 40}")
    logger.info(f"Starting EXO | pid={os.getpid()}")
    logger.info(f"{'=' * 40}")
    logger.info(f"EXO_LIBP2P_NAMESPACE: {os.getenv('EXO_LIBP2P_NAMESPACE')}")

    if args.offline:
        logger.info("Running in OFFLINE mode — no internet checks, local models only")

    if args.bootstrap_peers:
        logger.info(f"Bootstrap peers: {args.bootstrap_peers}")

    if args.no_batch:
        os.environ["EXO_NO_BATCH"] = "1"
        logger.info("Continuous batching disabled (--no-batch)")

    # Set FAST_SYNCH override env var for runner subprocesses
    if args.fast_synch is True:
        os.environ["EXO_FAST_SYNCH"] = "true"
        logger.info("FAST_SYNCH forced ON")
    elif args.fast_synch is False:
        os.environ["EXO_FAST_SYNCH"] = "false"
        logger.info("FAST_SYNCH forced OFF")

    node = anyio.run(Node.create, args)
    try:
        anyio.run(node.run)
    except BaseException as exception:
        logger.opt(exception=exception).critical(
            "EXO terminated due to unhandled exception"
        )
        raise
    finally:
        logger.info("EXO Shutdown complete")
        logger_cleanup()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return "unknown"
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else "unknown"


class Args(FrozenModel):
    verbosity: int = 0
    force_master: bool = False
    spawn_api: bool = False
    api_port: PositiveInt = 52415
    tb_only: bool = False
    no_worker: bool = False
    no_downloads: bool = False
    offline: bool = os.getenv("EXO_OFFLINE", "false").lower() == "true"
    no_batch: bool = False
    fast_synch: bool | None = None  # None = auto, True = force on, False = force off
    bootstrap_peers: list[str] = []
    libp2p_port: int

    @classmethod
    def parse(cls) -> Self:
        parser = argparse.ArgumentParser(prog="EXO")
        default_verbosity = 0
        parser.add_argument(
            "-q",
            "--quiet",
            action="store_const",
            const=-1,
            dest="verbosity",
            default=default_verbosity,
        )
        parser.add_argument(
            "-v",
            "--verbose",
            action="count",
            dest="verbosity",
            default=default_verbosity,
        )
        parser.add_argument(
            "-m",
            "--force-master",
            action="store_true",
            dest="force_master",
        )
        parser.add_argument(
            "--no-api",
            action="store_false",
            dest="spawn_api",
        )
        parser.add_argument(
            "--api-port",
            type=int,
            dest="api_port",
            default=52415,
        )
        parser.add_argument(
            "--no-worker",
            action="store_true",
        )
        parser.add_argument(
            "--no-downloads",
            action="store_true",
            help="Disable the download coordinator (node won't download models)",
        )
        parser.add_argument(
            "--offline",
            action="store_true",
            default=os.getenv("EXO_OFFLINE", "false").lower() == "true",
            help="Run in offline/air-gapped mode: skip internet checks, use only pre-staged local models",
        )
        parser.add_argument(
            "--no-batch",
            action="store_true",
            help="Disable continuous batching, use sequential generation",
        )
        parser.add_argument(
            "--bootstrap-peers",
            type=lambda s: [p for p in s.split(",") if p],
            default=os.getenv("EXO_BOOTSTRAP_PEERS", "").split(",")
            if os.getenv("EXO_BOOTSTRAP_PEERS")
            else [],
            dest="bootstrap_peers",
            help="Comma-separated libp2p multiaddrs to dial on startup (env: EXO_BOOTSTRAP_PEERS)",
        )
        parser.add_argument(
            "--libp2p-port",
            type=int,
            default=0,
            dest="libp2p_port",
            help="Fixed TCP port for libp2p to listen on (0 = OS-assigned).",
        )
        fast_synch_group = parser.add_mutually_exclusive_group()
        fast_synch_group.add_argument(
            "--fast-synch",
            action="store_true",
            dest="fast_synch",
            default=None,
            help="Force MLX FAST_SYNCH on (for JACCL backend)",
        )
        fast_synch_group.add_argument(
            "--no-fast-synch",
            action="store_false",
            dest="fast_synch",
            help="Force MLX FAST_SYNCH off",
        )

        args = parser.parse_args()
        return cls(**vars(args))  # pyright: ignore[reportAny] - We are intentionally validating here, we can't do it statically
