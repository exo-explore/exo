import multiprocessing as mp
import os
import resource
import signal
import sys
from dataclasses import dataclass, field
from typing import Self

import anyio
from anyio.lowlevel import checkpoint as anyio_checkpoint
from daemon import DaemonContext  # pyright: ignore[reportMissingTypeStubs]
from exo_rs import (
    AppSettings,
    BootstrapSettings,
    CliArgs,
    Pidfile,
    PidfileError,
)
from loguru import logger

import exo.routing.topics as topics
import exo.shared.config as config
from exo.api.main import API
from exo.download.coordinator import DownloadCoordinator
from exo.download.impl_shard_downloader import exo_shard_downloader
from exo.master.main import Master
from exo.routing.event_router import EventRouter
from exo.routing.router import Router, get_node_zid
from exo.shared.election import Election, ElectionResult
from exo.shared.logging import logger_cleanup, logger_setup
from exo.shared.types.common import NodeId, SessionId
from exo.utils import STDIO_FDS
from exo.utils.channels import Receiver, channel
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
    async def create(cls, args: CliArgs) -> Self:
        node_id = get_node_zid()
        session_id = SessionId(master_node_id=node_id, election_clock=0)
        router = Router.create(
            node_id,
            namespace=args.namespace,
            listen_port=args.zenoh_port,
            discovery_service_port=args.discovery_port,
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
        offline = config.app().offline

        # Create DownloadCoordinator (unless --no-downloads)
        if args.downloads_enabled:
            download_coordinator = DownloadCoordinator(
                node_id,
                exo_shard_downloader(offline=offline),
                event_sender=event_router.sender(),
                download_command_receiver=router.receiver(topics.DOWNLOAD_COMMANDS),
                offline=offline,
            )
        else:
            download_coordinator = None

        if args.api_enabled:
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

        if args.worker_enabled:
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

        return cls(
            router,
            event_router,
            download_coordinator,
            worker,
            election,
            er_recv,
            master,
            api,
            node_id,
            offline,
            args.api_port,
        )

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
                    await anyio_checkpoint()
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
                    assert not result.is_new_master, (
                        "cannot be new master if we remain master"
                    )
                    logger.info("Node elected Master - maintaining self")
                elif (
                    result.session_id.master_node_id == self.node_id
                    and self.master is None
                ):
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
                    logger.info(
                        f"Node {result.session_id.master_node_id} elected master - demoting self"
                    )
                    await self.master.shutdown()
                    self.master = None
                else:
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


def main():
    # Parse args first & resolve/load bootstrap + app settings
    #   => --help or bad args don't require PID-locking
    args = CliArgs.parse()
    bootstrap_settings = BootstrapSettings.resolve(args.bootstrap)
    config.load(
        bootstrap_settings,
        AppSettings.resolve(args.app, bootstrap_settings),
    )

    # Exit early if cannot acquire PID file
    try:
        pidfile_path = config.bootstrap().pid_file
        pidfile = Pidfile(pidfile_path, 0o0600)
    except PidfileError as e:
        print(e, file=sys.stderr)
        raise SystemExit(1) from e

    try:
        if args.legacy_daemon:
            # keep stdio backed by explicit /dev/null streams. multiprocessing spawn expects
            # valid stdio FDs; letting DaemonContext close/reopen them can break runner startup.
            for stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
                if stream is not None:
                    stream.flush()
            stdin = open(os.devnull, "r")  # noqa: SIM115
            stdout = open(os.devnull, "w")  # noqa: SIM115
            stderr = open(os.devnull, "w")  # noqa: SIM115

            with DaemonContext(
                detach_process=True,
                files_preserve=[pidfile.as_raw_fd()],
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
            ):
                # cleanup loose file descriptors (as long as they aren't stdio)
                for f in (
                    f for f in (stdin, stdout, stderr) if f.fileno() not in STDIO_FDS
                ):
                    f.close()

                # 1) if daemonizing => fork then write PID
                try:
                    pidfile.write()
                except PidfileError as e:
                    print(e, file=sys.stderr)
                    raise SystemExit(1) from e
                main_inner(args)
        else:
            # 2) otherwise      => just write PID
            try:
                pidfile.write()
            except PidfileError as e:
                print(e, file=sys.stderr)
                raise SystemExit(1) from e
            main_inner(args)
    finally:
        pidfile.close()


def main_inner(args: CliArgs):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(max(soft, 65535), hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))

    mp.set_start_method("spawn", force=True)

    # TODO: Refactor the current verbosity system
    logger_setup(config.bootstrap().log_files.exo_log, config.app().verbosity)

    logger.info(f"pid = {os.getpid()}")
    logger.info(f"Discovery namespace: {args.namespace}")

    if config.app().offline:
        logger.info("Running in OFFLINE mode — no internet checks, local models only")

    if not config.app().continuous_batching_enabled:
        logger.info("Continuous batching disabled (--no-batch)")

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
