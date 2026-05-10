import argparse
import ipaddress
import multiprocessing as mp
import os
import resource
import signal
import subprocess
import sys
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
from exo.download.peer_file_server import PeerFileServer
from exo.master.main import Master
from exo.routing.event_router import EventRouter
from exo.routing.router import Router, get_node_id_keypair
from exo.shared.constants import (
    EXO_LOG,
    EXO_MODELS_DIRS,
    EXO_MODELS_READ_ONLY_DIRS,
    EXO_PEER_DOWNLOAD_PORT,
)
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
    _libp2p_port: int
    _peer_download_port: int
    peer_file_server: PeerFileServer | None = None
    _tg: TaskGroup = field(init=False, default_factory=TaskGroup)

    @classmethod
    async def create(cls, args: "Args") -> Self:
        # Codex P1 (PR #16 round-(N+3), main.py:74): scope the on-disk
        # node-ID keypair by the *combination* of ports the operator
        # has chosen, not just ``--peer-download-port``. The earlier
        # peer-download-only scope leaked identity collisions when
        # ``--no-downloads`` / ``--no-peer-download`` is set: that
        # mode doesn't bind the peer file server, so two same-host
        # processes can legitimately keep the default
        # ``peer_download_port`` and would then load the same scoped
        # keypair file -- producing identical ``NodeId``s and
        # breaking election/routing's unique-NodeId invariants.
        #
        # Combined-port scoping is robust against every same-host
        # multi-process configuration: at least one of the listening
        # ports MUST differ between processes (libp2p, peer-download,
        # api -- each is a distinct local socket bind), so the scope
        # tuple differs whenever the actual configuration differs.
        # Single-process deployments on default ports keep a stable
        # filename (e.g. ``node_id.libp2p-0.api-52415.peer-52416.keypair``)
        # so identity persists across restarts.
        process_scope = _node_id_keypair_scope(args)
        keypair = get_node_id_keypair(process_scope=process_scope)
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

        peer_file_server: PeerFileServer | None = None
        peer_download_enabled = not args.no_peer_download and not args.no_downloads

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
                # Each node now binds its own peer-download listener on
                # ``--peer-download-port`` (default ``EXO_PEER_DOWNLOAD_PORT``).
                # The Worker uses this same value when discovering peers,
                # so all nodes in a cluster MUST agree on it (typically
                # via the shared ``EXO_PEER_DOWNLOAD_PORT`` env var).
                # Pre-fix this was a single import-time module constant,
                # making same-host multi-node setups impossible (Codex
                # P2, PR #16 round 3).
                peer_download_port=args.peer_download_port,
            )
        else:
            worker = None

        if peer_download_enabled:
            # Serve from every configured model directory so peers can fetch
            # any locally-resident shard regardless of which directory the
            # downloader landed it in. ``EXO_MODELS_DIRS`` already includes
            # ``EXO_DEFAULT_MODELS_DIR`` as its first entry; ``EXO_MODELS_READ_ONLY_DIRS``
            # captures pre-populated mounts (e.g. shared NFS caches) that
            # ``select_download_dir_for_shard`` excludes from new writes but
            # which other peers still benefit from being able to read.
            peer_file_server = PeerFileServer(
                host="0.0.0.0",
                port=args.peer_download_port,
                models_dirs=(*EXO_MODELS_DIRS, *EXO_MODELS_READ_ONLY_DIRS),
            )

        if not args.no_downloads:
            download_coordinator: DownloadCoordinator | None = DownloadCoordinator(
                node_id,
                exo_shard_downloader(
                    offline=args.offline,
                    peer_download_enabled=peer_download_enabled,
                ),
                event_sender=event_router.sender(),
                download_command_receiver=router.receiver(topics.DOWNLOAD_COMMANDS),
                offline=args.offline,
            )
        else:
            download_coordinator = None

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
            args.libp2p_port,
            args.peer_download_port,
            peer_file_server,
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
            if self.peer_file_server:
                tg.start_soon(self.peer_file_server.run)
            if self.download_coordinator:
                tg.start_soon(self.download_coordinator.run)
            if self.worker:
                tg.start_soon(self.worker.run)
            if self.master:
                tg.start_soon(self.master.run)
            if self.api:
                tg.start_soon(self.api.run)
            if sys.platform == "darwin" and self._libp2p_port != 0:
                tg.start_soon(
                    _darwin_mdns_broadcast_announcer,
                    self.node_id,
                    self._libp2p_port,
                )
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
                    if self.master is not None:
                        await self.master.shutdown()
                        self.master = None
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
                            exo_shard_downloader(
                                offline=self.offline,
                                peer_download_enabled=self.peer_file_server is not None,
                            ),
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
                            peer_download_port=self._peer_download_port,
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


def _node_id_keypair_scope(args: "Args") -> str:
    """Produce a stable per-process scope for the node-ID keypair file.

    Combines every listening port the operator could plausibly
    distinguish between same-host processes: ``--libp2p-port``,
    ``--api-port``, and ``--peer-download-port``. At least one of
    these MUST differ between two processes that share a host (each
    is a distinct local socket bind), so the resulting scope is
    always unique per process while remaining stable across
    restarts of the same configuration.

    Used by :func:`get_node_id_keypair` to avoid two same-host
    processes loading the same scoped keypair file when peer
    download is disabled (which would otherwise let them collide
    on the default ``peer_download_port`` since no socket is
    actually being bound). See Codex P1 (PR #16 round-(N+3),
    main.py:74).

    Codex P1 (PR #16 round-(N+8), main.py:457): when
    ``--libp2p-port 0`` is set, the configured value is the literal
    ``0`` even though each process actually binds a different
    ephemeral port at runtime. Two same-host worker-only processes
    (no API, no peer download) sharing the default
    ``peer_download_port`` and ``api_port`` -- but each binding
    ``libp2p_port=0`` -- would otherwise produce identical scope
    strings ``"libp2p-0.api-...peer-..."`` and load the same
    keypair file, breaking the unique-NodeId invariant.
    Stability across restarts is impossible in this configuration
    anyway (the OS hands out a different ephemeral port on every
    bind), so fold in ``os.getpid()`` as a per-process
    discriminator. The trade-off (ephemeral identity for
    ephemeral ports) is the right semantic: the operator opted
    into ephemeral binding by setting ``libp2p_port=0``.
    """
    if args.libp2p_port == 0:
        return (
            f"libp2p-pid-{os.getpid()}."
            f"api-{args.api_port}.peer-{args.peer_download_port}"
        )
    return (
        f"libp2p-{args.libp2p_port}.api-{args.api_port}.peer-{args.peer_download_port}"
    )


def _darwin_en0_ip_address() -> str | None:
    try:
        return subprocess.check_output(
            ["ipconfig", "getifaddr", "en0"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _darwin_en0_broadcast_address(ip_address: str) -> str | None:
    try:
        subnet_mask = subprocess.check_output(
            ["ipconfig", "getoption", "en0", "subnet_mask"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        interface = ipaddress.IPv4Interface(f"{ip_address}/{subnet_mask}")
        return str(interface.network.broadcast_address)
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


async def _darwin_mdns_broadcast_announcer(node_id: NodeId, libp2p_port: int) -> None:
    ip_address = _darwin_en0_ip_address()
    if not ip_address:
        logger.debug("Darwin mDNS broadcast announcer disabled: no en0 IPv4 address")
        return

    broadcast_address = _darwin_en0_broadcast_address(ip_address)
    logger.debug(
        f"Darwin mDNS announcer advertising {node_id} at {ip_address}:{libp2p_port}"
    )
    command = [
        sys.executable,
        "-m",
        "exo.routing.mdns_announcer",
        "--node-id",
        str(node_id),
        "--ip-address",
        ip_address,
        "--libp2p-port",
        str(libp2p_port),
    ]
    if broadcast_address is not None:
        command.extend(["--broadcast-address", broadcast_address])
    process = subprocess.Popen(
        command,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
    )
    try:
        while process.poll() is None:
            await anyio.sleep(60)
        logger.debug(
            f"Darwin mDNS announcer subprocess exited with {process.returncode}"
        )
    finally:
        if process.poll() is None:
            process.terminate()
            with anyio.move_on_after(2):
                while process.poll() is None:
                    await anyio.sleep(0.1)
            if process.poll() is None:
                process.kill()
                await anyio.sleep(0)


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

    # Set trust_remote_code override env var for runner subprocesses
    if args.trust_remote_code:
        os.environ["EXO_TRUST_REMOTE_CODE"] = "1"
        logger.warning(
            "--trust-remote-code enabled: models may execute arbitrary code during loading"
        )

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
    no_peer_download: bool = False
    offline: bool = os.getenv("EXO_OFFLINE", "false").lower() == "true"
    no_batch: bool = False
    fast_synch: bool | None = None  # None = auto, True = force on, False = force off
    bootstrap_peers: list[str] = []
    libp2p_port: int
    # Per-process listener port for peer-to-peer model file serving.
    # Defaults to ``EXO_PEER_DOWNLOAD_PORT`` so existing single-node-per-
    # host deployments keep working unchanged. Operators running
    # multiple nodes on the same host MUST set this to a distinct value
    # for each process; the cluster-wide convention is that every node
    # exposes the same port, since peer discovery currently uses each
    # node's local value as the assumed remote endpoint (see
    # ``Worker._peer_download_port``). A future state-sync change can
    # advertise per-node ports across the cluster -- tracked as a
    # follow-up to Codex P2 (PR #16 round 3).
    peer_download_port: PositiveInt = EXO_PEER_DOWNLOAD_PORT
    trust_remote_code: bool = False

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
            "--no-peer-download",
            action="store_true",
            help="Disable peer-to-peer model downloads (each node downloads from HuggingFace independently)",
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
            "--trust-remote-code",
            action="store_true",
            help="Allow models to execute custom code during tokenizer loading (security-sensitive, CLI-only)",
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
        parser.add_argument(
            "--peer-download-port",
            type=int,
            default=EXO_PEER_DOWNLOAD_PORT,
            dest="peer_download_port",
            help=(
                "TCP port for peer-to-peer model file serving (default: "
                "EXO_PEER_DOWNLOAD_PORT, currently 52416). Required to "
                "differ between processes when running multiple nodes "
                "on the same host; otherwise the second node's "
                "PeerFileServer hits 'address already in use'. All "
                "nodes in a cluster must use the same value (peer "
                "discovery uses the local port as the assumed remote "
                "port)."
            ),
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
