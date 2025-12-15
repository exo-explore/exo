import argparse
import multiprocessing as mp
import signal
from dataclasses import dataclass, field
from typing import Self

import anyio
from anyio.abc import TaskGroup
from loguru import logger
from pydantic import PositiveInt

import exo.routing.topics as topics
from exo.master.api import API  # TODO: should API be in master?
from exo.master.main import Master
from exo.routing.router import Router, get_node_id_keypair
from exo.shared.constants import EXO_LOG
from exo.shared.election import Election, ElectionResult
from exo.shared.logging import logger_cleanup, logger_setup
from exo.shared.types.common import NodeId, SessionId
from exo.utils.channels import Receiver, channel
from exo.utils.pydantic_ext import CamelCaseModel
from exo.worker.download.impl_shard_downloader import exo_shard_downloader
from exo.worker.main import Worker


# I marked this as a dataclass as I want trivial constructors.
@dataclass
class Node:
    router: Router
    worker: Worker
    election: Election  # Every node participates in election, as we do want a node to become master even if it isn't a master candidate if no master candidates are present.
    election_result_receiver: Receiver[ElectionResult]
    master: Master | None
    api: API | None

    node_id: NodeId
    _tg: TaskGroup = field(init=False, default_factory=anyio.create_task_group)

    @classmethod
    async def create(cls, args: "Args") -> "Self":
        keypair = get_node_id_keypair()
        node_id = NodeId(keypair.to_peer_id().to_base58())
        session_id = SessionId(master_node_id=node_id, election_clock=0)
        router = Router.create(keypair)
        await router.register_topic(topics.GLOBAL_EVENTS)
        await router.register_topic(topics.LOCAL_EVENTS)
        await router.register_topic(topics.COMMANDS)
        await router.register_topic(topics.ELECTION_MESSAGES)
        await router.register_topic(topics.CONNECTION_MESSAGES)

        logger.info(f"Starting node {node_id}")
        if args.spawn_api:
            api = API(
                node_id,
                session_id,
                port=args.api_port,
                global_event_receiver=router.receiver(topics.GLOBAL_EVENTS),
                command_sender=router.sender(topics.COMMANDS),
                election_receiver=router.receiver(topics.ELECTION_MESSAGES),
            )
        else:
            api = None

        worker = Worker(
            node_id,
            session_id,
            exo_shard_downloader(),
            connection_message_receiver=router.receiver(topics.CONNECTION_MESSAGES),
            global_event_receiver=router.receiver(topics.GLOBAL_EVENTS),
            local_event_sender=router.sender(topics.LOCAL_EVENTS),
            command_sender=router.sender(topics.COMMANDS),
        )
        # We start every node with a master
        master = Master(
            node_id,
            session_id,
            global_event_sender=router.sender(topics.GLOBAL_EVENTS),
            local_event_receiver=router.receiver(topics.LOCAL_EVENTS),
            command_receiver=router.receiver(topics.COMMANDS),
            tb_only=args.tb_only,
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

        return cls(router, worker, election, er_recv, master, api, node_id)

    async def run(self):
        async with self._tg as tg:
            signal.signal(signal.SIGINT, lambda _, __: self.shutdown())
            tg.start_soon(self.router.run)
            tg.start_soon(self.worker.run)
            tg.start_soon(self.election.run)
            if self.master:
                tg.start_soon(self.master.run)
            if self.api:
                tg.start_soon(self.api.run)
            tg.start_soon(self._elect_loop)

    def shutdown(self):
        # if this is our second call to shutdown, just sys.exit
        if self._tg.cancel_scope.cancel_called:
            import sys

            sys.exit(1)
        self._tg.cancel_scope.cancel()

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

                if (
                    result.session_id.master_node_id == self.node_id
                    and self.master is not None
                ):
                    logger.info("Node elected Master")
                elif (
                    result.session_id.master_node_id == self.node_id
                    and self.master is None
                ):
                    logger.info("Node elected Master - promoting self")
                    self.master = Master(
                        self.node_id,
                        result.session_id,
                        global_event_sender=self.router.sender(topics.GLOBAL_EVENTS),
                        local_event_receiver=self.router.receiver(topics.LOCAL_EVENTS),
                        command_receiver=self.router.receiver(topics.COMMANDS),
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
                    await anyio.sleep(0)
                    if self.worker:
                        self.worker.shutdown()
                        # TODO: add profiling etc to resource monitor
                        self.worker = Worker(
                            self.node_id,
                            result.session_id,
                            exo_shard_downloader(),
                            connection_message_receiver=self.router.receiver(
                                topics.CONNECTION_MESSAGES
                            ),
                            global_event_receiver=self.router.receiver(
                                topics.GLOBAL_EVENTS
                            ),
                            local_event_sender=self.router.sender(topics.LOCAL_EVENTS),
                            command_sender=self.router.sender(topics.COMMANDS),
                        )
                        self._tg.start_soon(self.worker.run)
                    if self.api:
                        self.api.reset(result.session_id, result.won_clock)
                else:
                    if self.api:
                        self.api.unpause(result.won_clock)


def main():
    args = Args.parse()

    mp.set_start_method("spawn")
    # TODO: Refactor the current verbosity system
    logger_setup(EXO_LOG, args.verbosity)
    logger.info("Starting EXO")

    node = anyio.run(Node.create, args)
    anyio.run(node.run)
    logger.info("EXO Shutdown complete")
    logger_cleanup()


class Args(CamelCaseModel):
    verbosity: int = 0
    force_master: bool = False
    spawn_api: bool = False
    api_port: PositiveInt = 8000
    tb_only: bool = False

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
            default=8000,
        )
        parser.add_argument(
            "--tb-only",
            action="store_true",
            dest="tb_only",
        )

        args = parser.parse_args()
        return cls(**vars(args))  # pyright: ignore[reportAny] - We are intentionally validating here, we can't do it statically
