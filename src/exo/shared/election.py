from typing import Self

import anyio
from anyio import (
    CancelScope,
    Event,
    create_task_group,
    get_cancelled_exc_class,
)
from anyio.abc import TaskGroup
from loguru import logger

from exo.routing.connection_message import ConnectionMessage
from exo.shared.types.commands import ForwarderCommand
from exo.shared.types.common import NodeId, SessionId
from exo.utils.channels import Receiver, Sender
from exo.utils.pydantic_ext import CamelCaseModel

DEFAULT_ELECTION_TIMEOUT = 3.0


class ElectionMessage(CamelCaseModel):
    clock: int
    seniority: int
    proposed_session: SessionId
    commands_seen: int

    # Could eventually include a list of neighbour nodes for centrality
    def __lt__(self, other: Self) -> bool:
        if self.clock != other.clock:
            return self.clock < other.clock
        if self.seniority != other.seniority:
            return self.seniority < other.seniority
        elif self.commands_seen != other.commands_seen:
            return self.commands_seen < other.commands_seen
        else:
            return (
                self.proposed_session.master_node_id
                < other.proposed_session.master_node_id
            )


class ElectionResult(CamelCaseModel):
    session_id: SessionId
    won_clock: int
    is_new_master: bool


class Election:
    def __init__(
        self,
        node_id: NodeId,
        *,
        election_message_receiver: Receiver[ElectionMessage],
        election_message_sender: Sender[ElectionMessage],
        election_result_sender: Sender[ElectionResult],
        connection_message_receiver: Receiver[ConnectionMessage],
        command_receiver: Receiver[ForwarderCommand],
        is_candidate: bool = True,
        seniority: int = 0,
    ):
        # If we aren't a candidate, simply don't increment seniority.
        # For reference: This node can be elected master if all nodes are not master candidates
        # Any master candidate will automatically win out over this node.
        self.seniority = seniority if is_candidate else -1
        self.clock = 0
        self.node_id = node_id
        self.commands_seen = 0
        # Every node spawns as master
        self.current_session: SessionId = SessionId(
            master_node_id=node_id, election_clock=0
        )

        # Senders/Receivers
        self._em_sender = election_message_sender
        self._em_receiver = election_message_receiver
        self._er_sender = election_result_sender
        self._cm_receiver = connection_message_receiver
        self._co_receiver = command_receiver

        # Campaign state
        self._candidates: list[ElectionMessage] = []
        self._campaign_cancel_scope: CancelScope | None = None
        self._campaign_done: Event | None = None
        self._tg: TaskGroup | None = None

    async def run(self):
        logger.info("Starting Election")
        async with create_task_group() as tg:
            self._tg = tg
            tg.start_soon(self._election_receiver)
            tg.start_soon(self._connection_receiver)
            tg.start_soon(self._command_counter)

            # And start an election immediately, that instantly resolves
            candidates: list[ElectionMessage] = []
            logger.debug("Starting initial campaign")
            self._candidates = candidates
            await self._campaign(candidates, campaign_timeout=0.0)
            logger.debug("Initial campaign finished")

        # Cancel and wait for the last election to end
        if self._campaign_cancel_scope is not None:
            logger.debug("Cancelling campaign")
            self._campaign_cancel_scope.cancel()
        if self._campaign_done is not None:
            logger.debug("Waiting for campaign to finish")
            await self._campaign_done.wait()
        logger.debug("Campaign cancelled and finished")
        logger.info("Election finished")

    async def elect(self, em: ElectionMessage) -> None:
        logger.debug(f"Electing: {em}")
        is_new_master = em.proposed_session != self.current_session
        self.current_session = em.proposed_session
        logger.debug(f"Current session: {self.current_session}")
        await self._er_sender.send(
            ElectionResult(
                won_clock=em.clock,
                session_id=em.proposed_session,
                is_new_master=is_new_master,
            )
        )

    async def shutdown(self) -> None:
        if not self._tg:
            logger.warning(
                "Attempted to shutdown election service that was not running"
            )
            return
        self._tg.cancel_scope.cancel()

    async def _election_receiver(self) -> None:
        with self._em_receiver as election_messages:
            async for message in election_messages:
                logger.debug(f"Election message received: {message}")
                if message.proposed_session.master_node_id == self.node_id:
                    logger.debug("Dropping message from ourselves")
                    # Drop messages from us (See exo.routing.router)
                    continue
                # If a new round is starting, we participate
                if message.clock > self.clock:
                    self.clock = message.clock
                    logger.debug(f"New clock: {self.clock}")
                    assert self._tg is not None
                    logger.debug("Starting new campaign")
                    candidates: list[ElectionMessage] = [message]
                    logger.debug(f"Candidates: {candidates}")
                    logger.debug(f"Current candidates: {self._candidates}")
                    self._candidates = candidates
                    logger.debug(f"New candidates: {self._candidates}")
                    logger.debug("Starting new campaign")
                    self._tg.start_soon(
                        self._campaign, candidates, DEFAULT_ELECTION_TIMEOUT
                    )
                    logger.debug("Campaign started")
                    continue
                # Dismiss old messages
                if message.clock < self.clock:
                    logger.debug(f"Dropping old message: {message}")
                    continue
                logger.debug(f"Election added candidate {message}")
                # Now we are processing this rounds messages - including the message that triggered this round.
                self._candidates.append(message)

    async def _connection_receiver(self) -> None:
        with self._cm_receiver as connection_messages:
            async for first in connection_messages:
                # Delay after connection message for time to symmetrically setup
                await anyio.sleep(0.2)
                rest = connection_messages.collect()

                logger.debug(
                    f"Connection messages received: {first} followed by {rest}"
                )
                logger.debug(f"Current clock: {self.clock}")
                # These messages are strictly peer to peer
                self.clock += 1
                logger.debug(f"New clock: {self.clock}")
                assert self._tg is not None
                candidates: list[ElectionMessage] = []
                self._candidates = candidates
                logger.debug("Starting new campaign")
                self._tg.start_soon(
                    self._campaign, candidates, DEFAULT_ELECTION_TIMEOUT
                )
                logger.debug("Campaign started")
                logger.debug("Connection message added")

    async def _command_counter(self) -> None:
        with self._co_receiver as commands:
            async for _command in commands:
                self.commands_seen += 1

    async def _campaign(
        self, candidates: list[ElectionMessage], campaign_timeout: float
    ) -> None:
        clock = self.clock

        # Kill the old campaign
        if self._campaign_cancel_scope:
            logger.info("Cancelling other campaign")
            self._campaign_cancel_scope.cancel()
        if self._campaign_done:
            logger.info("Waiting for other campaign to finish")
            await self._campaign_done.wait()

        done = Event()
        self._campaign_done = done
        scope = CancelScope()
        self._campaign_cancel_scope = scope

        try:
            with scope:
                logger.debug(f"Election {clock} started")

                status = self._election_status(clock)
                candidates.append(status)
                await self._em_sender.send(status)

                logger.debug(f"Sleeping for {campaign_timeout} seconds")
                await anyio.sleep(campaign_timeout)
                # minor hack - rebroadcast status in case anyone has missed it.
                await self._em_sender.send(status)
                logger.debug("Woke up from sleep")
                # add an anyio checkpoint - anyio.lowlevel.chekpoint() or checkpoint_if_cancelled() is preferred, but wasn't typechecking last I checked
                await anyio.sleep(0)

                # Election finished!
                elected = max(candidates)
                logger.debug(f"Election queue {candidates}")
                logger.debug(f"Elected: {elected}")
                if (
                    self.node_id == elected.proposed_session.master_node_id
                    and self.seniority >= 0
                ):
                    logger.debug(
                        f"Node is a candidate and seniority is {self.seniority}"
                    )
                    self.seniority = max(self.seniority, len(candidates))
                    logger.debug(f"New seniority: {self.seniority}")
                else:
                    logger.debug(
                        f"Node is not a candidate or seniority is not {self.seniority}"
                    )
                logger.debug(
                    f"Election finished, new SessionId({elected.proposed_session}) with queue {candidates}"
                )
                logger.debug("Sending election result")
                await self.elect(elected)
                logger.debug("Election result sent")
        except get_cancelled_exc_class():
            logger.debug(f"Election {clock} cancelled")
        finally:
            logger.debug(f"Election {clock} finally")
            if self._campaign_cancel_scope is scope:
                self._campaign_cancel_scope = None
            logger.debug("Setting done event")
            done.set()
            logger.debug("Done event set")

    def _election_status(self, clock: int | None = None) -> ElectionMessage:
        c = self.clock if clock is None else clock
        return ElectionMessage(
            proposed_session=(
                self.current_session
                if self.current_session.master_node_id == self.node_id
                else SessionId(master_node_id=self.node_id, election_clock=c)
            ),
            clock=c,
            seniority=self.seniority,
            commands_seen=self.commands_seen,
        )
