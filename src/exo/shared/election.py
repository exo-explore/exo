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
from exo.shared.types.common import NodeId
from exo.utils.channels import Receiver, Sender
from exo.utils.pydantic_ext import CamelCaseModel

ELECTION_TIMEOUT = 3.0


class ElectionMessage(CamelCaseModel):
    clock: int
    seniority: int
    node_id: NodeId

    # Could eventually include a list of neighbour nodes for centrality
    def __lt__(self, other: Self):
        if self.seniority != other.seniority:
            return self.seniority < other.seniority
        else:
            return self.node_id < other.node_id


class ElectionResult(CamelCaseModel):
    node_id: NodeId
    is_new_master: bool
    historic_messages: list[ConnectionMessage]


class Election:
    def __init__(
        self,
        node_id: NodeId,
        election_message_receiver: Receiver[ElectionMessage],
        election_message_sender: Sender[ElectionMessage],
        election_result_sender: Sender[ElectionResult],
        connection_message_receiver: Receiver[ConnectionMessage],
        *,
        is_candidate: bool = True,
        seniority: int = 0,
    ):
        # If we aren't a candidate, simply don't increment seniority.
        # For reference: This node can be elected master if all nodes are not master candidates
        # Any master candidate will automatically win out over this node.
        self.seniority = seniority if is_candidate else -1
        self.clock = 0
        self.node_id = node_id
        # Every node spawns as master
        self.master_node_id: NodeId = node_id

        self._em_sender = election_message_sender
        self._em_receiver = election_message_receiver
        self._er_sender = election_result_sender
        self._cm_receiver = connection_message_receiver

        # Campaign state
        self._candidates: list[ElectionMessage] = []
        self._campaign_cancel_scope: CancelScope | None = None
        self._campaign_done: Event | None = None
        self._tg: TaskGroup | None = None
        self._connection_messages: list[ConnectionMessage] = []

    async def run(self):
        logger.info("Starting Election")
        async with create_task_group() as tg:
            self._tg = tg
            tg.start_soon(self._election_receiver)
            tg.start_soon(self._connection_receiver)
            await self._campaign(None)

        if self._campaign_cancel_scope is not None:
            self._campaign_cancel_scope.cancel()
        # Only exit once the latest campaign has finished
        if self._campaign_done is not None:
            await self._campaign_done.wait()

    async def elect(self, node_id: NodeId) -> None:
        is_new_master = node_id != self.master_node_id
        self.master_node_id = node_id
        await self._er_sender.send(
            ElectionResult(
                node_id=node_id,
                is_new_master=is_new_master,
                historic_messages=self._connection_messages,
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
                if message.node_id == self.node_id:
                    # Drop messages from us (See exo.routing.router)
                    continue
                # If a new round is starting, we participate
                if message.clock > self.clock:
                    self.clock = message.clock
                    await self._campaign(message)
                    continue
                # Dismiss old messages
                if message.clock < self.clock:
                    continue
                logger.debug(f"Election added candidate {message}")
                # Now we are processing this rounds messages - including the message that triggered this round.
                self._candidates.append(message)

    async def _connection_receiver(self) -> None:
        with self._cm_receiver as connection_messages:
            async for msg in connection_messages:
                # These messages are strictly peer to peer
                self.clock += 1
                await self._campaign(None)
                self._connection_messages.append(msg)

    async def _campaign(self, initial_message: ElectionMessage | None) -> None:
        # Kill the old campaign
        if self._campaign_cancel_scope:
            self._campaign_cancel_scope.cancel()
        if self._campaign_done:
            await self._campaign_done.wait()

        candidates: list[ElectionMessage] = []
        if initial_message:
            candidates.append(initial_message)
        self._candidates = candidates
        done = Event()
        self._campaign_done = done

        assert self._tg is not None, (
            "Election campaign started before election service initialized"
        )
        # Spin off a new campaign
        self._tg.start_soon(self._complete_campaign, self.clock, candidates, done)

    async def _complete_campaign(
        self, clock: int, candidates: list[ElectionMessage], done: Event
    ) -> None:
        scope = CancelScope()
        try:
            with scope:
                self._campaign_cancel_scope = scope
                logger.info(f"Election {clock} started")

                candidates.append(self._election_status(clock))
                await self._em_sender.send(self._election_status(clock))

                await anyio.sleep(ELECTION_TIMEOUT)

                # Election finished!
                candidates = sorted(candidates)
                logger.debug(f"Election queue {candidates}")
                elected = candidates[-1]
                logger.info("Election finished")
                if self.node_id == elected.node_id and self.seniority >= 0:
                    self.seniority = max(self.seniority, len(candidates))
                await self.elect(elected.node_id)
        except get_cancelled_exc_class():
            logger.info("Election cancelled")
        finally:
            if self._campaign_cancel_scope is scope:
                self._campaign_cancel_scope = None
        done.set()

    def _election_status(self, clock: int | None = None) -> ElectionMessage:
        c = self.clock if clock is None else clock
        return ElectionMessage(clock=c, seniority=self.seniority, node_id=self.node_id)
