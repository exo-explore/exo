"""The deployments every node keeps on disk, and how two views of them are merged.

Deployments are decided by the master, but a master can lose them: a new master starts from
its own node's copy of the state, which may be behind, and in a heal the side that loses adopts
the winner's state. So every node keeps the deployments it has seen, and the stops (tombstones),
in a file, and tells each new master. Two views merge the same way everywhere: the union of the
deployments, without those stopped; for one both have, the higher revision wins. A stop always
wins, and nothing but a stop removes a deployment: a state that is empty or still catching up
never deletes anything.
"""

import contextlib
import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import final

import anyio
from loguru import logger
from pydantic import ValidationError

from exo.shared.types.commands import AnnounceDeployments, ForwarderCommand
from exo.shared.types.common import DeploymentId, NodeId, SystemId
from exo.shared.types.deployments import (
    REMOVED_DEPLOYMENTS_KEPT,
    Deployment,
    DeploymentRecord,
)
from exo.shared.types.state import State
from exo.utils.channels import Sender

# The file's format. A newer exo may write a newer one: this one refuses to read it, and leaves
# it alone, rather than overwrite deployments it can't understand.
FORMAT = 1


def record_of(
    deployment: Deployment, running_on: Sequence[NodeId] = ()
) -> DeploymentRecord:
    """What a node keeps of a deployment: the nodes it runs on now, or else last ran on."""
    return DeploymentRecord(
        deployment_id=deployment.deployment_id,
        spec=deployment.spec,
        model_card=deployment.model_card,
        revision=deployment.revision,
        last_nodes=list(running_on) or deployment.last_nodes,
    )


def records_of(state: State) -> list[DeploymentRecord]:
    running_on: dict[DeploymentId, list[NodeId]] = {
        instance.deployment_id: list(instance.shard_assignments.node_to_runner)
        for instance in state.instances.values()
    }
    return [
        record_of(deployment, running_on.get(deployment_id, ()))
        for deployment_id, deployment in state.deployments.items()
    ]


@dataclass
class Deployments:
    """A view of the deployments: those launched, and those stopped (with when)."""

    records: dict[DeploymentId, DeploymentRecord] = field(default_factory=dict)
    removed: dict[DeploymentId, str] = field(default_factory=dict)

    def merge(
        self,
        records: Iterable[DeploymentRecord],
        removed: Mapping[DeploymentId, str],
    ) -> bool:
        """Merge another view into this one. Returns whether this one changed."""
        before = (dict(self.records), dict(self.removed))
        for deployment_id, at in removed.items():
            if deployment_id not in self.removed or at > self.removed[deployment_id]:
                self.removed[deployment_id] = at
        for record in records:
            current = self.records.get(record.deployment_id)
            if (
                current is None
                or record.revision > current.revision
                # Where it ran isn't the user's: the latest word on it is kept
                or (
                    record.revision == current.revision
                    and record.last_nodes
                    and record.last_nodes != current.last_nodes
                )
            ):
                self.records[record.deployment_id] = record
        for deployment_id in self.removed:
            self.records.pop(deployment_id, None)
        kept = sorted(self.removed.items(), key=lambda item: item[1])
        self.removed = dict(kept[-REMOVED_DEPLOYMENTS_KEPT:])
        return (self.records, self.removed) != before


def _document(deployments: Deployments) -> str:
    return json.dumps(
        {
            "format": FORMAT,
            "deployments": [
                record.model_dump(mode="json", by_alias=True)
                for record in deployments.records.values()
            ],
            "removed": deployments.removed,
        },
        indent=1,
    )


@dataclass
class Loaded:
    deployments: Deployments
    # False when the file is of a format this exo doesn't know: it must not be written over
    writable: bool


def load(path: Path) -> Loaded:
    """The deployments in the file. A record that can't be read is left out; a file that can't
    be read at all is moved aside; a file of a newer format is left as it is, unread."""
    try:
        text = path.read_text()
    except FileNotFoundError:
        return Loaded(Deployments(), writable=True)
    try:
        document: object = json.loads(text)  # pyright: ignore[reportAny]
        if not isinstance(document, dict):
            raise ValueError("not an object")
    except ValueError as error:
        aside = path.with_name(
            f"{path.name}.unreadable-{datetime.now(tz=timezone.utc):%Y%m%dT%H%M%S}"
        )
        logger.warning(f"Can't read {path} ({error}); moved it to {aside}")
        with contextlib.suppress(OSError):
            path.rename(aside)
        return Loaded(Deployments(), writable=True)
    file_format = document.get("format")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if not isinstance(file_format, int) or file_format > FORMAT:
        logger.warning(
            f"{path} is of format {file_format}, which this version of exo doesn't know: "
            "leaving it alone, and not keeping deployments on this node"
        )
        return Loaded(Deployments(), writable=False)
    deployments = Deployments()
    raw_records: object = document.get("deployments", [])  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    for raw in raw_records if isinstance(raw_records, list) else []:  # pyright: ignore[reportUnknownVariableType]
        try:
            # From JSON, as on the network: strict models take enums only as JSON strings
            record = DeploymentRecord.model_validate_json(json.dumps(raw))
        except ValidationError as error:
            logger.warning(
                f"Leaving out a deployment in {path} that can't be read: {error}"
            )
            continue
        deployments.records[record.deployment_id] = record
    raw_removed: object = document.get("removed", {})  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if isinstance(raw_removed, dict):
        for deployment_id, at in raw_removed.items():  # pyright: ignore[reportUnknownVariableType]
            if isinstance(deployment_id, str) and isinstance(at, str):
                deployments.removed[DeploymentId(deployment_id)] = at
    deployments.merge([], {})
    return Loaded(deployments, writable=True)


def save(path: Path, deployments: Deployments) -> None:
    """Write the file in one go: a crash midway leaves the previous one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(_document(deployments))
    os.replace(temporary, path)


# How often the keeper looks at the state for deployments to save. The custom model cards,
# which are kept the same way, use the same interval.
SYNC_INTERVAL = 1.0
# How often it tells the master again while the master's state still lacks some of them (the
# announcement may have been lost, e.g. while the master changed). Same as the custom cards.
ANNOUNCE_INTERVAL = 10.0


@final
class DeploymentKeeper:
    """Keeps this node's copy of the deployments on disk, and tells each new master.

    A deployment read from disk that no master has run since this node started is announced
    as restored: if no node has it running, the whole cluster restarted, and it comes back
    held until the user resumes it.
    """

    def __init__(
        self,
        path: Path,
        state: Callable[[], State | None],
        command_sender: Sender[ForwarderCommand],
    ):
        loaded = load(path)
        self._path = path
        self._writable = loaded.writable
        self._known = loaded.deployments
        # Read from disk and not yet seen running in a master's state
        self._restored: set[DeploymentId] = set(loaded.deployments.records)
        # Running in the state when this node last looked. A new master's own hold of one is
        # its side's, and is lifted. The first announcement to a new master tells what the
        # previous one was running, as this node last saw it before the new master's state
        # replaced it; later ones tell what runs now, so a hold the new master made since
        # isn't lifted by an old view
        self._running: set[DeploymentId] = set()
        self._running_before_the_master_changed: set[DeploymentId] | None = None
        self._state = state
        self._command_sender = command_sender
        self._system_id = SystemId()
        # Each master is told once (it may lack stops, or a deployment held as restored that
        # this node has run), and again while its state lacks a deployment this node knows
        self._announce_now = True
        self._next_announcement = 0.0

    def sync(self) -> None:
        """Take in what the state knows, and save it if anything changed."""
        state = self._state()
        if state is None:
            return
        self._restored -= {
            deployment_id
            for deployment_id, deployment in state.deployments.items()
            if deployment.hold is None or deployment.hold.kind != "Restored"
        }
        self._running = {
            instance.deployment_id for instance in state.instances.values()
        }
        changed = self._known.merge(records_of(state), state.removed_deployments)
        if changed and self._writable:
            try:
                save(self._path, self._known)
            except OSError as error:
                logger.warning(
                    f"Couldn't save the deployments to {self._path}: {error}"
                )

    def unknown_to(self, state: State) -> bool:
        """Whether the state lacks something this node knows: a deployment, the user's later
        change to one (a resume), or a stop. A stop older than every one the state keeps
        doesn't count: the state would drop it again."""
        for deployment_id, record in self._known.records.items():
            current = state.deployments.get(deployment_id)
            if current is None and deployment_id not in state.removed_deployments:
                return True
            if current is not None and record.revision > current.revision:
                return True
        full = len(state.removed_deployments) >= REMOVED_DEPLOYMENTS_KEPT
        oldest = min(state.removed_deployments.values(), default=None)
        return any(
            deployment_id not in state.removed_deployments
            and not (full and oldest is not None and at <= oldest)
            for deployment_id, at in self._known.removed.items()
        )

    def announcement(self) -> AnnounceDeployments:
        return AnnounceDeployments(
            deployments=list(self._known.records.values()),
            removed=dict(self._known.removed),
            restored=sorted(self._restored & self._known.records.keys()),
            running=sorted(
                (
                    self._running
                    if self._running_before_the_master_changed is None
                    else self._running_before_the_master_changed
                )
                & self._known.records.keys()
            ),
        )

    def follow_new_master(self) -> None:
        """Tell the new master at the next round, with what the previous one was running (as
        this node last saw it, before the new master's state replaces it)."""
        self._announce_now = True
        self._running_before_the_master_changed = set(self._running)

    def told(self) -> None:
        """An announcement went out: the next ones tell what runs now."""
        self._running_before_the_master_changed = None

    def announcement_due(self, state: State, now: float) -> bool:
        if not (self._known.records or self._known.removed):
            return False
        return self._announce_now or (
            self.unknown_to(state) and now >= self._next_announcement
        )

    async def run(self) -> None:
        while True:
            self.sync()
            state = self._state()
            now = anyio.current_time()
            if state is not None and self.announcement_due(state, now):
                self._announce_now = False
                self._next_announcement = now + ANNOUNCE_INTERVAL
                await self._command_sender.send(
                    ForwarderCommand(
                        origin=self._system_id, command=self.announcement()
                    )
                )
                self.told()
            await anyio.sleep(SYNC_INTERVAL)
