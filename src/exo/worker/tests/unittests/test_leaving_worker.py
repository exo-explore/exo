"""A worker whose node is leaving the cluster reports nothing more about the node: applied
after the departure, it would put the node back into the cluster's state."""

from collections.abc import Callable
from pathlib import Path

import anyio
import pytest

import exo.worker.main as worker_main
from exo.shared.models.model_cards import ModelCard
from exo.shared.types.common import ModelId
from exo.shared.types.events import Event, NodeDownloadProgress, TaskStatusUpdated
from exo.shared.types.state import State
from exo.shared.types.tasks import TaskStatus
from exo.shared.types.worker.runners import RunnerIdle
from exo.utils.channels import Receiver
from exo.worker.main import Worker
from exo.worker.tests.constants import INSTANCE_1_ID, NODE_A
from exo.worker.tests.unittests.test_follow_new_master import (
    INSTANCE,
    running_worker,
    with_runner,
)


@pytest.mark.parametrize("leaving", [False, True])
async def test_a_leaving_worker_does_not_report_a_model_it_finds_on_disk(
    leaving: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The node's instance is still in its state while its departure is on its way, so the
    worker can still look for the instance's model, find it and report it."""

    def found(_model_id: ModelId, _card: ModelCard | None = None) -> Path:
        return tmp_path

    monkeypatch.setattr(worker_main, "resolve_existing_model", found)
    state = State(instances={INSTANCE_1_ID: INSTANCE}, downloads={NODE_A: []})

    def planning(worker: Worker) -> Callable[[], object]:
        with_runner(worker, INSTANCE, RunnerIdle())
        if leaving:
            worker.stop_reporting()
        return worker.plan_step

    sent: list[Event] = []

    async def until_the_download_task_is_done(events: Receiver[Event]) -> None:
        with anyio.fail_after(5):
            while not any(
                isinstance(event, TaskStatusUpdated)
                and event.task_status == TaskStatus.Complete
                for event in sent
            ):
                sent.append(await events.receive())

    async with running_worker(state, planning) as (_worker, _events, events):
        await until_the_download_task_is_done(events)

    reported = [event for event in sent if isinstance(event, NodeDownloadProgress)]
    assert (reported == []) == leaving
