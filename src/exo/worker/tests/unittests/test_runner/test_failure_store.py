import pytest

from exo.shared.models.model_cards import ModelId
from exo.shared.types.common import NodeId
from exo.shared.types.events import Event, RunnerStatusUpdated
from exo.shared.types.worker.instances import BoundInstance, InstanceId
from exo.shared.types.worker.runners import RunnerFailed, RunnerId
from exo.utils.channels import channel
from exo.worker.runner import failure_store
from exo.worker.tests.unittests.conftest import get_bound_mlx_ring_instance


@pytest.mark.asyncio
async def test_replay_persisted_runner_failures_republishes_and_clears(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(failure_store, "EXO_LOG_DIR", tmp_path)

    bound_instance: BoundInstance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-f"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-f"),
        node_id=NodeId("node-f"),
    )
    stored = failure_store.persist_runner_failure(
        bound_instance,
        error_message="persist me",
        source="unit_test",
    )
    assert stored.exists()

    event_sender, event_receiver = channel[Event]()

    replayed = await failure_store.replay_persisted_runner_failures(event_sender)
    event = await event_receiver.receive()

    assert replayed == 1
    assert isinstance(event, RunnerStatusUpdated)
    assert event.runner_id == RunnerId("runner-f")
    assert isinstance(event.runner_status, RunnerFailed)
    assert event.runner_status.error_message == "persist me"
    assert not stored.exists()

    event_sender.close()


@pytest.mark.asyncio
async def test_replay_persisted_runner_failures_quarantines_invalid_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(failure_store, "EXO_LOG_DIR", tmp_path)

    pending_dir = tmp_path / "pending_runner_failures"
    pending_dir.mkdir(parents=True, exist_ok=True)
    invalid_path = pending_dir / "invalid.json"
    invalid_path.write_text('{"runner_id":"runner-x"}')

    event_sender, _ = channel[Event]()

    replayed = await failure_store.replay_persisted_runner_failures(event_sender)

    assert replayed == 0
    assert not invalid_path.exists()
    assert (pending_dir / "invalid.invalid").exists()

    event_sender.close()
