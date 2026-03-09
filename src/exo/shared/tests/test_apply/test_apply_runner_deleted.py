from exo.shared.apply import apply_runner_status_updated
from exo.shared.types.events import RunnerStatusUpdated
from exo.shared.types.state import State
from exo.shared.types.worker.runners import RunnerId, RunnerIdle, RunnerShutdown


def test_apply_runner_shutdown_removes_runner():
    runner_id = RunnerId()
    state = State(runners={runner_id: RunnerIdle()})

    new_state = apply_runner_status_updated(
        RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerShutdown()), state
    )

    assert runner_id not in new_state.runners


def test_apply_runner_status_updated_adds_runner():
    runner_id = RunnerId()
    state = State()

    new_state = apply_runner_status_updated(
        RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerIdle()), state
    )

    assert runner_id in new_state.runners
