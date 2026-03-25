# pyright: reportPrivateUsage=false
"""Tests for image generation cancellation logic.

Tests the NaN sentinel protocol, cancellation checking, and the
image runner's cancel_checker integration.
"""

from collections.abc import Callable
from unittest.mock import MagicMock

import mlx.core as mx

from exo.shared.types.tasks import CANCEL_ALL_TASKS, TaskId
from exo.worker.engines.image.pipeline.runner import DiffusionRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runner() -> DiffusionRunner:
    """Create a DiffusionRunner with minimal config for unit testing.

    Uses a mock adapter and no distributed group (single-node).
    """
    mock_config = MagicMock()
    mock_config.joint_block_count = 10
    mock_config.single_block_count = 10
    mock_config.total_blocks = 20
    mock_config.guidance_scale = None

    mock_adapter = MagicMock()

    mock_shard = MagicMock()
    mock_shard.device_rank = 0
    mock_shard.world_size = 1
    mock_shard.start_layer = 0
    mock_shard.end_layer = 20

    runner = DiffusionRunner(
        config=mock_config,
        adapter=mock_adapter,
        group=None,
        shard_metadata=mock_shard,
    )
    return runner


class FakeCancelReceiver:
    """Fake MpReceiver that returns pre-loaded items from collect()."""

    def __init__(self, items: list[TaskId] | None = None):
        self._items = list(items) if items else []

    def collect(self) -> list[TaskId]:
        result = self._items
        self._items = []
        return result


class FakeImageRunner:
    """Fake image runner for testing _check_cancelled logic."""

    def __init__(self, cancel_items: list[TaskId] | None = None) -> None:
        self.cancel_receiver = FakeCancelReceiver(cancel_items)
        self.cancelled_tasks = set[TaskId]()

    def _check_cancelled(self, task_id: TaskId) -> bool:
        for cancel_id in self.cancel_receiver.collect():
            self.cancelled_tasks.add(cancel_id)
        return (
            task_id in self.cancelled_tasks or CANCEL_ALL_TASKS in self.cancelled_tasks
        )


# ---------------------------------------------------------------------------
# _is_sentinel
# ---------------------------------------------------------------------------


class TestIsSentinel:
    def test_all_nan_is_sentinel(self) -> None:
        runner = _make_runner()
        tensor = mx.full((2, 3), float("nan"))
        mx.eval(tensor)
        assert runner._is_sentinel(tensor) is True

    def test_all_zeros_is_not_sentinel(self) -> None:
        runner = _make_runner()
        tensor = mx.zeros((2, 3))
        mx.eval(tensor)
        assert runner._is_sentinel(tensor) is False

    def test_mixed_nan_and_real_is_not_sentinel(self) -> None:
        """A tensor with some NaN and some real values must NOT be a sentinel.
        Using mx.any(isnan) would incorrectly flag this as a sentinel.
        """
        runner = _make_runner()
        tensor = mx.array([float("nan"), 1.0, 2.0])
        mx.eval(tensor)
        assert runner._is_sentinel(tensor) is False

    def test_single_element_nan(self) -> None:
        runner = _make_runner()
        tensor = mx.array([float("nan")])
        mx.eval(tensor)
        assert runner._is_sentinel(tensor) is True

    def test_large_tensor_all_nan(self) -> None:
        runner = _make_runner()
        tensor = mx.full((64, 128, 32), float("nan"))
        mx.eval(tensor)
        assert runner._is_sentinel(tensor) is True

    def test_real_data_not_sentinel(self) -> None:
        runner = _make_runner()
        tensor = mx.random.normal((4, 8))
        mx.eval(tensor)
        assert runner._is_sentinel(tensor) is False


# ---------------------------------------------------------------------------
# _check_cancellation
# ---------------------------------------------------------------------------


class TestCheckCancellation:
    def test_first_stage_polls_checker(self) -> None:
        runner = _make_runner()
        assert runner.is_first_stage  # single-node is always first stage

        checker: Callable[[], bool] = MagicMock(return_value=True)
        runner._cancel_checker = checker

        runner._check_cancellation()

        checker.assert_called_once()
        assert runner._cancelling is True

    def test_checker_returning_false_does_not_cancel(self) -> None:
        runner = _make_runner()
        checker: Callable[[], bool] = MagicMock(return_value=False)
        runner._cancel_checker = checker

        runner._check_cancellation()

        assert runner._cancelling is False

    def test_no_checker_does_not_cancel(self) -> None:
        runner = _make_runner()
        runner._cancel_checker = None

        runner._check_cancellation()

        assert runner._cancelling is False

    def test_already_cancelling_skips_checker(self) -> None:
        runner = _make_runner()
        runner._cancelling = True
        checker: Callable[[], bool] = MagicMock(return_value=False)
        runner._cancel_checker = checker

        runner._check_cancellation()

        checker.assert_not_called()
        assert runner._cancelling is True  # stays True

    def test_cancelling_flag_is_false_on_init(self) -> None:
        """_cancelling defaults to False on a fresh runner."""
        runner = _make_runner()
        assert runner._cancelling is False


# ---------------------------------------------------------------------------
# _send wrapper
# ---------------------------------------------------------------------------


class TestSendWrapper:
    def test_send_replaces_data_with_nan_when_cancelling(self) -> None:
        """When _cancelling is True, _send should replace data with NaN."""
        runner = _make_runner()
        runner._cancelling = True
        # _send asserts group is not None, so we need a mock group
        runner.group = MagicMock()

        data = mx.ones((2, 3))
        mx.eval(data)

        # Mock mx.distributed.send to capture what's sent
        original_send = mx.distributed.send
        sent_data: list[mx.array] = []

        def mock_send(d: mx.array, dst: int, group: mx.distributed.Group) -> mx.array:
            mx.eval(d)
            sent_data.append(d)
            return d

        mx.distributed.send = mock_send
        try:
            runner._send(data, dst=1)
            assert len(sent_data) == 1
            mx.eval(sent_data[0])
            assert mx.all(mx.isnan(sent_data[0])).item()
            assert sent_data[0].shape == (2, 3)
        finally:
            mx.distributed.send = original_send

    def test_send_passes_real_data_when_not_cancelling(self) -> None:
        runner = _make_runner()
        runner._cancelling = False
        runner.group = MagicMock()

        data = mx.ones((2, 3))
        mx.eval(data)

        sent_data: list[mx.array] = []

        def mock_send(d: mx.array, dst: int, group: mx.distributed.Group) -> mx.array:
            mx.eval(d)
            sent_data.append(d)
            return d

        original_send = mx.distributed.send
        mx.distributed.send = mock_send
        try:
            runner._send(data, dst=1)
            assert len(sent_data) == 1
            mx.eval(sent_data[0])
            assert not mx.any(mx.isnan(sent_data[0])).item()
        finally:
            mx.distributed.send = original_send


# ---------------------------------------------------------------------------
# Image runner _check_cancelled
# ---------------------------------------------------------------------------


class TestImageRunnerCheckCancelled:
    """Tests for the image runner's _check_cancelled method."""

    def test_no_cancellation(self) -> None:
        runner = FakeImageRunner()
        assert runner._check_cancelled(TaskId("task-1")) is False

    def test_specific_task_cancelled(self) -> None:
        task_id = TaskId("task-1")
        runner = FakeImageRunner([task_id])
        assert runner._check_cancelled(task_id) is True

    def test_different_task_not_cancelled(self) -> None:
        runner = FakeImageRunner([TaskId("task-2")])
        assert runner._check_cancelled(TaskId("task-1")) is False

    def test_cancel_all_tasks(self) -> None:
        runner = FakeImageRunner([CANCEL_ALL_TASKS])
        assert runner._check_cancelled(TaskId("any-task")) is True

    def test_collect_accumulates(self) -> None:
        """Multiple collect() calls accumulate cancelled task IDs."""
        runner = FakeImageRunner([TaskId("task-1")])
        runner._check_cancelled(TaskId("task-1"))

        # First collect drained the receiver, but task-1 is in cancelled_tasks
        assert runner._check_cancelled(TaskId("task-1")) is True

    def test_collect_empty_after_drain(self) -> None:
        """After draining, collect returns empty and previous cancellations persist."""
        runner = FakeImageRunner([TaskId("task-1")])

        # First call drains
        runner._check_cancelled(TaskId("other"))
        # task-1 is now in cancelled_tasks but "other" was never cancelled
        assert runner._check_cancelled(TaskId("other")) is False
        assert runner._check_cancelled(TaskId("task-1")) is True


# ---------------------------------------------------------------------------
# Drain condition logic
# ---------------------------------------------------------------------------


class TestDrainCondition:
    """Verify the drain condition evaluates correctly for various scenarios."""

    def _should_drain(
        self,
        *,
        cancelling: bool,
        is_first_stage: bool,
        is_last_stage: bool,
        is_distributed: bool,
        t: int,
        init_time_step: int,
        num_sync_steps: int,
        num_inference_steps: int,
    ) -> bool:
        """Replicate the drain condition from _run_diffusion_loop."""
        return (
            cancelling
            and is_first_stage
            and not is_last_stage
            and is_distributed
            and t >= init_time_step + num_sync_steps
            and t != num_inference_steps - 1
        )

    def test_no_drain_during_sync_step(self) -> None:
        """Sync steps have no cross-timestep ring state."""
        assert not self._should_drain(
            cancelling=True,
            is_first_stage=True,
            is_last_stage=False,
            is_distributed=True,
            t=0,  # sync step
            init_time_step=0,
            num_sync_steps=2,
            num_inference_steps=10,
        )

    def test_drain_during_async_step(self) -> None:
        assert self._should_drain(
            cancelling=True,
            is_first_stage=True,
            is_last_stage=False,
            is_distributed=True,
            t=3,  # async step
            init_time_step=0,
            num_sync_steps=2,
            num_inference_steps=10,
        )

    def test_no_drain_on_last_step(self) -> None:
        """Last step doesn't send, so nothing to drain."""
        assert not self._should_drain(
            cancelling=True,
            is_first_stage=True,
            is_last_stage=False,
            is_distributed=True,
            t=9,  # last step
            init_time_step=0,
            num_sync_steps=2,
            num_inference_steps=10,
        )

    def test_no_drain_when_not_cancelling(self) -> None:
        assert not self._should_drain(
            cancelling=False,
            is_first_stage=True,
            is_last_stage=False,
            is_distributed=True,
            t=5,
            init_time_step=0,
            num_sync_steps=2,
            num_inference_steps=10,
        )

    def test_no_drain_on_last_stage(self) -> None:
        """Last stage is also first stage (single pipeline) — no ring."""
        assert not self._should_drain(
            cancelling=True,
            is_first_stage=True,
            is_last_stage=True,
            is_distributed=True,
            t=5,
            init_time_step=0,
            num_sync_steps=2,
            num_inference_steps=10,
        )

    def test_no_drain_single_node(self) -> None:
        assert not self._should_drain(
            cancelling=True,
            is_first_stage=True,
            is_last_stage=False,
            is_distributed=False,
            t=5,
            init_time_step=0,
            num_sync_steps=2,
            num_inference_steps=10,
        )

    def test_no_drain_not_first_stage(self) -> None:
        """Only first stage needs to drain (it's the one receiving)."""
        assert not self._should_drain(
            cancelling=True,
            is_first_stage=False,
            is_last_stage=False,
            is_distributed=True,
            t=5,
            init_time_step=0,
            num_sync_steps=2,
            num_inference_steps=10,
        )

    def test_drain_first_async_step(self) -> None:
        """First async step: last stage sends, so drain is needed."""
        assert self._should_drain(
            cancelling=True,
            is_first_stage=True,
            is_last_stage=False,
            is_distributed=True,
            t=2,  # first async step (init=0, sync=2)
            init_time_step=0,
            num_sync_steps=2,
            num_inference_steps=10,
        )

    def test_drain_with_nonzero_init_time_step(self) -> None:
        """img2img can have init_time_step > 0."""
        assert self._should_drain(
            cancelling=True,
            is_first_stage=True,
            is_last_stage=False,
            is_distributed=True,
            t=5,
            init_time_step=3,
            num_sync_steps=1,
            num_inference_steps=10,
        )

    def test_no_drain_sync_with_nonzero_init(self) -> None:
        assert not self._should_drain(
            cancelling=True,
            is_first_stage=True,
            is_last_stage=False,
            is_distributed=True,
            t=3,
            init_time_step=3,
            num_sync_steps=1,
            num_inference_steps=10,
        )
