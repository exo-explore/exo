"""Time budget iterator for controlling generation loop timing in distributed mode.

Based on mlx-lm's TimeBudget pattern - runs for a time budget then syncs,
rather than syncing every token. This reduces distributed sync overhead.
"""

import time
from typing import Iterator

import mlx.core as mx

from exo.worker.runner.bootstrap import logger

generation_stream = mx.new_stream(mx.default_device())


class TimeBudget(Iterator[None]):
    """Controls generation loop timing, syncing across ranks periodically.

    In distributed mode, periodically syncs timing across all ranks to
    dynamically adjust iteration count based on actual performance.

    In non-distributed mode, simply runs for the time budget.

    Usage:
        for _ in TimeBudget(budget=0.5):
            batch_engine.step()
            # ... process responses ...
    """

    def __init__(
        self,
        budget: float = 0.5,
        iterations: int = 25,
        sync_frequency: int = 10,
        group: mx.distributed.Group | None = None,
    ):
        """Initialize TimeBudget.

        Args:
            budget: Time budget in seconds before yielding control
            iterations: Initial number of iterations per budget period (distributed only)
            sync_frequency: How often to sync timing across ranks (distributed only)
            group: Distributed group, or None for non-distributed mode
        """
        self._budget = budget
        self._iterations = iterations
        self._sync_frequency = sync_frequency
        self._group = group
        self._is_distributed = group is not None and group.size() > 1

        # Runtime state
        self._start: float = 0.0
        self._current_iterations: int = 0
        self._loops: int = 0
        self._time_spent: float = 0.0

    def __iter__(self) -> "TimeBudget":
        self._start = time.perf_counter()
        self._current_iterations = 0
        return self

    def __next__(self) -> None:
        if not self._is_distributed:
            # Non-distributed: just check time budget
            if time.perf_counter() - self._start > self._budget:
                raise StopIteration()
            return None

        # Distributed mode: iteration-based with periodic timing sync
        self._current_iterations += 1
        if self._current_iterations > self._iterations:
            self._loops += 1
            self._time_spent += time.perf_counter() - self._start

            if self._loops % self._sync_frequency == 0:
                # Sync timing across all ranks
                assert self._group is not None
                with mx.stream(generation_stream):
                    time_array = mx.array([self._time_spent], dtype=mx.float32)
                    total_time = mx.distributed.all_sum(time_array, group=self._group)
                    mx.eval(total_time)
                    loop_time = float(total_time.item())

                avg_loop_time = loop_time / (self._group.size() * self._sync_frequency)

                if avg_loop_time > 0:
                    factor = self._budget / avg_loop_time
                    self._iterations = max(round(self._iterations * factor), 1)
                    logger.debug(
                        f"TimeBudget adjusted iterations to {self._iterations}"
                    )

                self._loops = 0
                self._time_spent = 0.0

            raise StopIteration()

        return None

    @property
    def iterations(self) -> int:
        """Current iterations per budget period."""
        return self._iterations
