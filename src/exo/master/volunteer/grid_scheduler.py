import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable

from exo.master.volunteer.health_monitor import VolunteerNodeHealthMonitor
from exo.shared.types.volunteer_grid import (
    BatchInferenceJob,
    GridDispatchTelemetry,
)


class VolunteerGridScheduler:
    """Schedules batch inference across unreliable edge volunteer nodes with speculative dispatch."""

    def __init__(self, health_monitor: VolunteerNodeHealthMonitor) -> None:
        self._health_monitor = health_monitor
        self._interactive_queue: list[BatchInferenceJob] = []
        self._batch_queue: list[BatchInferenceJob] = []

        # Telemetry counters
        self._total_dispatched: int = 0
        self._jobs_completed: int = 0
        self._stragglers_mitigated: int = 0
        self._redundant_cancelled: int = 0
        self._accumulated_tokens: int = 0
        self._accumulated_seconds: float = 0.0

    def submit_job(self, job: BatchInferenceJob) -> None:
        """Submits a job to the corresponding priority queue."""
        if job.priority == "interactive":
            self._interactive_queue.append(job)
        else:
            self._batch_queue.append(job)

    def pending_jobs_count(self) -> int:
        return len(self._interactive_queue) + len(self._batch_queue)

    async def execute_job_speculative(
        self,
        job: BatchInferenceJob,
        node_executor_fn: Callable[[str, BatchInferenceJob], Awaitable[str]],
    ) -> str:
        """Executes a job across volunteer nodes with speculative multi-path dispatch and cancellation.

        Args:
            job: The batch job to execute.
            node_executor_fn: Asynchronous coroutine taking (node_id, job) and returning the inference text.

        Returns:
            The output string of the first successful replica.
        """
        available_nodes = self._health_monitor.get_ranked_available_nodes()
        if not available_nodes:
            raise RuntimeError("No available volunteer nodes in the grid to execute job")

        # Select primary candidate
        primary_node = available_nodes[0]
        self._total_dispatched += 1

        active_tasks: dict[asyncio.Task, str] = {}
        t_start = time.perf_counter()

        # Launch primary execution
        primary_task = asyncio.create_task(
            node_executor_fn(primary_node.node_id, job),
            name=f"job-{job.job_id}-node-{primary_node.node_id}",
        )
        active_tasks[primary_task] = primary_node.node_id

        # Wait for completion or early straggler threshold
        straggler_threshold = max(0.05, job.timeout_seconds * 0.4)
        done, _ = await asyncio.wait(active_tasks.keys(), timeout=straggler_threshold)

        primary_succeeded = False
        if done:
            task = next(iter(done))
            if not task.cancelled() and task.exception() is None:
                primary_succeeded = True

        dispatched_node_ids: set[str] = {primary_node.node_id}

        # If primary didn't succeed within threshold and we have backup nodes, speculatively dispatch replica
        if not primary_succeeded and len(available_nodes) > 1 and job.redundancy_factor > 1:
            backup_node = available_nodes[1]
            dispatched_node_ids.add(backup_node.node_id)
            self._stragglers_mitigated += 1
            self._total_dispatched += 1
            backup_task = asyncio.create_task(
                node_executor_fn(backup_node.node_id, job),
                name=f"job-{job.job_id}-node-{backup_node.node_id}",
            )
            active_tasks[backup_task] = backup_node.node_id

        # Wait for the first task across all replicas to succeed
        while active_tasks:
            done, _ = await asyncio.wait(
                active_tasks.keys(),
                return_when=asyncio.FIRST_COMPLETED,
            )

            for completed_task in done:
                node_id = active_tasks.pop(completed_task)
                try:
                    result = completed_task.result()
                    elapsed = time.perf_counter() - t_start
                    tokens = sum(len(c.split()) for c in job.prompt_chunks) or 10

                    # Record success on winning node
                    self._health_monitor.record_job_success(
                        node_id=node_id,
                        elapsed_seconds=elapsed,
                        tokens_processed=tokens,
                    )
                    self._jobs_completed += 1
                    self._accumulated_tokens += tokens
                    self._accumulated_seconds += elapsed

                    # Cancel all other redundant speculative replicas immediately
                    for remaining_task in active_tasks:
                        remaining_task.cancel()
                        self._redundant_cancelled += 1

                    return result
                except (RuntimeError, TimeoutError, asyncio.CancelledError, OSError, ValueError):
                    # Record failure / timeout on failed node
                    self._health_monitor.record_job_timeout(node_id)

            if not active_tasks:
                # Failover to remaining untried available nodes if any
                untried = [
                    n
                    for n in self._health_monitor.get_ranked_available_nodes()
                    if n.node_id not in dispatched_node_ids
                ]
                if untried:
                    next_node = untried[0]
                    dispatched_node_ids.add(next_node.node_id)
                    self._total_dispatched += 1
                    next_task = asyncio.create_task(
                        node_executor_fn(next_node.node_id, job),
                        name=f"job-{job.job_id}-node-{next_node.node_id}",
                    )
                    active_tasks[next_task] = next_node.node_id
                else:
                    break

        raise TimeoutError(f"Job {job.job_id} failed across all assigned volunteer nodes")

    def get_telemetry(self) -> GridDispatchTelemetry:
        """Retrieves aggregated performance and resiliency telemetry of the volunteer grid."""
        aggregate_tps = (
            (self._accumulated_tokens / self._accumulated_seconds)
            if self._accumulated_seconds > 0
            else 0.0
        )
        return GridDispatchTelemetry(
            telemetry_id=f"telemetry-{uuid.uuid4().hex[:8]}",
            total_jobs_dispatched=self._total_dispatched,
            jobs_completed=self._jobs_completed,
            stragglers_mitigated=self._stragglers_mitigated,
            redundant_jobs_cancelled=self._redundant_cancelled,
            aggregate_tps=aggregate_tps,
        )
