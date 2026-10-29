import asyncio
import time

import pytest

from exo.master.volunteer import (
    VolunteerGridScheduler,
    VolunteerNodeHealthMonitor,
)
from exo.shared.types.volunteer_grid import (
    BatchInferenceJob,
    VolunteerNodeProfile,
)


def test_volunteer_node_profile_properties() -> None:
    node = VolunteerNodeProfile(
        node_id="node-1",
        reliability_score=0.9,
        status="healthy",
    )
    assert node.is_available_for_dispatch

    unreliable_node = VolunteerNodeProfile(
        node_id="node-2",
        reliability_score=0.1,
        status="healthy",
    )
    assert not unreliable_node.is_available_for_dispatch

    offline_node = VolunteerNodeProfile(
        node_id="node-3",
        reliability_score=0.9,
        status="offline",
    )
    assert not offline_node.is_available_for_dispatch


def test_health_monitor_reliability_and_timeouts() -> None:
    monitor = VolunteerNodeHealthMonitor(default_heartbeat_timeout_seconds=5.0)
    node = VolunteerNodeProfile(node_id="node-a", reliability_score=0.5, estimated_tps=10.0)
    monitor.register_node(node)

    # 1. Job success rewards reliability and updates TPS
    monitor.record_job_success(node_id="node-a", elapsed_seconds=2.0, tokens_processed=40)
    updated = monitor.get_node("node-a")
    assert updated is not None
    assert updated.reliability_score == 0.6  # 0.5 + 0.1
    assert updated.total_jobs_completed == 1
    assert updated.consecutive_timeouts == 0

    # 2. Timeout drops reliability score
    monitor.record_job_timeout("node-a")
    updated = monitor.get_node("node-a")
    assert updated is not None
    assert updated.consecutive_timeouts == 1
    assert updated.status == "healthy"

    # Second timeout makes node a straggler
    monitor.record_job_timeout("node-a")
    updated = monitor.get_node("node-a")
    assert updated is not None
    assert updated.consecutive_timeouts == 2
    assert updated.status == "straggler"

    # Heartbeat updates timestamp
    monitor.record_heartbeat("node-a")
    updated = monitor.get_node("node-a")
    assert updated is not None
    assert updated.status == "straggler"


def test_health_monitor_ranking_and_staleness() -> None:
    monitor = VolunteerNodeHealthMonitor(default_heartbeat_timeout_seconds=0.1)
    n1 = VolunteerNodeProfile(node_id="n1", reliability_score=0.9, estimated_tps=20.0)
    n2 = VolunteerNodeProfile(node_id="n2", reliability_score=0.7, estimated_tps=30.0)
    monitor.register_node(n1)
    monitor.register_node(n2)

    ranked = monitor.get_ranked_available_nodes()
    assert [n.node_id for n in ranked] == ["n1", "n2"]

    # Sleep past heartbeat timeout to trigger offline transition
    time.sleep(0.15)
    ranked_after = monitor.get_ranked_available_nodes()
    assert len(ranked_after) == 0


@pytest.mark.asyncio
async def test_scheduler_happy_path() -> None:
    monitor = VolunteerNodeHealthMonitor()
    n1 = VolunteerNodeProfile(node_id="fast-node", reliability_score=1.0)
    monitor.register_node(n1)

    scheduler = VolunteerGridScheduler(health_monitor=monitor)
    job = BatchInferenceJob(
        job_id="job-1",
        prompt_chunks=("Summarize this document paragraph.",),
        priority="batch_normal",
    )
    scheduler.submit_job(job)
    assert scheduler.pending_jobs_count() == 1

    async def dummy_executor(node_id: str, j: BatchInferenceJob) -> str:
        await asyncio.sleep(0.01)
        return f"Summary from {node_id}"

    res = await scheduler.execute_job_speculative(job, dummy_executor)
    assert res == "Summary from fast-node"

    tel = scheduler.get_telemetry()
    assert tel.jobs_completed == 1
    assert tel.total_jobs_dispatched == 1
    assert tel.stragglers_mitigated == 0
    assert tel.redundant_jobs_cancelled == 0


@pytest.mark.asyncio
async def test_scheduler_speculative_straggler_mitigation() -> None:
    monitor = VolunteerNodeHealthMonitor()
    # n1 is slow, n2 is fast
    n1 = VolunteerNodeProfile(node_id="slow-node", reliability_score=0.9)
    n2 = VolunteerNodeProfile(node_id="fast-node", reliability_score=0.85)
    monitor.register_node(n1)
    monitor.register_node(n2)

    scheduler = VolunteerGridScheduler(health_monitor=monitor)
    job = BatchInferenceJob(
        job_id="job-straggler",
        prompt_chunks=("Analyze code.",),
        timeout_seconds=0.2,  # Threshold ~ 0.08s
        redundancy_factor=2,
    )

    async def slow_or_fast_executor(node_id: str, j: BatchInferenceJob) -> str:
        if node_id == "slow-node":
            await asyncio.sleep(0.3)  # Exceeds threshold and runs slow
            return "Late result"
        else:
            await asyncio.sleep(0.01)  # Backup finishes quickly
            return "Fast result"

    res = await scheduler.execute_job_speculative(job, slow_or_fast_executor)
    assert res == "Fast result"

    tel = scheduler.get_telemetry()
    assert tel.jobs_completed == 1
    assert tel.stragglers_mitigated == 1
    assert tel.redundant_jobs_cancelled == 1


@pytest.mark.asyncio
async def test_scheduler_node_failure_recovery() -> None:
    monitor = VolunteerNodeHealthMonitor()
    n1 = VolunteerNodeProfile(node_id="failing-node", reliability_score=0.95)
    n2 = VolunteerNodeProfile(node_id="healthy-node", reliability_score=0.90)
    monitor.register_node(n1)
    monitor.register_node(n2)

    scheduler = VolunteerGridScheduler(health_monitor=monitor)
    job = BatchInferenceJob(
        job_id="job-failover",
        prompt_chunks=("Translate text.",),
        timeout_seconds=0.1,
        redundancy_factor=2,
    )

    async def failing_executor(node_id: str, j: BatchInferenceJob) -> str:
        if node_id == "failing-node":
            raise RuntimeError("Out of memory on volunteer phone")
        await asyncio.sleep(0.01)
        return "Recovered result"

    res = await scheduler.execute_job_speculative(job, failing_executor)
    assert res == "Recovered result"

    # Verify failing node was penalized
    penalized = monitor.get_node("failing-node")
    assert penalized is not None
    assert penalized.consecutive_timeouts >= 1
